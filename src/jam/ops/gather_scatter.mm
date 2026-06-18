// ops/gather_scatter.mm — single-axis gather/scatter, sort (single key or key+iota), chlo.top_k.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// Windowed gather (vmap(dynamic_slice) / sliding window): every operand dim is indexed with a window
// (slice_size >= 1), no collapse, no operand batching. MPSGraph has no windowed gather, so lower to a
// point gatherND over explicit coordinates: for output dim of operand-axis d, coord = clamp(start_d,
// 0, dim_d - slice_d) + iota_along_window. Build each coordinate at the output shape, stack along a
// trailing axis (depth == rank), gatherND.
void WindowedGather(Lowering& L, mlir::Operation* op) {
  auto g = mlir::cast<mlir::stablehlo::GatherOp>(op);
  auto dn = g.getDimensionNumbers();
  llvm::ArrayRef<int64_t> startMap = dn.getStartIndexMap();
  llvm::ArrayRef<int64_t> offsetDims = dn.getOffsetDims();
  int64_t ivd = dn.getIndexVectorDim();
  llvm::ArrayRef<int64_t> sliceSizes = g.getSliceSizes();
  llvm::ArrayRef<int64_t> opShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  int64_t R = static_cast<int64_t>(opShape.size());
  llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  int64_t idxRank = static_cast<int64_t>(ish.size());
  llvm::ArrayRef<int64_t> outDims = OutShape(op);
  int64_t outRank = static_cast<int64_t>(outDims.size());
  int64_t numBatch = idxRank - 1;

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* idx = L.value(op->getOperand(1));
  std::vector<int64_t> idxPerm;  // move index-vector dim last: indices = [batch..., depth]
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);
  idx = Casted(L, Transposed(L, idx, idxPerm), MPSDataTypeInt32);

  auto isOffset = [&](int64_t ax) { for (int64_t o : offsetDims) if (o == ax) return true; return false; };
  std::vector<MPSGraphTensor*> coord(R, nullptr);
  for (int64_t k = 0; k < R; ++k) {
    int64_t d = startMap[k];           // operand dim this index component addresses
    int64_t winAxis = offsetDims[d];   // output axis holding d's window (offset_dims indexed by operand dim)
    // start component k: idx[..., k] reshaped into the output shape (1 at offset axes, batch sizes else)
    MPSGraphTensor* sc = [L.graph() sliceTensor:idx dimension:(NSUInteger)numBatch start:(NSInteger)k length:1 name:nil];
    std::vector<int64_t> rO; int64_t bi = 0;
    for (int64_t ax = 0; ax < outRank; ++ax) rO.push_back(isOffset(ax) ? 1 : ish[idxPerm[bi++]]);
    sc = [L.graph() reshapeTensor:sc withShape:IntArray(rO) name:nil];
    // clamp start to [0, dim_d - slice_d] so the window fits (XLA windowed-gather clamp)
    MPSGraphTensor* hi = [L.graph() constantWithScalar:(double)(opShape[d] - sliceSizes[d]) dataType:MPSDataTypeInt32];
    MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeInt32];
    sc = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:sc secondaryTensor:hi name:nil]
                               secondaryTensor:lo name:nil];
    // coord = broadcast(start) + window iota along winAxis
    MPSGraphTensor* it = Casted(L, [L.graph() coordinateAlongAxis:(NSInteger)winAxis withShape:ShapeArray(outDims) name:nil], MPSDataTypeInt32);
    MPSGraphTensor* c = [L.graph() additionWithPrimaryTensor:sc secondaryTensor:it name:nil];
    std::vector<int64_t> oc(outDims.begin(), outDims.end()); oc.push_back(1);  // trailing 1 for stacking
    coord[d] = [L.graph() reshapeTensor:c withShape:IntArray(oc) name:nil];
  }
  NSMutableArray<MPSGraphTensor*>* arr = [NSMutableArray array];
  for (int64_t d = 0; d < R; ++d) [arr addObject:coord[d]];  // operand-dim order = gatherND index order
  MPSGraphTensor* coords = [L.graph() concatTensors:arr dimension:(NSInteger)outRank name:nil];
  MPSGraphTensor* gathered = [L.graph() gatherNDWithUpdatesTensor:data indicesTensor:coords batchDimensions:0 name:nil];
  Set(L, op, Reshaped(L, gathered, outDims));
}

// General multi-axis "advanced indexing" gather (x[rows,cols], diagonal, y[bi,:,ki]) → MPSGraph
// gatherND. Point-indexing on >=2 operand dims (collapsed == start_index_map, slice size 1 there),
// no operand batching; full-slice ("offset") dims ride along. Transposes the indexed dims to the
// front, gatherND over them, maps the result back to the StableHLO output order. Falls back to fail.
void GatherND(Lowering& L, mlir::Operation* op) {
  auto g = mlir::cast<mlir::stablehlo::GatherOp>(op);
  auto dn = g.getDimensionNumbers();
  llvm::ArrayRef<int64_t> collapsed = dn.getCollapsedSliceDims();
  llvm::ArrayRef<int64_t> startMap = dn.getStartIndexMap();
  llvm::ArrayRef<int64_t> offsetDims = dn.getOffsetDims();
  int64_t ivd = dn.getIndexVectorDim();
  llvm::ArrayRef<int64_t> sliceSizes = g.getSliceSizes();
  if (!dn.getOperandBatchingDims().empty()) { L.fail("jam: gather: batched general gather unsupported"); return; }
  llvm::ArrayRef<int64_t> opShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  int64_t R = static_cast<int64_t>(opShape.size());
  int64_t depth = static_cast<int64_t>(startMap.size());
  auto inStartMap = [&](int64_t d) { for (int64_t k = 0; k < depth; ++k) if (startMap[k] == d) return true; return false; };
  auto inCollapsed = [&](int64_t d) { for (int64_t c : collapsed) if (c == d) return true; return false; };
  for (int64_t d : collapsed) if (!inStartMap(d)) { L.fail("jam: gather: collapsed dim is not indexed"); return; }
  // Windowed gather (some indexed dim has slice_size>1), every operand dim indexed, no collapse →
  // WindowedGather (coordinate expansion). Point gathers keep the path below.
  bool windowed = false;
  for (int64_t k = 0; k < depth; ++k) if (sliceSizes[startMap[k]] > 1) windowed = true;
  if (windowed && collapsed.empty() && depth == R && static_cast<int64_t>(offsetDims.size()) == R) {
    WindowedGather(L, op);
    return;
  }
  // Point gather only: every indexed dim has slice size 1, every other dim is a full slice. True
  // windowed gathers (slice>1 on an indexed dim) are not expressible via gatherND → clean failure.
  for (unsigned i = 0; i < sliceSizes.size(); ++i)
    if (sliceSizes[i] != (inStartMap(i) ? 1 : opShape[i])) { L.fail("jam: gather: windowed (slice>1) gather unsupported"); return; }

  llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  int64_t idxRank = static_cast<int64_t>(ish.size());
  if (ivd < 0 || ivd >= idxRank || ish[ivd] != depth) { L.fail("jam: gather: general gather index vector unsupported"); return; }

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* idx = L.value(op->getOperand(1));
  // Transpose operand so the indexed dims lead (start_index_map order), then the non-indexed dims.
  std::vector<int64_t> opPerm;
  for (int64_t k = 0; k < depth; ++k) opPerm.push_back(startMap[k]);
  for (int64_t d = 0; d < R; ++d) if (!inStartMap(d)) opPerm.push_back(d);
  data = Transposed(L, data, opPerm);
  // Output "offset" dims = every operand dim that isn't collapsed (ascending): the full-slice
  // non-indexed dims plus any indexed-but-not-collapsed dims (which become size-1, e.g. searchsorted).
  std::vector<int64_t> rest;
  for (int64_t d = 0; d < R; ++d) if (!inCollapsed(d)) rest.push_back(d);
  // Arrange indices to [batch..., depth] (index vector last).
  std::vector<int64_t> idxPerm;
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);
  idx = Casted(L, Transposed(L, idx, idxPerm), MPSDataTypeInt32);
  // Per-component clamp to [0, dim-1] (XLA clamps OOB; MPSGraph zero-fills). hi broadcasts over batch.
  std::vector<int32_t> hi32(depth);
  for (int64_t k = 0; k < depth; ++k) hi32[k] = (int32_t)(opShape[startMap[k]] - 1);
  NSData* hd = [NSData dataWithBytes:hi32.data() length:depth * sizeof(int32_t)];
  MPSGraphTensor* hi = [L.graph() constantWithData:hd shape:@[@(depth)] dataType:MPSDataTypeInt32];
  MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeInt32];
  idx = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:idx secondaryTensor:hi name:nil]
                             secondaryTensor:lo name:nil];
  MPSGraphTensor* gathered = [L.graph() gatherNDWithUpdatesTensor:data indicesTensor:idx batchDimensions:0 name:nil];
  // gatherND collapsed ALL indexed dims → [batch (indices dims sans ivd, in order), non-indexed dims
  // ascending]. Re-insert size-1 dims for the indexed-but-not-collapsed dims (reshape to mShape, whose
  // rest entry per dim is its slice size: 1 for an indexed dim, full for a non-indexed dim), then map
  // to StableHLO output order: rest dims go to offset_dims positions (ascending), batch to the others.
  int64_t numBatch = idxRank - 1, outRank = numBatch + static_cast<int64_t>(rest.size());
  std::vector<int64_t> mShape;
  for (int64_t bd = 0; bd < numBatch; ++bd) mShape.push_back(ish[idxPerm[bd]]);
  for (int64_t d : rest) mShape.push_back(sliceSizes[d]);
  MPSGraphTensor* m = Reshaped(L, gathered, mShape);
  auto inOffset = [&](int64_t p) { for (int64_t o : offsetDims) if (o == p) return true; return false; };
  std::vector<int64_t> outPerm(outRank, 0);
  for (int64_t p = 0, bc = 0, oc = 0; p < outRank; ++p) outPerm[p] = inOffset(p) ? numBatch + (oc++) : bc++;
  Set(L, op, Reshaped(L, Transposed(L, m, outPerm), OutShape(op)));
}

// gather: single-axis take/embedding only (one collapsed dim, full slices elsewhere).
void Gather(Lowering& L, mlir::Operation* op) {
  auto g = mlir::cast<mlir::stablehlo::GatherOp>(op);
  auto dn = g.getDimensionNumbers();
  llvm::ArrayRef<int64_t> collapsed = dn.getCollapsedSliceDims();
  llvm::ArrayRef<int64_t> startMap = dn.getStartIndexMap();
  llvm::ArrayRef<int64_t> obd = dn.getOperandBatchingDims();
  llvm::ArrayRef<int64_t> sibd = dn.getStartIndicesBatchingDims();
  int64_t ivd = dn.getIndexVectorDim();
  llvm::ArrayRef<int64_t> sliceSizes = g.getSliceSizes();

  // Single-axis take/embedding/take_along (the one indexed dim is collapsed) uses the fast path below;
  // everything else — multi-axis advanced indexing, keep-dim point gathers (searchsorted/digitize) —
  // goes to GatherND (which fails clean on true windowed slice>1 gathers).
  if (!(collapsed.size() == 1 && startMap.size() == 1 && collapsed[0] == startMap[0])) {
    GatherND(L, op);
    return;
  }
  int64_t b = static_cast<int64_t>(obd.size());
  if (static_cast<int64_t>(sibd.size()) != b) { L.fail("jam: gather: mismatched batch dims"); return; }
  int64_t axis = startMap[0];
  llvm::ArrayRef<int64_t> operandShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  int64_t R = static_cast<int64_t>(operandShape.size());

  auto isBatchOperand = [&](int64_t d) {
    for (int64_t k = 0; k < b; ++k) if (obd[k] == d) return true;
    return false;
  };
  if (isBatchOperand(axis)) { L.fail("jam: gather: gather axis within batch dims"); return; }
  // Slice sizes: 1 on the gather axis and every batch dim, full elsewhere ("rest"/offset dims).
  for (unsigned i = 0; i < sliceSizes.size(); ++i) {
    bool one = (static_cast<int64_t>(i) == axis) || isBatchOperand(i);
    int64_t want = one ? 1 : operandShape[i];
    if (sliceSizes[i] != want) { L.fail("jam: gather: non-full slice (general gather) unsupported"); return; }
  }

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* idx = L.value(op->getOperand(1));
  // StableHLO indices carry a size-1 index-vector dim; drop it.
  llvm::ArrayRef<int64_t> ishape = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  bool ivdSqueezed = (ivd >= 0 && ivd < static_cast<int64_t>(ishape.size()) && ishape[ivd] == 1);
  if (ivdSqueezed) {
    std::vector<int64_t> squeezed;
    for (unsigned i = 0; i < ishape.size(); ++i)
      if (static_cast<int64_t>(i) != ivd) squeezed.push_back(ishape[i]);
    if (squeezed.empty()) squeezed.push_back(1);
    idx = Reshaped(L, idx, squeezed);
  }
  // XLA clamps gather start-indices in-bounds; MPSGraph fills 0 for OOB, so clamp to [0, dim-1].
  MPSGraphTensor* hi = [L.graph() constantWithScalar:(double)(operandShape[axis] - 1) dataType:idx.dataType];
  MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:idx.dataType];
  idx = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:idx secondaryTensor:hi name:nil]
                             secondaryTensor:lo name:nil];

  // Fast path: batch dims are the leading dims [0..b) on both operand and indices, gather axis after
  // them (take_along_axis on a leading axis, embedding when b==0). MPSGraph batched gather needs no
  // transpose — dims between b and axis and after axis ride along, then a final reshape to OutShape.
  bool leading = (axis >= b);
  for (int64_t k = 0; leading && k < b; ++k)
    if (obd[k] != k || sibd[k] != k) leading = false;
  if (leading) {
    MPSGraphTensor* gathered = [L.graph() gatherWithUpdatesTensor:data
                                                    indicesTensor:idx
                                                             axis:(NSUInteger)axis
                                                  batchDimensions:(NSUInteger)b
                                                             name:nil];
    Set(L, op, Reshaped(L, gathered, OutShape(op)));
    return;
  }

  // General path (take_along_axis on a non-leading/interior axis): the squeezed indices already carry
  // the gather axis at position `axis` and align with the operand on every other axis — which is
  // exactly MPSGraph gatherAlongAxis (numpy take_along_axis semantics on any axis). No operand/index/
  // result transpose. Requires the take_along_axis shape: full-rank indices, every non-axis operand
  // dim batched (b == R-1); anything else (batched embedding with offset dims) falls through to fail.
  if (!ivdSqueezed || b != R - 1 || static_cast<int64_t>(ishape.size()) - 1 != R) {
    L.fail("jam: gather: only leading batch dims or batched take_along_axis supported");
    return;
  }
  MPSGraphTensor* gathered = [L.graph() gatherAlongAxis:(NSInteger)axis
                                       withUpdatesTensor:data
                                           indicesTensor:idx
                                                    name:nil];
  Set(L, op, Reshaped(L, gathered, OutShape(op)));
}

// Update-computation → MPSGraphScatterMode (bare return ⇒ Set; add/max/min/mul). ok=false on unknown.
static MPSGraphScatterMode ScatterModeOf(mlir::stablehlo::ScatterOp sc, bool& ok) {
  ok = true;
  std::vector<mlir::Operation*> bodyOps;
  for (mlir::Operation& inner : sc.getUpdateComputation().front())
    if (inner.getName().getStringRef() != "stablehlo.return") bodyOps.push_back(&inner);
  if (bodyOps.empty()) return MPSGraphScatterModeSet;
  llvm::StringRef f = bodyOps.front()->getName().getStringRef();
  if (f == "stablehlo.add") return MPSGraphScatterModeAdd;
  if (f == "stablehlo.maximum") return MPSGraphScatterModeMax;
  if (f == "stablehlo.minimum") return MPSGraphScatterModeMin;
  if (f == "stablehlo.multiply") return MPSGraphScatterModeMul;
  ok = false;
  return MPSGraphScatterModeSet;
}

// General multi-axis scatter (x.at[rows,cols].add/.set/.max) → MPSGraph scatterND. Point-scatter on
// ALL operand dims (inserted == scatter_dims_to_operand_dims, depth == rank, no window dims), no input
// batching. Transposes operand to scatter_dims_to_operand_dims order, scatterND, then back.
void ScatterND(Lowering& L, mlir::Operation* op) {
  auto sc = mlir::cast<mlir::stablehlo::ScatterOp>(op);
  auto dn = sc.getScatterDimensionNumbers();
  llvm::ArrayRef<int64_t> inserted = dn.getInsertedWindowDims();
  llvm::ArrayRef<int64_t> toOperand = dn.getScatterDimsToOperandDims();
  llvm::ArrayRef<int64_t> uwd = dn.getUpdateWindowDims();
  int64_t ivd = dn.getIndexVectorDim();
  if (!dn.getInputBatchingDims().empty()) { L.fail("jam: scatter: batched general scatter unsupported"); return; }
  int64_t rank = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank();
  int64_t depth = static_cast<int64_t>(toOperand.size());
  auto inToOp = [&](int64_t d) { for (int64_t k = 0; k < depth; ++k) if (toOperand[k] == d) return true; return false; };
  if (depth != rank || !uwd.empty() || static_cast<int64_t>(inserted.size()) != depth) {
    L.fail("jam: scatter: only single-axis or full point-scatter (advanced index assign) supported"); return;
  }
  for (int64_t d : inserted) if (!inToOp(d)) { L.fail("jam: scatter: general scatter needs inserted==scatter_dims_to_operand_dims"); return; }
  llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  int64_t idxRank = static_cast<int64_t>(ish.size());
  if (ivd < 0 || ivd >= idxRank || ish[ivd] != depth) { L.fail("jam: scatter: general scatter index vector unsupported"); return; }
  bool ok; MPSGraphScatterMode mode = ScatterModeOf(sc, ok);
  if (!ok) { L.fail("jam: scatter: unsupported update computation"); return; }

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* indices = L.value(op->getOperand(1));
  MPSGraphTensor* updates = L.value(op->getOperand(2));
  std::vector<int64_t> opPerm(toOperand.begin(), toOperand.end());  // depth==rank ⇒ full permutation
  data = Transposed(L, data, opPerm);
  std::vector<int64_t> idxPerm;
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);  // index vector last
  indices = Transposed(L, indices, idxPerm);
  // updates = scatter batch dims (indices sans ivd, ascending) — already that order, no permutation.
  MPSGraphTensor* scattered = [L.graph() scatterNDWithDataTensor:data updatesTensor:updates
                                                   indicesTensor:indices batchDimensions:0 mode:mode name:nil];
  std::vector<int64_t> inv(rank, 0);
  for (int64_t i = 0; i < rank; ++i) inv[opPerm[i]] = i;
  Set(L, op, Reshaped(L, Transposed(L, scattered, inv), OutShape(op)));
}

// scatter: single-axis set/add/min/max/mul; move axis to front for scatterND, then back.
void Scatter(Lowering& L, mlir::Operation* op) {
  auto sc = mlir::cast<mlir::stablehlo::ScatterOp>(op);
  auto dn = sc.getScatterDimensionNumbers();
  llvm::ArrayRef<int64_t> inserted = dn.getInsertedWindowDims();
  llvm::ArrayRef<int64_t> toOperand = dn.getScatterDimsToOperandDims();
  llvm::ArrayRef<int64_t> ibd = dn.getInputBatchingDims();
  llvm::ArrayRef<int64_t> sibd = dn.getScatterIndicesBatchingDims();
  if (toOperand.size() > 1) { ScatterND(L, op); return; }  // multi-axis advanced index assignment
  if (inserted.size() != 1 || toOperand.size() != 1 || inserted[0] != toOperand[0]) {
    L.fail("jam: scatter: only single-axis scatter supported");
    return;
  }
  int64_t axis = inserted[0];
  int64_t b = static_cast<int64_t>(ibd.size());
  if (static_cast<int64_t>(sibd.size()) != b) { L.fail("jam: scatter: mismatched batch dims"); return; }
  // Simple emission (below): batch dims are the leading dims [0..b) and the scatter axis follows them
  // (axis==b), so MPSGraph scatterND batchDimensions needs no operand permutation. Anything else (the
  // non-leading take_along_axis VJP) goes through the general permuting path further down.
  bool simple = true;
  for (int64_t i = 0; i < b; ++i)
    if (ibd[i] != i || sibd[i] != i) simple = false;
  if (b > 0 && axis != b) simple = false;

  bool ok; MPSGraphScatterMode mode = ScatterModeOf(sc, ok);
  if (!ok) { L.fail("jam: scatter: unsupported update computation"); return; }

  int64_t rank = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank();
  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* indices = L.value(op->getOperand(1));
  MPSGraphTensor* updates = L.value(op->getOperand(2));

  // General path (take_along_axis VJP on a non-leading/interior axis): mirror of the gather general
  // path. The squeezed indices/updates already align with the operand on every non-axis dim, so this
  // is exactly MPSGraph scatterAlongAxis (numpy put_along_axis with an accumulation mode) — no
  // operand/index/update transpose. Restricted to the no-window-dim, full-rank-index take_along_axis
  // VJP shape (every non-axis operand dim batched ⇒ b == rank-1).
  if (!simple) {
    int64_t ivd = dn.getIndexVectorDim();
    llvm::ArrayRef<int64_t> uwd = dn.getUpdateWindowDims();
    llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
    int64_t idxRank = static_cast<int64_t>(ish.size());
    if (b != rank - 1 || !uwd.empty() || ivd < 0 || ivd >= idxRank || ish[ivd] != 1 || idxRank - 1 != rank) {
      L.fail("jam: scatter: only leading batch dims or batched take_along_axis VJP supported");
      return;
    }
    // Drop the size-1 index-vector dim so indices align with data/updates (full operand rank).
    std::vector<int64_t> sq;
    for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) sq.push_back(ish[d]);
    indices = Reshaped(L, indices, sq);
    MPSGraphTensor* scattered = [L.graph() scatterAlongAxis:(NSInteger)axis
                                              withDataTensor:data
                                               updatesTensor:updates
                                               indicesTensor:indices
                                                        mode:mode
                                                        name:nil];
    Set(L, op, Reshaped(L, scattered, OutShape(op)));
    return;
  }

  // Non-batched scatter on a non-zero axis: move it to front for scatterND, then back. Batched
  // scatter keeps batch dims leading (axis already follows them), so no transpose.
  NSMutableArray<NSNumber*>* fwd = nil;
  if (b == 0 && axis != 0) {
    fwd = [NSMutableArray array];
    [fwd addObject:@(axis)];
    for (int64_t d = 0; d < rank; ++d) if (d != axis) [fwd addObject:@(d)];
    data = [L.graph() transposeTensor:data permutation:fwd name:nil];
    updates = [L.graph() transposeTensor:updates permutation:fwd name:nil];
  }
  MPSGraphTensor* scattered = [L.graph() scatterNDWithDataTensor:data
                                                   updatesTensor:updates
                                                   indicesTensor:indices
                                                 batchDimensions:(NSUInteger)b
                                                            mode:mode
                                                            name:nil];
  if (b == 0 && axis != 0) {
    NSMutableArray<NSNumber*>* inv = [NSMutableArray arrayWithCapacity:rank];
    for (int64_t i = 0; i < rank; ++i) [inv addObject:@0];
    for (int64_t i = 0; i < rank; ++i) inv[[fwd[i] integerValue]] = @(i);
    scattered = [L.graph() transposeTensor:scattered permutation:inv name:nil];
  }
  Set(L, op, scattered);
}

// Does `v` depend on block arg `want` and no other block arg of `block`?
bool TracesToOnlyArg(mlir::Value v, mlir::Block& block, unsigned want, int depth = 0) {
  if (depth > 24) return false;
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v))
    return ba.getOwner() == &block && ba.getArgNumber() == want;
  mlir::Operation* def = v.getDefiningOp();
  if (!def) return false;
  if (mlir::isa<mlir::stablehlo::ConstantOp>(def)) return true;  // constants don't bind an arg
  bool sawWant = false;
  for (mlir::Value o : def->getOperands()) {
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(o)) {
      if (ba.getOwner() != &block) continue;
      if (ba.getArgNumber() == want) { sawWant = true; continue; }
      return false;  // depends on a different block arg (e.g. the iota)
    }
    if (!TracesToOnlyArg(o, block, want, depth + 1)) return false;
    sawWant = true;  // a non-arg operand that itself traces to `want`
  }
  return sawWant;
}

// sort: single key, or the (key, iota) form JAX emits for top_k/argsort.
// Direction from the comparator's compare: LT/LE ⇒ asc, GT/GE ⇒ desc.
void Sort(Lowering& L, mlir::Operation* op) {
  auto s = mlir::cast<mlir::stablehlo::SortOp>(op);
  int64_t axis = s.getDimension();
  unsigned nOperands = op->getNumOperands();

  mlir::Operation* term = s.getComparator().front().getTerminator();
  if (!term || term->getNumOperands() != 1) { L.fail("jam: sort: unexpected comparator return"); return; }
  auto cmp = mlir::dyn_cast_or_null<mlir::stablehlo::CompareOp>(term->getOperand(0).getDefiningOp());
  if (!cmp) { L.fail("jam: sort: only single-key compare comparator supported"); return; }
  using Dir = mlir::stablehlo::ComparisonDirection;
  Dir d = cmp.getComparisonDirection();
  if (d != Dir::GT && d != Dir::GE && d != Dir::LT && d != Dir::LE) {
    L.fail("jam: sort: unsupported comparator direction");
    return;
  }
  bool descending = (d == Dir::GT || d == Dir::GE);

  // Unsigned keys: MPSGraph sort/argSort ASSERT on a uint source, so map unsigned order onto signed
  // order by flipping the MSB (xor) — signed sort of the flipped key = unsigned order — then flip the
  // sorted key back. The permutation is unaffected. (Same trick as the unsigned Compare/argmax.)
  MPSDataType uty;
  bool uns = UnsignedIntOperand(op, 0, uty);
  unsigned uw = uns ? ((uty == MPSDataTypeUInt8) ? 8 : (uty == MPSDataTypeUInt16) ? 16 : 32) : 0;
  auto flip = [&](MPSGraphTensor* t) {
    return [L.graph() bitwiseXORWithPrimaryTensor:t
                                   secondaryTensor:[L.graph() constantWithScalar:-(double)(1ULL << (uw - 1)) dataType:t.dataType]
                                              name:nil];
  };

  if (nOperands == 1) {
    MPSGraphTensor* k = L.value(op->getOperand(0));
    if (uns) k = flip(k);
    MPSGraphTensor* sorted = [L.graph() sortWithTensor:k axis:axis descending:descending name:nil];
    if (uns) sorted = flip(sorted);
    Set(L, op, sorted);
    return;
  }

  // (key, payload-iota) → (sorted key, sort permutation).
  if (nOperands != 2) { L.fail("jam: sort: only single-key or key+iota (top_k/argsort) sort supported"); return; }
  mlir::Block& body = s.getComparator().front();
  if (!TracesToOnlyArg(cmp.getLhs(), body, 0) || !TracesToOnlyArg(cmp.getRhs(), body, 1)) {
    L.fail("jam: sort: comparator must order by the leading key (key+iota form only)");
    return;
  }
  auto iota = mlir::dyn_cast_or_null<mlir::stablehlo::IotaOp>(op->getOperand(1).getDefiningOp());
  if (!iota || static_cast<int64_t>(iota.getIotaDimension()) != axis) {
    L.fail("jam: sort: key+payload sort only supported when the payload is an index iota along the sort axis");
    return;
  }
  MPSGraphTensor* key = L.value(op->getOperand(0));
  if (uns) key = flip(key);  // unsigned key ordering via MSB flip (argSort below sees the flipped key)
  MPSGraphTensor* sortedKey = [L.graph() sortWithTensor:key axis:axis descending:descending name:nil];
  if (uns) sortedKey = flip(sortedKey);
  Set(L, op, sortedKey);
  MPSGraphTensor* perm = [L.graph() argSortWithTensor:key axis:axis descending:descending name:nil];
  auto idxTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  L.bind(op->getResult(1), Casted(L, perm, Lowering::MpsDType(idxTy.getElementType())));
}

// chlo.top_k → (values, indices) along the last axis.
void TopK(Lowering& L, mlir::Operation* op) {
  auto tk = mlir::cast<mlir::chlo::TopKOp>(op);
  int64_t axis = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank() - 1;
  // (top_k is over float logits in practice; jax strips the unsigned type before chlo.top_k, so a
  // uint top_k can't be detected/corrected here — a non-issue since uint top_k is never used in ML.)
  NSArray<MPSGraphTensor*>* r = [L.graph() topKWithSourceTensor:L.value(op->getOperand(0))
                                                            axis:axis
                                                               k:(NSUInteger)tk.getK()
                                                            name:nil];
  Set(L, op, r[0]);  // values
  if (op->getNumResults() > 1) {
    auto idxTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
    L.bind(op->getResult(1), Casted(L, r[1], Lowering::MpsDType(idxTy.getElementType())));
  }
}

}  // namespace

void RegisterGatherScatter() {
  RegisterOp("stablehlo.gather", Gather);
  RegisterOp("stablehlo.scatter", Scatter);
  RegisterOp("stablehlo.sort", Sort);
  RegisterOp("chlo.top_k", TopK);
}

}  // namespace mccl_jax::jam
