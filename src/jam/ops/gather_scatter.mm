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
  std::vector<int64_t> idxPerm;
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);
  idx = Casted(L, Transposed(L, idx, idxPerm), MPSDataTypeInt32);

  auto isOffset = [&](int64_t ax) { for (int64_t o : offsetDims) if (o == ax) return true; return false; };
  std::vector<MPSGraphTensor*> coord(R, nullptr);
  for (int64_t k = 0; k < R; ++k) {
    int64_t d = startMap[k];
    int64_t winAxis = offsetDims[d];

    MPSGraphTensor* sc = [L.graph() sliceTensor:idx dimension:(NSUInteger)numBatch start:(NSInteger)k length:1 name:nil];
    std::vector<int64_t> rO; int64_t bi = 0;
    for (int64_t ax = 0; ax < outRank; ++ax) rO.push_back(isOffset(ax) ? 1 : ish[idxPerm[bi++]]);
    sc = [L.graph() reshapeTensor:sc withShape:IntArray(rO) name:nil];

    MPSGraphTensor* hi = [L.graph() constantWithScalar:(double)(opShape[d] - sliceSizes[d]) dataType:MPSDataTypeInt32];
    MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeInt32];
    sc = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:sc secondaryTensor:hi name:nil]
                               secondaryTensor:lo name:nil];

    MPSGraphTensor* it = Casted(L, [L.graph() coordinateAlongAxis:(NSInteger)winAxis withShape:ShapeArray(outDims) name:nil], MPSDataTypeInt32);
    MPSGraphTensor* c = [L.graph() additionWithPrimaryTensor:sc secondaryTensor:it name:nil];
    std::vector<int64_t> oc(outDims.begin(), outDims.end()); oc.push_back(1);
    coord[d] = [L.graph() reshapeTensor:c withShape:IntArray(oc) name:nil];
  }
  NSMutableArray<MPSGraphTensor*>* arr = [NSMutableArray array];
  for (int64_t d = 0; d < R; ++d) [arr addObject:coord[d]];
  MPSGraphTensor* coords = [L.graph() concatTensors:arr dimension:(NSInteger)outRank name:nil];
  MPSGraphTensor* gathered = [L.graph() gatherNDWithUpdatesTensor:data indicesTensor:coords batchDimensions:0 name:nil];
  Set(L, op, Reshaped(L, gathered, outDims));
}

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

  bool windowed = false;
  for (int64_t k = 0; k < depth; ++k) if (sliceSizes[startMap[k]] > 1) windowed = true;
  if (windowed && collapsed.empty() && depth == R && static_cast<int64_t>(offsetDims.size()) == R) {
    WindowedGather(L, op);
    return;
  }

  for (unsigned i = 0; i < sliceSizes.size(); ++i)
    if (sliceSizes[i] != (inStartMap(i) ? 1 : opShape[i])) { L.fail("jam: gather: windowed (slice>1) gather unsupported"); return; }

  llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  int64_t idxRank = static_cast<int64_t>(ish.size());
  if (ivd < 0 || ivd >= idxRank || ish[ivd] != depth) { L.fail("jam: gather: general gather index vector unsupported"); return; }

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* idx = L.value(op->getOperand(1));

  std::vector<int64_t> opPerm;
  for (int64_t k = 0; k < depth; ++k) opPerm.push_back(startMap[k]);
  for (int64_t d = 0; d < R; ++d) if (!inStartMap(d)) opPerm.push_back(d);
  data = Transposed(L, data, opPerm);

  std::vector<int64_t> rest;
  for (int64_t d = 0; d < R; ++d) if (!inCollapsed(d)) rest.push_back(d);

  std::vector<int64_t> idxPerm;
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);
  idx = Casted(L, Transposed(L, idx, idxPerm), MPSDataTypeInt32);

  std::vector<int32_t> hi32(depth);
  for (int64_t k = 0; k < depth; ++k) hi32[k] = (int32_t)(opShape[startMap[k]] - 1);
  NSData* hd = [NSData dataWithBytes:hi32.data() length:depth * sizeof(int32_t)];
  MPSGraphTensor* hi = [L.graph() constantWithData:hd shape:@[@(depth)] dataType:MPSDataTypeInt32];
  MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeInt32];
  idx = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:idx secondaryTensor:hi name:nil]
                             secondaryTensor:lo name:nil];
  MPSGraphTensor* gathered = [L.graph() gatherNDWithUpdatesTensor:data indicesTensor:idx batchDimensions:0 name:nil];

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

void Gather(Lowering& L, mlir::Operation* op) {
  auto g = mlir::cast<mlir::stablehlo::GatherOp>(op);
  auto dn = g.getDimensionNumbers();
  llvm::ArrayRef<int64_t> collapsed = dn.getCollapsedSliceDims();
  llvm::ArrayRef<int64_t> startMap = dn.getStartIndexMap();
  llvm::ArrayRef<int64_t> obd = dn.getOperandBatchingDims();
  llvm::ArrayRef<int64_t> sibd = dn.getStartIndicesBatchingDims();
  int64_t ivd = dn.getIndexVectorDim();
  llvm::ArrayRef<int64_t> sliceSizes = g.getSliceSizes();

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

  for (unsigned i = 0; i < sliceSizes.size(); ++i) {
    bool one = (static_cast<int64_t>(i) == axis) || isBatchOperand(i);
    int64_t want = one ? 1 : operandShape[i];
    if (sliceSizes[i] != want) { L.fail("jam: gather: non-full slice (general gather) unsupported"); return; }
  }

  MPSGraphTensor* data = L.value(op->getOperand(0));
  MPSGraphTensor* idx = L.value(op->getOperand(1));

  llvm::ArrayRef<int64_t> ishape = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  bool ivdSqueezed = (ivd >= 0 && ivd < static_cast<int64_t>(ishape.size()) && ishape[ivd] == 1);
  if (ivdSqueezed) {
    std::vector<int64_t> squeezed;
    for (unsigned i = 0; i < ishape.size(); ++i)
      if (static_cast<int64_t>(i) != ivd) squeezed.push_back(ishape[i]);
    if (squeezed.empty()) squeezed.push_back(1);
    idx = Reshaped(L, idx, squeezed);
  }

  MPSGraphTensor* hi = [L.graph() constantWithScalar:(double)(operandShape[axis] - 1) dataType:idx.dataType];
  MPSGraphTensor* lo = [L.graph() constantWithScalar:0.0 dataType:idx.dataType];
  idx = [L.graph() maximumWithPrimaryTensor:[L.graph() minimumWithPrimaryTensor:idx secondaryTensor:hi name:nil]
                             secondaryTensor:lo name:nil];

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
  std::vector<int64_t> opPerm(toOperand.begin(), toOperand.end());
  data = Transposed(L, data, opPerm);
  std::vector<int64_t> idxPerm;
  for (int64_t d = 0; d < idxRank; ++d) if (d != ivd) idxPerm.push_back(d);
  idxPerm.push_back(ivd);
  indices = Transposed(L, indices, idxPerm);

  MPSGraphTensor* scattered = [L.graph() scatterNDWithDataTensor:data updatesTensor:updates
                                                   indicesTensor:indices batchDimensions:0 mode:mode name:nil];
  std::vector<int64_t> inv(rank, 0);
  for (int64_t i = 0; i < rank; ++i) inv[opPerm[i]] = i;
  Set(L, op, Reshaped(L, Transposed(L, scattered, inv), OutShape(op)));
}

void Scatter(Lowering& L, mlir::Operation* op) {
  auto sc = mlir::cast<mlir::stablehlo::ScatterOp>(op);
  auto dn = sc.getScatterDimensionNumbers();
  llvm::ArrayRef<int64_t> inserted = dn.getInsertedWindowDims();
  llvm::ArrayRef<int64_t> toOperand = dn.getScatterDimsToOperandDims();
  llvm::ArrayRef<int64_t> ibd = dn.getInputBatchingDims();
  llvm::ArrayRef<int64_t> sibd = dn.getScatterIndicesBatchingDims();
  if (toOperand.size() > 1) { ScatterND(L, op); return; }
  if (inserted.size() != 1 || toOperand.size() != 1 || inserted[0] != toOperand[0]) {
    L.fail("jam: scatter: only single-axis scatter supported");
    return;
  }
  int64_t axis = inserted[0];
  int64_t b = static_cast<int64_t>(ibd.size());
  if (static_cast<int64_t>(sibd.size()) != b) { L.fail("jam: scatter: mismatched batch dims"); return; }

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

  if (!simple) {
    int64_t ivd = dn.getIndexVectorDim();
    llvm::ArrayRef<int64_t> uwd = dn.getUpdateWindowDims();
    llvm::ArrayRef<int64_t> ish = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
    int64_t idxRank = static_cast<int64_t>(ish.size());
    if (b != rank - 1 || !uwd.empty() || ivd < 0 || ivd >= idxRank || ish[ivd] != 1 || idxRank - 1 != rank) {
      L.fail("jam: scatter: only leading batch dims or batched take_along_axis VJP supported");
      return;
    }

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

bool TracesToOnlyArg(mlir::Value v, mlir::Block& block, unsigned want, int depth = 0) {
  if (depth > 24) return false;
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v))
    return ba.getOwner() == &block && ba.getArgNumber() == want;
  mlir::Operation* def = v.getDefiningOp();
  if (!def) return false;
  if (mlir::isa<mlir::stablehlo::ConstantOp>(def)) return true;
  bool sawWant = false;
  for (mlir::Value o : def->getOperands()) {
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(o)) {
      if (ba.getOwner() != &block) continue;
      if (ba.getArgNumber() == want) { sawWant = true; continue; }
      return false;
    }
    if (!TracesToOnlyArg(o, block, want, depth + 1)) return false;
    sawWant = true;
  }
  return sawWant;
}

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
  if (uns) key = flip(key);
  MPSGraphTensor* sortedKey = [L.graph() sortWithTensor:key axis:axis descending:descending name:nil];
  if (uns) sortedKey = flip(sortedKey);
  Set(L, op, sortedKey);
  MPSGraphTensor* perm = [L.graph() argSortWithTensor:key axis:axis descending:descending name:nil];
  auto idxTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  L.bind(op->getResult(1), Casted(L, perm, Lowering::MpsDType(idxTy.getElementType())));
}

void TopK(Lowering& L, mlir::Operation* op) {
  auto tk = mlir::cast<mlir::chlo::TopKOp>(op);
  int64_t axis = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank() - 1;

  NSArray<MPSGraphTensor*>* r = [L.graph() topKWithSourceTensor:L.value(op->getOperand(0))
                                                            axis:axis
                                                               k:(NSUInteger)tk.getK()
                                                            name:nil];
  Set(L, op, r[0]);
  if (op->getNumResults() > 1) {
    auto idxTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
    L.bind(op->getResult(1), Casted(L, r[1], Lowering::MpsDType(idxTy.getElementType())));
  }
}

}

void RegisterGatherScatter() {
  RegisterOp("stablehlo.gather", Gather);
  RegisterOp("stablehlo.scatter", Scatter);
  RegisterOp("stablehlo.sort", Sort);
  RegisterOp("chlo.top_k", TopK);
}

}
