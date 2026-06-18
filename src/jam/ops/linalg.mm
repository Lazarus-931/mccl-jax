// ops/linalg.mm — dot_general (batched, single contracting dim) and convolution (2D).

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

bool Contains(llvm::ArrayRef<int64_t> v, int64_t x) {
  for (int64_t e : v) if (e == x) return true;
  return false;
}

NSArray<NSNumber*>* PermFrom(const std::vector<int64_t>& target) {
  NSMutableArray<NSNumber*>* p = [NSMutableArray array];
  for (int64_t d : target) [p addObject:@(d)];
  return p;
}

// dot_general: transpose each operand to [batch..., free..., contract...] (lhs) / [batch...,
// contract..., free...] (rhs), flatten the free and contract groups so it's a batched matmul over
// the last two dims, multiply, then reshape the [batch..., free_l, free_r] product to the result
// shape. Handles any number of batch/contract/free dims (the i-th lhs contract pairs with i-th rhs).
static bool IsIdentityPerm(const std::vector<int64_t>& perm) {
  for (std::size_t i = 0; i < perm.size(); ++i)
    if (perm[i] != (int64_t)i) return false;
  return true;
}

// Transpose only if `perm` reorders, reshape only if the shape actually changes. For the common 2D
// [M,K]@[K,N] matmul both are no-ops; emitting them anyway would add dead transpose/reshape kernels.
static MPSGraphTensor* AlignOperand(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> shape,
                                    const std::vector<int64_t>& perm, const std::vector<int64_t>& reshape) {
  std::vector<int64_t> cur(shape.begin(), shape.end());
  if (!IsIdentityPerm(perm)) {
    t = [L.graph() transposeTensor:t permutation:PermFrom(perm) name:nil];
    std::vector<int64_t> permuted;
    for (int64_t d : perm) permuted.push_back(shape[d]);
    cur.swap(permuted);
  }
  if (cur != reshape) t = [L.graph() reshapeTensor:t withShape:PermFrom(reshape) name:nil];
  return t;
}

void DotGeneral(Lowering& L, mlir::Operation* op) {
  auto dg = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto dn = dg.getDotDimensionNumbers();
  llvm::ArrayRef<int64_t> lc = dn.getLhsContractingDimensions();
  llvm::ArrayRef<int64_t> rc = dn.getRhsContractingDimensions();
  llvm::ArrayRef<int64_t> lb = dn.getLhsBatchingDimensions();
  llvm::ArrayRef<int64_t> rb = dn.getRhsBatchingDimensions();
  if (lc.size() != rc.size() || lb.size() != rb.size()) {
    L.fail("jam: dot_general contracting/batching dims must pair up");
    return;
  }
  // MPSGraph matrixMultiplication is float-only; reject integer matmul cleanly (else MPSGraph aborts).
  if (!mlir::isa<mlir::FloatType>(
          mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getElementType())) {
    L.fail("jam: dot_general requires floating-point operands (MPSGraph has no integer matmul)");
    return;
  }
  auto lshape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  auto rshape = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
  int64_t lr = lshape.size(), rr = rshape.size();

  // Build permutations + the flattened [batch..., group_a, group_b] reshape target for each operand.
  auto plan = [](llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> batch,
                 llvm::ArrayRef<int64_t> contract, bool contract_first,
                 std::vector<int64_t>& perm, std::vector<int64_t>& reshape) {
    int64_t rank = shape.size();
    std::vector<int64_t> freeDims;
    for (int64_t d = 0; d < rank; ++d)
      if (!Contains(batch, d) && !Contains(contract, d)) freeDims.push_back(d);
    int64_t freeProd = 1, contractProd = 1;
    for (int64_t d : freeDims) freeProd *= shape[d];
    for (int64_t d : contract) contractProd *= shape[d];
    for (int64_t d : batch) { perm.push_back(d); reshape.push_back(shape[d]); }
    if (contract_first) {  // rhs: [batch, contract, free]
      for (int64_t d : contract) perm.push_back(d);
      for (int64_t d : freeDims) perm.push_back(d);
      reshape.push_back(contractProd);
      reshape.push_back(freeProd);
    } else {               // lhs: [batch, free, contract]
      for (int64_t d : freeDims) perm.push_back(d);
      for (int64_t d : contract) perm.push_back(d);
      reshape.push_back(freeProd);
      reshape.push_back(contractProd);
    }
  };
  std::vector<int64_t> lp, lrs, rp, rrs;
  plan(lshape, lb, lc, /*contract_first=*/false, lp, lrs);
  plan(rshape, rb, rc, /*contract_first=*/true, rp, rrs);
  (void)lr; (void)rr;

  MPSGraphTensor* la = AlignOperand(L, L.value(op->getOperand(0)), lshape, lp, lrs);
  MPSGraphTensor* ra = AlignOperand(L, L.value(op->getOperand(1)), rshape, rp, rrs);
  MPSGraphTensor* mm = [L.graph() matrixMultiplicationWithPrimaryTensor:la secondaryTensor:ra name:nil];

  // matmul output is [batch..., free_l, free_r] (= lrs with its contracting dim replaced by free_r);
  // skip the final reshape when that already equals the result shape (the 2D case).
  std::vector<int64_t> mmShape = lrs;
  if (!mmShape.empty() && !rrs.empty()) mmShape.back() = rrs.back();
  std::vector<int64_t> outDims(
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape());
  Set(L, op, mmShape == outDims ? mm : [L.graph() reshapeTensor:mm withShape:PermFrom(outDims) name:nil]);
}

// convolution (1D and 2D): permute to NHWC data / HWIO weights, conv, permute result back.
// 1D is promoted to 2D with a singleton height dimension.
void Convolution(Lowering& L, mlir::Operation* op) {
  auto conv = mlir::cast<mlir::stablehlo::ConvolutionOp>(op);
  auto dn = conv.getDimensionNumbers();
  llvm::ArrayRef<int64_t> inSpatial = dn.getInputSpatialDimensions();
  llvm::ArrayRef<int64_t> kSpatial = dn.getKernelSpatialDimensions();
  llvm::ArrayRef<int64_t> outSpatial = dn.getOutputSpatialDimensions();
  if (inSpatial.size() != 1 && inSpatial.size() != 2) {
    L.fail("jam: convolution: only 1D and 2D conv supported");
    return;
  }
  if (conv.getBatchGroupCount() != 1) { L.fail("jam: convolution: batch_group_count != 1 unsupported"); return; }
  // lhs_dilation > 1 = transposed/fractionally-strided conv: interior-dilate the input spatial dims,
  // then a regular (stride-1) conv. Collected here, applied per-path below.
  std::vector<int64_t> lhsD;
  if (auto ld = conv.getLhsDilation()) for (int64_t v : *ld) lhsD.push_back(v);
  bool transposed = false;
  for (int64_t v : lhsD) if (v > 1) transposed = true;

  // ---- 1D: promote to 2D conv with H=1 (a [1,KW] kernel over [N,1,W,C] data) ----
  if (inSpatial.size() == 1) {
    auto dS = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
    auto wS = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType()).getShape();
    auto oS = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
    int64_t N = dS[dn.getInputBatchDimension()], W = dS[inSpatial[0]], C = dS[dn.getInputFeatureDimension()];
    std::vector<int64_t> dPerm = {dn.getInputBatchDimension(), inSpatial[0], dn.getInputFeatureDimension()};
    MPSGraphTensor* s = Transposed(L, L.value(op->getOperand(0)), dPerm);
    s = [L.graph() reshapeTensor:s withShape:@[ @(N), @1, @(W), @(C) ] name:nil];  // NWC -> N,1,W,C
    if (transposed) {  // dilate W (dim 2) by lhs_dilation-1
      std::vector<int64_t> sh = {N, 1, W, C}, inter = {0, 0, (lhsD.size() > 0 ? lhsD[0] - 1 : 0), 0};
      s = InteriorDilate(L, s, sh, inter, 0.0);
    }
    int64_t KW = wS[kSpatial[0]], KI = wS[dn.getKernelInputFeatureDimension()], KO = wS[dn.getKernelOutputFeatureDimension()];
    std::vector<int64_t> wPerm = {kSpatial[0], dn.getKernelInputFeatureDimension(), dn.getKernelOutputFeatureDimension()};
    MPSGraphTensor* w = Transposed(L, L.value(op->getOperand(1)), wPerm);
    w = [L.graph() reshapeTensor:w withShape:@[ @1, @(KW), @(KI), @(KO) ] name:nil];  // WIO -> 1,W,I,O
    int64_t sX = 1, dX = 1, pL = 0, pR = 0;
    if (auto st = conv.getWindowStrides()) { auto a = *st; if (a.size() == 1) sX = a[0]; }
    if (auto rd = conv.getRhsDilation()) { auto a = *rd; if (a.size() == 1) dX = a[0]; }
    if (auto pad = conv.getPadding()) {
      std::vector<int64_t> pv;
      for (const llvm::APInt& v : pad->getValues<llvm::APInt>()) pv.push_back(v.getSExtValue());
      if (pv.size() == 2) { pL = pv[0]; pR = pv[1]; }
    }
    MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:(NSUInteger)sX strideInY:1 dilationRateInX:(NSUInteger)dX dilationRateInY:1
                         groups:(NSUInteger)conv.getFeatureGroupCount()
                    paddingLeft:(NSUInteger)pL paddingRight:(NSUInteger)pR paddingTop:0 paddingBottom:0
                   paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                  weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    MPSGraphTensor* o = [L.graph() convolution2DWithSourceTensor:s weightsTensor:w descriptor:desc name:nil];
    int64_t oN = oS[dn.getOutputBatchDimension()], oW = oS[outSpatial[0]], oC = oS[dn.getOutputFeatureDimension()];
    o = [L.graph() reshapeTensor:o withShape:@[ @(oN), @(oW), @(oC) ] name:nil];  // N,1,W',C -> NWC
    std::vector<int64_t> nwcSlotOf(3, 0);  // NWC -> StableHLO output layout
    nwcSlotOf[dn.getOutputBatchDimension()] = 0;
    nwcSlotOf[outSpatial[0]] = 1;
    nwcSlotOf[dn.getOutputFeatureDimension()] = 2;
    Set(L, op, Transposed(L, o, nwcSlotOf));
    return;
  }

  // ---- 2D ----
  std::vector<int64_t> dataPerm = {dn.getInputBatchDimension(), inSpatial[0], inSpatial[1],
                                   dn.getInputFeatureDimension()};  // → NHWC
  MPSGraphTensor* src = Transposed(L, L.value(op->getOperand(0)), dataPerm);
  if (transposed) {  // interior-dilate H,W by lhs_dilation-1 (then conv runs at stride 1)
    auto dS = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
    std::vector<int64_t> sh = {dS[dn.getInputBatchDimension()], dS[inSpatial[0]], dS[inSpatial[1]],
                               dS[dn.getInputFeatureDimension()]};
    std::vector<int64_t> inter = {0, (lhsD.size() > 0 ? lhsD[0] - 1 : 0),
                                  (lhsD.size() > 1 ? lhsD[1] - 1 : 0), 0};
    src = InteriorDilate(L, src, sh, inter, 0.0);
  }
  std::vector<int64_t> wPerm = {kSpatial[0], kSpatial[1], dn.getKernelInputFeatureDimension(),
                                dn.getKernelOutputFeatureDimension()};  // → HWIO
  MPSGraphTensor* w = Transposed(L, L.value(op->getOperand(1)), wPerm);

  int64_t strideY = 1, strideX = 1, dilY = 1, dilX = 1;
  if (auto s = conv.getWindowStrides()) { auto sr = *s; if (sr.size() == 2) { strideY = sr[0]; strideX = sr[1]; } }
  if (auto d = conv.getRhsDilation()) { auto dr = *d; if (dr.size() == 2) { dilY = dr[0]; dilX = dr[1]; } }
  int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
  if (auto pad = conv.getPadding()) {
    std::vector<int64_t> pv;
    for (const llvm::APInt& v : pad->getValues<llvm::APInt>()) pv.push_back(v.getSExtValue());
    if (pv.size() == 4) { padTop = pv[0]; padBottom = pv[1]; padLeft = pv[2]; padRight = pv[3]; }
  }
  int64_t groups = static_cast<int64_t>(conv.getFeatureGroupCount());

  MPSGraphConvolution2DOpDescriptor* desc =
      [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)strideX
                                                       strideInY:(NSUInteger)strideY
                                                 dilationRateInX:(NSUInteger)dilX
                                                 dilationRateInY:(NSUInteger)dilY
                                                          groups:(NSUInteger)groups
                                                     paddingLeft:(NSUInteger)padLeft
                                                    paddingRight:(NSUInteger)padRight
                                                      paddingTop:(NSUInteger)padTop
                                                   paddingBottom:(NSUInteger)padBottom
                                                    paddingStyle:MPSGraphPaddingStyleExplicit
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                   weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
  MPSGraphTensor* out = [L.graph() convolution2DWithSourceTensor:src weightsTensor:w descriptor:desc name:nil];

  // out is NHWC; permute back to the StableHLO output layout.
  std::vector<int64_t> nhwcSlotOf(4, 0);
  nhwcSlotOf[dn.getOutputBatchDimension()] = 0;
  nhwcSlotOf[outSpatial[0]] = 1;
  nhwcSlotOf[outSpatial[1]] = 2;
  nhwcSlotOf[dn.getOutputFeatureDimension()] = 3;
  Set(L, op, Transposed(L, out, nhwcSlotOf));
}

}  // namespace

void RegisterLinalg() {
  RegisterOp("stablehlo.dot_general", DotGeneral);
  RegisterOp("stablehlo.convolution", Convolution);
}

}  // namespace mccl_jax::jam
