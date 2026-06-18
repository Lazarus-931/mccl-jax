// ops/norm.mm — batch_norm_{training,inference,grad} and select_and_scatter (maxpool
// backward) via MPSGraph normalization + pooling-gradient nodes.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// All axes except `feature` — the batch-norm reduction axes (batch + spatial).
NSArray<NSNumber*>* AllAxesExcept(int64_t rank, int64_t feature) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d = 0; d < rank; ++d) if (d != feature) [a addObject:@(d)];
  return a;
}

// Reshape a per-channel 1-D param [C] to broadcast against the feature dim of a rank-N tensor.
MPSGraphTensor* ChannelBroadcast(Lowering& L, MPSGraphTensor* param, int64_t rank, int64_t feature) {
  std::vector<int64_t> sh(rank, 1);
  sh[feature] = -1;
  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  for (int64_t d = 0; d < rank; ++d) [shape addObject:@(d == feature ? -1 : 1)];
  return [L.graph() reshapeTensor:param withShape:shape name:nil];
}

// batch_norm_inference: out = scale*(x-mean)/sqrt(var+eps) + offset, per channel.
void BatchNormInference(Lowering& L, mlir::Operation* op) {
  auto bn = mlir::cast<mlir::stablehlo::BatchNormInferenceOp>(op);
  int64_t feature = bn.getFeatureIndex();
  float eps = bn.getEpsilon().convertToFloat();
  int64_t rank = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank();

  MPSGraphTensor* x = L.value(op->getOperand(0));
  MPSGraphTensor* scale = ChannelBroadcast(L, L.value(op->getOperand(1)), rank, feature);
  MPSGraphTensor* offset = ChannelBroadcast(L, L.value(op->getOperand(2)), rank, feature);
  MPSGraphTensor* mean = ChannelBroadcast(L, L.value(op->getOperand(3)), rank, feature);
  MPSGraphTensor* var = ChannelBroadcast(L, L.value(op->getOperand(4)), rank, feature);
  Set(L, op, [L.graph() normalizationWithTensor:x meanTensor:mean varianceTensor:var
                                    gammaTensor:scale betaTensor:offset epsilon:eps name:nil]);
}

// batch_norm_training -> (output, batch_mean, batch_var); per-channel stats over batch+spatial.
void BatchNormTraining(Lowering& L, mlir::Operation* op) {
  auto bn = mlir::cast<mlir::stablehlo::BatchNormTrainingOp>(op);
  int64_t feature = bn.getFeatureIndex();
  float eps = bn.getEpsilon().convertToFloat();
  int64_t rank = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank();
  NSArray<NSNumber*>* axes = AllAxesExcept(rank, feature);

  MPSGraphTensor* x = L.value(op->getOperand(0));
  MPSGraphTensor* mean = [L.graph() meanOfTensor:x axes:axes name:nil];      // shape [1,..,C,..,1]
  MPSGraphTensor* var = [L.graph() varianceOfTensor:x meanTensor:mean axes:axes name:nil];
  MPSGraphTensor* scale = ChannelBroadcast(L, L.value(op->getOperand(1)), rank, feature);
  MPSGraphTensor* offset = ChannelBroadcast(L, L.value(op->getOperand(2)), rank, feature);
  Set(L, op, [L.graph() normalizationWithTensor:x meanTensor:mean varianceTensor:var
                                    gammaTensor:scale betaTensor:offset epsilon:eps name:nil]);

  // results 1,2 = batch mean and variance, shape [C] (squeeze the reduced dims).
  auto meanTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  L.bind(op->getResult(1), Reshaped(L, mean, meanTy.getShape()));
  auto varTy = mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());
  L.bind(op->getResult(2), Reshaped(L, var, varTy.getShape()));
}

// batch_norm_grad -> (grad_operand, grad_scale, grad_offset) via MPSGraph normalization gradients.
void BatchNormGrad(Lowering& L, mlir::Operation* op) {
  auto bn = mlir::cast<mlir::stablehlo::BatchNormGradOp>(op);
  int64_t feature = bn.getFeatureIndex();
  float eps = bn.getEpsilon().convertToFloat();
  int64_t rank = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getRank();
  NSArray<NSNumber*>* axes = AllAxesExcept(rank, feature);

  MPSGraphTensor* x = L.value(op->getOperand(0));
  MPSGraphTensor* scale = ChannelBroadcast(L, L.value(op->getOperand(1)), rank, feature);
  MPSGraphTensor* mean = ChannelBroadcast(L, L.value(op->getOperand(2)), rank, feature);
  MPSGraphTensor* var = ChannelBroadcast(L, L.value(op->getOperand(3)), rank, feature);
  MPSGraphTensor* dout = L.value(op->getOperand(4));

  MPSGraphTensor* dGamma = [L.graph() normalizationGammaGradientWithIncomingGradientTensor:dout
      sourceTensor:x meanTensor:mean varianceTensor:var reductionAxes:axes epsilon:eps name:nil];
  MPSGraphTensor* dBeta = [L.graph() normalizationBetaGradientWithIncomingGradientTensor:dout
      sourceTensor:x reductionAxes:axes name:nil];
  MPSGraphTensor* dX = [L.graph() normalizationGradientWithIncomingGradientTensor:dout
      sourceTensor:x meanTensor:mean varianceTensor:var gammaTensor:scale
      gammaGradientTensor:dGamma betaGradientTensor:dBeta reductionAxes:axes epsilon:eps name:nil];

  Set(L, op, dX);
  auto gsTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  L.bind(op->getResult(1), Reshaped(L, dGamma, gsTy.getShape()));
  auto goTy = mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());
  L.bind(op->getResult(2), Reshaped(L, dBeta, goTy.getShape()));
}

static bool AllOnes4(std::optional<llvm::ArrayRef<int64_t>> a) {
  if (!a.has_value()) return true;
  for (int64_t v : *a) if (v != 1) return false;
  return true;
}

// select_and_scatter: NHWC max-pool backward (GE select + add scatter) → maxPooling2DGradient.
void SelectAndScatter(Lowering& L, mlir::Operation* op) {
  auto ss = mlir::cast<mlir::stablehlo::SelectAndScatterOp>(op);
  llvm::ArrayRef<int64_t> wd = ss.getWindowDimensions().value_or(llvm::ArrayRef<int64_t>{});
  if (wd.size() != 4 || wd[0] != 1 || wd[3] != 1) {
    L.fail("jam: select_and_scatter: only NHWC 2D max-pool backward supported");
    return;
  }
  // confirm the select region is a GE/GT compare (max selection).
  bool isMaxSelect = false;
  for (mlir::Operation& inner : ss.getSelect().front()) {
    if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&inner)) {
      using Dir = mlir::stablehlo::ComparisonDirection;
      Dir d = cmp.getComparisonDirection();
      isMaxSelect = (d == Dir::GE || d == Dir::GT);
      break;
    }
  }
  if (!isMaxSelect) { L.fail("jam: select_and_scatter: only max (>=) selection supported"); return; }

  int64_t kh = wd[1], kw = wd[2], sh = kh, sw = kw;
  if (auto s = ss.getWindowStrides()) { auto sr = *s; if (sr.size() == 4) { sh = sr[1]; sw = sr[2]; } }
  int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
  if (auto padAttr = ss.getPadding()) {
    std::vector<int64_t> pv;
    for (const llvm::APInt& v : padAttr->getValues<llvm::APInt>()) pv.push_back(v.getSExtValue());
    if (pv.size() == 8) { padTop = pv[2]; padBottom = pv[3]; padLeft = pv[4]; padRight = pv[5]; }
  }

  MPSGraphPooling2DOpDescriptor* desc =
      [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kw kernelHeight:(NSUInteger)kh
          strideInX:(NSUInteger)sw strideInY:(NSUInteger)sh dilationRateInX:1 dilationRateInY:1
          paddingLeft:(NSUInteger)padLeft paddingRight:(NSUInteger)padRight
          paddingTop:(NSUInteger)padTop paddingBottom:(NSUInteger)padBottom
          paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

  MPSGraphTensor* source = L.value(op->getOperand(0));  // original input (forward source)
  MPSGraphTensor* grad = L.value(op->getOperand(1));    // cotangent (per-output gradient)
  Set(L, op, [L.graph() maxPooling2DGradientWithGradientTensor:grad sourceTensor:source
                                                    descriptor:desc name:nil]);
}

}  // namespace

void RegisterNorm() {
  RegisterOp("stablehlo.batch_norm_inference", BatchNormInference);
  RegisterOp("stablehlo.batch_norm_training", BatchNormTraining);
  RegisterOp("stablehlo.batch_norm_grad", BatchNormGrad);
  RegisterOp("stablehlo.select_and_scatter", SelectAndScatter);
}

}  // namespace mccl_jax::jam
