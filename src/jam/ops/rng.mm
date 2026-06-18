#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

NSUInteger SeedFor(mlir::Operation* op) {
  uint64_t h = 0x9E3779B97F4A7C15ull;
  for (int64_t d : OutShape(op, op->getNumResults() - 1)) h = h * 1099511628211ull + (uint64_t)d;
  return (NSUInteger)h;
}

void RngBitGenerator(Lowering& L, mlir::Operation* op) {
  auto outBitsTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  MPSDataType mps = Lowering::MpsDType(outBitsTy.getElementType());
  NSArray<NSNumber*>* shape = ShapeArray(outBitsTy.getShape());

  MPSGraphRandomOpDescriptor* desc =
      [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                    dataType:MPSDataTypeFloat32];
  desc.min = 0.0f;
  desc.max = 1.0f;
  MPSGraphTensor* u = [L.graph() randomTensorWithShape:shape descriptor:desc seed:SeedFor(op) name:nil];

  MPSGraphTensor* scaled = [L.graph() multiplicationWithPrimaryTensor:u
      secondaryTensor:[L.graph() constantWithScalar:4294967296.0 dataType:MPSDataTypeFloat32]
                 name:nil];
  MPSGraphTensor* bits = [L.graph() castTensor:scaled toType:mps name:nil];

  L.bind(op->getResult(0), L.value(op->getOperand(0)));
  L.bind(op->getResult(1), bits);
}

void Rng(Lowering& L, mlir::Operation* op) {
  auto rng = mlir::cast<mlir::stablehlo::RngOp>(op);
  auto outTy = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  MPSDataType mps = Lowering::MpsDType(outTy.getElementType());
  NSArray<NSNumber*>* shape = ShapeArray(outTy.getShape());

  bool normal = (rng.getRngDistribution() == mlir::stablehlo::RngDistribution::NORMAL);
  MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor
      descriptorWithDistribution:(normal ? MPSGraphRandomDistributionNormal
                                         : MPSGraphRandomDistributionUniform)
                        dataType:(mps == MPSDataTypeFloat16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32)];

  MPSGraphTensor* a = L.value(op->getOperand(0));
  MPSGraphTensor* b = L.value(op->getOperand(1));
  MPSGraphTensor* r = [L.graph() randomTensorWithShape:shape descriptor:desc seed:SeedFor(op) name:nil];
  if (normal) {

    r = [L.graph() multiplicationWithPrimaryTensor:r secondaryTensor:b name:nil];
    r = [L.graph() additionWithPrimaryTensor:r secondaryTensor:a name:nil];
  } else {

    MPSGraphTensor* span = [L.graph() subtractionWithPrimaryTensor:b secondaryTensor:a name:nil];
    r = [L.graph() multiplicationWithPrimaryTensor:r secondaryTensor:span name:nil];
    r = [L.graph() additionWithPrimaryTensor:r secondaryTensor:a name:nil];
  }
  Set(L, op, [L.graph() castTensor:r toType:mps name:nil]);
}

}

void RegisterRng() {
  RegisterOp("stablehlo.rng_bit_generator", RngBitGenerator);
  RegisterOp("stablehlo.rng", Rng);
}

}
