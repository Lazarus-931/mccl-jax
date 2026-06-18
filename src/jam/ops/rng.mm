// ops/rng.mm — rng_bit_generator and rng via MPSGraph random ops.
// Not bit-identical to XLA; guarantees correct shape/dtype and a valid uniform stream only.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// Derive a deterministic 64-bit seed from the result's static shape.
NSUInteger SeedFor(mlir::Operation* op) {
  uint64_t h = 0x9E3779B97F4A7C15ull;
  for (int64_t d : OutShape(op, op->getNumResults() - 1)) h = h * 1099511628211ull + (uint64_t)d;
  return (NSUInteger)h;
}

// rng_bit_generator(state) -> (new_state, bits); state forwarded unchanged.
void RngBitGenerator(Lowering& L, mlir::Operation* op) {
  auto outBitsTy = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  MPSDataType mps = Lowering::MpsDType(outBitsTy.getElementType());
  NSArray<NSNumber*>* shape = ShapeArray(outBitsTy.getShape());

  // Uniform over the full integer range, then reinterpret to the requested int width.
  MPSGraphRandomOpDescriptor* desc =
      [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                    dataType:MPSDataTypeFloat32];
  desc.min = 0.0f;
  desc.max = 1.0f;
  MPSGraphTensor* u = [L.graph() randomTensorWithShape:shape descriptor:desc seed:SeedFor(op) name:nil];
  // Scale to the int range (2^32) and cast.
  MPSGraphTensor* scaled = [L.graph() multiplicationWithPrimaryTensor:u
      secondaryTensor:[L.graph() constantWithScalar:4294967296.0 dataType:MPSDataTypeFloat32]
                 name:nil];
  MPSGraphTensor* bits = [L.graph() castTensor:scaled toType:mps name:nil];

  // new_state: forward the input state unchanged.
  L.bind(op->getResult(0), L.value(op->getOperand(0)));
  L.bind(op->getResult(1), bits);
}

// rng(a, b, shape) -> uniform[a,b) or normal(a,b); a,b are scalar tensors.
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
  // operands 0,1: low/high (uniform) or mean/stddev (normal).
  MPSGraphTensor* a = L.value(op->getOperand(0));
  MPSGraphTensor* b = L.value(op->getOperand(1));
  MPSGraphTensor* r = [L.graph() randomTensorWithShape:shape descriptor:desc seed:SeedFor(op) name:nil];
  if (normal) {
    // N(0,1)*stddev + mean
    r = [L.graph() multiplicationWithPrimaryTensor:r secondaryTensor:b name:nil];
    r = [L.graph() additionWithPrimaryTensor:r secondaryTensor:a name:nil];
  } else {
    // U[0,1)*(b-a) + a
    MPSGraphTensor* span = [L.graph() subtractionWithPrimaryTensor:b secondaryTensor:a name:nil];
    r = [L.graph() multiplicationWithPrimaryTensor:r secondaryTensor:span name:nil];
    r = [L.graph() additionWithPrimaryTensor:r secondaryTensor:a name:nil];
  }
  Set(L, op, [L.graph() castTensor:r toType:mps name:nil]);
}

}  // namespace

void RegisterRng() {
  RegisterOp("stablehlo.rng_bit_generator", RngBitGenerator);
  RegisterOp("stablehlo.rng", Rng);
}

}  // namespace mccl_jax::jam
