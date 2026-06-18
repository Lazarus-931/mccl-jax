#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"

namespace mccl_jax::jam {
namespace {

void Floor(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() floorWithTensor:A(L, op) name:nil]);
}
void Ceil(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() ceilWithTensor:A(L, op) name:nil]);
}
void Sign(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() signWithTensor:A(L, op) name:nil]);
}
void RoundEven(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() rintWithTensor:A(L, op) name:nil]);
}
void RoundAfz(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() roundWithTensor:A(L, op) name:nil]);
}
void Sine(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() sinWithTensor:A(L, op) name:nil]);
}
void Cosine(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() cosWithTensor:A(L, op) name:nil]);
}
void Tan(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() tanWithTensor:A(L, op) name:nil]);
}
void Popcnt(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() bitwisePopulationCountWithTensor:A(L, op) name:nil]);
}

void Expm1(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* e = [L.graph() exponentWithTensor:A(L, op) name:nil];
  MPSGraphTensor* one = [L.graph() constantWithScalar:1.0 dataType:e.dataType];
  Set(L, op, [L.graph() subtractionWithPrimaryTensor:e secondaryTensor:one name:nil]);
}
void Log1p(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* a = A(L, op);
  MPSGraphTensor* one = [L.graph() constantWithScalar:1.0 dataType:a.dataType];
  MPSGraphTensor* s = [L.graph() additionWithPrimaryTensor:a secondaryTensor:one name:nil];
  Set(L, op, [L.graph() logarithmWithTensor:s name:nil]);
}
void Cbrt(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* a = A(L, op);
  MPSGraphTensor* sgn = [L.graph() signWithTensor:a name:nil];
  MPSGraphTensor* mag = [L.graph() absoluteWithTensor:a name:nil];
  MPSGraphTensor* third = [L.graph() constantWithScalar:(1.0 / 3.0) dataType:a.dataType];
  MPSGraphTensor* p = [L.graph() powerWithPrimaryTensor:mag secondaryTensor:third name:nil];
  Set(L, op, [L.graph() multiplicationWithPrimaryTensor:sgn secondaryTensor:p name:nil]);
}
void IsFinite(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* a = A(L, op);
  MPSGraphTensor* inf = [L.graph() isInfiniteWithTensor:a name:nil];
  MPSGraphTensor* nan = [L.graph() isNaNWithTensor:a name:nil];
  MPSGraphTensor* bad = [L.graph() logicalORWithPrimaryTensor:inf secondaryTensor:nan name:nil];
  Set(L, op, [L.graph() notWithTensor:bad name:nil]);
}
void Not(Lowering& L, mlir::Operation* op) {

  MPSGraphTensor* a = A(L, op);
  if (Lowering::ElementType(op->getOperand(0).getType()).isInteger(1))
    Set(L, op, [L.graph() notWithTensor:a name:nil]);
  else
    Set(L, op, [L.graph() bitwiseNOTWithTensor:a name:nil]);
}

void Erf(Lowering& L, mlir::Operation* op)   { Set(L, op, [L.graph() erfWithTensor:A(L, op) name:nil]); }
void Asin(Lowering& L, mlir::Operation* op)  { Set(L, op, [L.graph() asinWithTensor:A(L, op) name:nil]); }
void Acos(Lowering& L, mlir::Operation* op)  { Set(L, op, [L.graph() acosWithTensor:A(L, op) name:nil]); }
void Atan(Lowering& L, mlir::Operation* op)  { Set(L, op, [L.graph() atanWithTensor:A(L, op) name:nil]); }
void Sinh(Lowering& L, mlir::Operation* op)  { Set(L, op, [L.graph() sinhWithTensor:A(L, op) name:nil]); }
void Cosh(Lowering& L, mlir::Operation* op)  { Set(L, op, [L.graph() coshWithTensor:A(L, op) name:nil]); }
void Asinh(Lowering& L, mlir::Operation* op) { Set(L, op, [L.graph() asinhWithTensor:A(L, op) name:nil]); }
void Acosh(Lowering& L, mlir::Operation* op) { Set(L, op, [L.graph() acoshWithTensor:A(L, op) name:nil]); }
void Atanh(Lowering& L, mlir::Operation* op) { Set(L, op, [L.graph() atanhWithTensor:A(L, op) name:nil]); }

}

void RegisterUnary() {
  RegisterOp("stablehlo.floor", Floor);
  RegisterOp("stablehlo.ceil", Ceil);
  RegisterOp("stablehlo.sign", Sign);
  RegisterOp("stablehlo.round_nearest_even", RoundEven);
  RegisterOp("stablehlo.round_nearest_afz", RoundAfz);
  RegisterOp("stablehlo.sine", Sine);
  RegisterOp("stablehlo.cosine", Cosine);
  RegisterOp("stablehlo.tan", Tan);
  RegisterOp("stablehlo.popcnt", Popcnt);
  RegisterOp("stablehlo.exponential_minus_one", Expm1);
  RegisterOp("stablehlo.log_plus_one", Log1p);
  RegisterOp("stablehlo.cbrt", Cbrt);
  RegisterOp("stablehlo.is_finite", IsFinite);
  RegisterOp("stablehlo.not", Not);
  RegisterOp("chlo.erf", Erf);
  RegisterOp("chlo.asin", Asin);
  RegisterOp("chlo.acos", Acos);
  RegisterOp("chlo.atan", Atan);
  RegisterOp("chlo.sinh", Sinh);
  RegisterOp("chlo.cosh", Cosh);
  RegisterOp("chlo.asinh", Asinh);
  RegisterOp("chlo.acosh", Acosh);
  RegisterOp("chlo.atanh", Atanh);
  RegisterOp("chlo.tan", Tan);
}

}
