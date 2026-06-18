// ops/elementwise.mm — basic StableHLO elementwise arithmetic → MPSGraph.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// True if `v` traces (through reshape/broadcast/convert) to a SPLAT float constant equal to `target`.
// Used to elide no-op ×1.0 / ÷1.0 — v*1==v and v/1==v EXACTLY for every v (incl. nan/inf/-0), and
// MPSGraph doesn't fold them, so they're otherwise dead kernels. Float-only (won't touch int ops).
bool TracesToSplat(Lowering& L, mlir::Value v, double target, int depth = 0) {
  if (depth > 12) return false;
  v = L.resolve(v);
  mlir::Operation* def = v.getDefiningOp();
  if (!def) return false;
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
    auto attr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
    return attr && attr.isSplat() && attr.getSplatValue<llvm::APFloat>().convertToDouble() == target;
  }
  llvm::StringRef n = def->getName().getStringRef();
  if (n == "stablehlo.reshape" || n == "stablehlo.broadcast_in_dim" || n == "stablehlo.convert")
    return TracesToSplat(L, def->getOperand(0), target, depth + 1);
  return false;
}
// `v` has the same shape as op's result (so eliding the op doesn't silently drop a broadcast of v).
bool ShapeMatchesResult(mlir::Operation* op, mlir::Value v) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto vt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return rt && vt && rt.getShape() == vt.getShape();
}

// binary
void Add(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() additionWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Subtract(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() subtractionWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Multiply(Lowering& L, mlir::Operation* op) {
  // Elide ×1.0 (no-op MPSGraph won't fold): if one operand is a splat-1 and the other keeps the
  // result shape, bind the other directly.
  mlir::Value a = op->getOperand(0), b = op->getOperand(1);
  if (TracesToSplat(L, b, 1.0) && ShapeMatchesResult(op, a)) { Set(L, op, L.value(a)); return; }
  if (TracesToSplat(L, a, 1.0) && ShapeMatchesResult(op, b)) { Set(L, op, L.value(b)); return; }
  Set(L, op, [L.graph() multiplicationWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Divide(Lowering& L, mlir::Operation* op) {
  // Elide ÷1.0 (v/1==v exactly): if the divisor is a splat-1 and the result keeps the dividend shape.
  if (TracesToSplat(L, op->getOperand(1), 1.0) && ShapeMatchesResult(op, op->getOperand(0))) {
    Set(L, op, A(L, op));
    return;
  }
  // UNSIGNED integer division: jam backs ui with a signed int, so plain division would be signed
  // (high-bit values divide wrong). Reinterpret to the unsigned type, divide, reinterpret back.
  // (float/signed division is unchanged.)
  MPSDataType uty;
  if (UnsignedIntOperand(op, 0, uty)) {
    MPSGraphTensor* au = [L.graph() reinterpretCastTensor:A(L, op) toType:uty name:nil];
    MPSGraphTensor* bu = [L.graph() reinterpretCastTensor:B(L, op) toType:uty name:nil];
    MPSGraphTensor* q = [L.graph() divisionWithPrimaryTensor:au secondaryTensor:bu name:nil];
    Set(L, op, [L.graph() reinterpretCastTensor:q
                                          toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType()))
                                            name:nil]);
    return;
  }
  Set(L, op, [L.graph() divisionWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Maximum(Lowering& L, mlir::Operation* op) {
  MPSDataType uty;  // unsigned max: reinterpret to unsigned (signed max picks wrong for high-bit values)
  if (UnsignedIntOperand(op, 0, uty)) {
    MPSGraphTensor* r = [L.graph() maximumWithPrimaryTensor:[L.graph() reinterpretCastTensor:A(L, op) toType:uty name:nil]
                                            secondaryTensor:[L.graph() reinterpretCastTensor:B(L, op) toType:uty name:nil] name:nil];
    Set(L, op, [L.graph() reinterpretCastTensor:r toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType())) name:nil]);
    return;
  }
  Set(L, op, [L.graph() maximumWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Minimum(Lowering& L, mlir::Operation* op) {
  MPSDataType uty;
  if (UnsignedIntOperand(op, 0, uty)) {
    MPSGraphTensor* r = [L.graph() minimumWithPrimaryTensor:[L.graph() reinterpretCastTensor:A(L, op) toType:uty name:nil]
                                            secondaryTensor:[L.graph() reinterpretCastTensor:B(L, op) toType:uty name:nil] name:nil];
    Set(L, op, [L.graph() reinterpretCastTensor:r toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType())) name:nil]);
    return;
  }
  Set(L, op, [L.graph() minimumWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Power(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() powerWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}

// unary
void Negate(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() negativeWithTensor:A(L, op) name:nil]);
}
void Exp(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() exponentWithTensor:A(L, op) name:nil]);
}
void Log(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() logarithmWithTensor:A(L, op) name:nil]);
}
void Tanh(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() tanhWithTensor:A(L, op) name:nil]);
}
void Sqrt(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() squareRootWithTensor:A(L, op) name:nil]);
}
void Rsqrt(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() reciprocalSquareRootWithTensor:A(L, op) name:nil]);
}
void Abs(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() absoluteWithTensor:A(L, op) name:nil]);
}
void Logistic(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() sigmoidWithTensor:A(L, op) name:nil]);
}

}  // namespace

void RegisterElementwise() {
  RegisterOp("stablehlo.add", Add);
  RegisterOp("stablehlo.subtract", Subtract);
  RegisterOp("stablehlo.multiply", Multiply);
  RegisterOp("stablehlo.divide", Divide);
  RegisterOp("stablehlo.maximum", Maximum);
  RegisterOp("stablehlo.minimum", Minimum);
  RegisterOp("stablehlo.power", Power);
  RegisterOp("stablehlo.negate", Negate);
  RegisterOp("stablehlo.exponential", Exp);
  RegisterOp("stablehlo.log", Log);
  RegisterOp("stablehlo.tanh", Tanh);
  RegisterOp("stablehlo.sqrt", Sqrt);
  RegisterOp("stablehlo.rsqrt", Rsqrt);
  RegisterOp("stablehlo.abs", Abs);
  RegisterOp("stablehlo.logistic", Logistic);
}

}  // namespace mccl_jax::jam
