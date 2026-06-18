#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

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

bool ShapeMatchesResult(mlir::Operation* op, mlir::Value v) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto vt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return rt && vt && rt.getShape() == vt.getShape();
}

mlir::Operation* DefNamed(Lowering& L, mlir::Value v, llvm::StringRef name) {
  mlir::Operation* d = L.resolve(v).getDefiningOp();
  return (d && d->getName().getStringRef() == name) ? d : nullptr;
}
mlir::Value StripBroadcast(Lowering& L, mlir::Value v) {
  for (;;) {
    v = L.resolve(v);
    mlir::Operation* d = v.getDefiningOp();
    if (d && d->getName().getStringRef() == "stablehlo.broadcast_in_dim") { v = d->getOperand(0); continue; }
    return v;
  }
}
bool SplatNegInf(Lowering& L, mlir::Value v) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(StripBroadcast(L, v).getDefiningOp());
  if (!cst) return false;
  auto attr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
  if (!attr || !attr.isSplat()) return false;
  llvm::APFloat f = attr.getSplatValue<llvm::APFloat>();
  return f.isInfinity() && f.isNegative();
}
bool ReduceAxisBody(mlir::Value v, llvm::StringRef bodyName, int64_t* axis, mlir::Value* input) {
  auto red = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(v.getDefiningOp());
  if (!red) return false;
  llvm::ArrayRef<int64_t> dims = red.getDimensions();
  if (dims.size() != 1 || red->getNumOperands() != 2) return false;
  mlir::Operation* body = nullptr;
  for (mlir::Operation& inner : red.getBody().front()) {
    if (inner.getName().getStringRef() == "stablehlo.return") continue;
    body = &inner; break;
  }
  if (!body || body->getName().getStringRef() != bodyName) return false;
  *axis = dims[0]; *input = red.getOperand(0);
  return true;
}
bool TrySoftmax(Lowering& L, mlir::Operation* op) {
  if (getenv("MCCL_JAX_NO_SOFTMAX")) return false;
  mlir::Value num = L.resolve(op->getOperand(0));
  if (!DefNamed(L, num, "stablehlo.exponential")) return false;
  int64_t aSum = 0; mlir::Value sumIn;
  if (!ReduceAxisBody(StripBroadcast(L, op->getOperand(1)), "stablehlo.add", &aSum, &sumIn)) return false;
  if (L.resolve(sumIn) != num) return false;
  mlir::Operation* sub = DefNamed(L, num.getDefiningOp()->getOperand(0), "stablehlo.subtract");
  if (!sub) return false;
  mlir::Value X = L.resolve(sub->getOperand(0));
  mlir::Value mb = StripBroadcast(L, sub->getOperand(1));
  if (mlir::Operation* mx = DefNamed(L, mb, "stablehlo.maximum"))
    mb = StripBroadcast(L, SplatNegInf(L, mx->getOperand(0)) ? mx->getOperand(1) : mx->getOperand(0));
  int64_t aMax = 0; mlir::Value maxIn;
  if (!ReduceAxisBody(mb, "stablehlo.maximum", &aMax, &maxIn)) return false;
  if (aMax != aSum || L.resolve(maxIn) != X) return false;
  Set(L, op, [L.graph() softMaxWithTensor:L.value(X) axis:(NSInteger)aSum name:nil]);
  return true;
}

void Add(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() additionWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Subtract(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() subtractionWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Multiply(Lowering& L, mlir::Operation* op) {

  mlir::Value a = op->getOperand(0), b = op->getOperand(1);
  if (TracesToSplat(L, b, 1.0) && ShapeMatchesResult(op, a)) { Set(L, op, L.value(a)); return; }
  if (TracesToSplat(L, a, 1.0) && ShapeMatchesResult(op, b)) { Set(L, op, L.value(b)); return; }
  Set(L, op, [L.graph() multiplicationWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Divide(Lowering& L, mlir::Operation* op) {
  if (TrySoftmax(L, op)) return;

  if (TracesToSplat(L, op->getOperand(1), 1.0) && ShapeMatchesResult(op, op->getOperand(0))) {
    Set(L, op, A(L, op));
    return;
  }

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
  MPSDataType uty;
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

}

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

}
