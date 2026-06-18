// ops/hard_limit.mm — hard ops: clz, reduce_precision, real/imag, custom_call; rest fail clearly.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// count_leading_zeros: clz(x) = popcount(~fill(x)), fill propagating the top set bit downward.
void Clz(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* x = A(L, op);
  MPSDataType dt = x.dataType;
  // Work in uint32 to make the shifts logical.
  MPSGraphTensor* u = [L.graph() reinterpretCastTensor:x toType:MPSDataTypeUInt32 name:nil];
  auto shr = [&](MPSGraphTensor* t, int n) {
    MPSGraphTensor* s = [L.graph() constantWithScalar:(double)n dataType:MPSDataTypeUInt32];
    return [L.graph() bitwiseRightShiftWithPrimaryTensor:t secondaryTensor:s name:nil];
  };
  auto orr = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
    return [L.graph() bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
  };
  MPSGraphTensor* f = u;
  f = orr(f, shr(f, 1));
  f = orr(f, shr(f, 2));
  f = orr(f, shr(f, 4));
  f = orr(f, shr(f, 8));
  f = orr(f, shr(f, 16));          // f now has all bits set below+including the top set bit
  MPSGraphTensor* nf = [L.graph() bitwiseNOTWithTensor:f name:nil];
  MPSGraphTensor* pc = [L.graph() bitwisePopulationCountWithTensor:nf name:nil];  // = clz
  Set(L, op, [L.graph() castTensor:pc toType:dt name:nil]);
}

// reduce_precision: round-trip through bf16/f16; f32 and other widths pass through.
void ReducePrecision(Lowering& L, mlir::Operation* op) {
  auto rp = mlir::cast<mlir::stablehlo::ReducePrecisionOp>(op);
  int eb = rp.getExponentBits(), mb = rp.getMantissaBits();
  MPSGraphTensor* x = A(L, op);
  MPSDataType src = x.dataType;
  if (eb == 8 && mb == 7) {        // bf16
    MPSGraphTensor* d = [L.graph() castTensor:x toType:MPSDataTypeBFloat16 name:nil];
    Set(L, op, [L.graph() castTensor:d toType:src name:nil]);
  } else if (eb == 5 && mb == 10) {  // f16
    MPSGraphTensor* d = [L.graph() castTensor:x toType:MPSDataTypeFloat16 name:nil];
    Set(L, op, [L.graph() castTensor:d toType:src name:nil]);
  } else {                          // f32 / unsupported widths → passthrough
    Set(L, op, x);
  }
}

// real(x) of a real input == x (jam has no complex support).
void Real(Lowering& L, mlir::Operation* op) { Set(L, op, A(L, op)); }
// imag(x) of a real input is 0.
void Imag(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* x = A(L, op);
  Set(L, op, [L.graph() multiplicationWithPrimaryTensor:x
      secondaryTensor:[L.graph() constantWithScalar:0.0 dataType:x.dataType] name:nil]);
}

// custom_call: pass through a few known identity targets; any other target is a clear error.
void CustomCall(Lowering& L, mlir::Operation* op) {
  auto cc = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
  llvm::StringRef target = cc.getCallTargetName();
  if (target == "mhlo.erf") {  // exact gelu lowers to erf via a custom_call
    Set(L, op, [L.graph() erfWithTensor:A(L, op) name:nil]);
    return;
  }
  if (target == "mhlo.topk" && op->getNumResults() == 2) {  // top_k over the last axis (values, indices)
    auto resTy = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    NSUInteger k = resTy.getShape().back();
    NSArray<MPSGraphTensor*>* r = [L.graph() topKWithSourceTensor:A(L, op) k:k name:nil];
    L.bind(op->getResult(0), r[0]);
    L.bind(op->getResult(1), Casted(L, r[1], MPSDataTypeInt32));
    return;
  }
  if (target == "Sharding" || target == "SPMDFullToShardShape" ||
      target == "SPMDShardToFullShape" || target == "annotate_device_placement") {
    for (unsigned i = 0; i < op->getNumResults() && i < op->getNumOperands(); ++i) {
      L.bind(op->getResult(i), L.value(op->getOperand(i)));
      L.substitute(op->getResult(i), op->getOperand(i));  // see through for slot routing
    }
    return;
  }
  L.fail("jam: custom_call target '" + target.str() + "' has no lowering (opaque)");
}

// Remaining hard limits: clear named error.
void Unsupported(Lowering& L, mlir::Operation* op) {
  L.fail("jam: op '" + op->getName().getStringRef().str() +
         "' is not supported on the MPSGraph backend (hard limit)");
}

}  // namespace

void RegisterHardLimit() {
  RegisterOp("stablehlo.count_leading_zeros", Clz);
  RegisterOp("stablehlo.reduce_precision", ReducePrecision);
  RegisterOp("stablehlo.real", Real);
  RegisterOp("stablehlo.imag", Imag);
  RegisterOp("stablehlo.custom_call", CustomCall);
  // Genuine hard limits → clear named error (FFT, complex algebra, linear solvers, host I/O).
  RegisterOp("stablehlo.fft", Unsupported);
  RegisterOp("stablehlo.complex", Unsupported);
  RegisterOp("stablehlo.cholesky", Unsupported);
  RegisterOp("stablehlo.triangular_solve", Unsupported);
  RegisterOp("stablehlo.uniform_quantize", Unsupported);
  RegisterOp("stablehlo.uniform_dequantize", Unsupported);
  RegisterOp("stablehlo.infeed", Unsupported);
  RegisterOp("stablehlo.outfeed", Unsupported);
  RegisterOp("stablehlo.send", Unsupported);
  RegisterOp("stablehlo.recv", Unsupported);
}

}  // namespace mccl_jax::jam
