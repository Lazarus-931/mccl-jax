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

void Clz(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* x = A(L, op);
  MPSDataType dt = x.dataType;

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
  f = orr(f, shr(f, 16));
  MPSGraphTensor* nf = [L.graph() bitwiseNOTWithTensor:f name:nil];
  MPSGraphTensor* pc = [L.graph() bitwisePopulationCountWithTensor:nf name:nil];
  Set(L, op, [L.graph() castTensor:pc toType:dt name:nil]);
}

void ReducePrecision(Lowering& L, mlir::Operation* op) {
  auto rp = mlir::cast<mlir::stablehlo::ReducePrecisionOp>(op);
  int eb = rp.getExponentBits(), mb = rp.getMantissaBits();
  MPSGraphTensor* x = A(L, op);
  MPSDataType src = x.dataType;
  if (eb == 8 && mb == 7) {
    MPSGraphTensor* d = [L.graph() castTensor:x toType:MPSDataTypeBFloat16 name:nil];
    Set(L, op, [L.graph() castTensor:d toType:src name:nil]);
  } else if (eb == 5 && mb == 10) {
    MPSGraphTensor* d = [L.graph() castTensor:x toType:MPSDataTypeFloat16 name:nil];
    Set(L, op, [L.graph() castTensor:d toType:src name:nil]);
  } else {
    Set(L, op, x);
  }
}

void Real(Lowering& L, mlir::Operation* op) { Set(L, op, A(L, op)); }

void Imag(Lowering& L, mlir::Operation* op) {
  MPSGraphTensor* x = A(L, op);
  Set(L, op, [L.graph() multiplicationWithPrimaryTensor:x
      secondaryTensor:[L.graph() constantWithScalar:0.0 dataType:x.dataType] name:nil]);
}

void CustomCall(Lowering& L, mlir::Operation* op) {
  auto cc = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
  llvm::StringRef target = cc.getCallTargetName();
  if (target == "mhlo.erf") {
    Set(L, op, [L.graph() erfWithTensor:A(L, op) name:nil]);
    return;
  }
  if (target == "mhlo.topk" && op->getNumResults() == 2) {
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
      L.substitute(op->getResult(i), op->getOperand(i));
    }
    return;
  }
  L.fail("jam: custom_call target '" + target.str() + "' has no lowering (opaque)");
}

void Unsupported(Lowering& L, mlir::Operation* op) {
  L.fail("jam: op '" + op->getName().getStringRef().str() +
         "' is not supported on the MPSGraph backend (hard limit)");
}

}

void RegisterHardLimit() {
  RegisterOp("stablehlo.count_leading_zeros", Clz);
  RegisterOp("stablehlo.reduce_precision", ReducePrecision);
  RegisterOp("stablehlo.real", Real);
  RegisterOp("stablehlo.imag", Imag);
  RegisterOp("stablehlo.custom_call", CustomCall);

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

}
