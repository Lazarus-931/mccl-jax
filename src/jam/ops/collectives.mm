#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"

namespace mccl_jax::jam {
namespace {

void CollectiveIdentity(Lowering& L, mlir::Operation* op) {
  unsigned n = op->getNumResults();
  for (unsigned i = 0; i < n && i < op->getNumOperands(); ++i)
    L.bind(op->getResult(i), L.value(op->getOperand(i)));
}

void RankId(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() constantWithScalar:0.0 shape:@[ @1 ] dataType:MPSDataTypeUInt32]);
}

}

void RegisterCollectives() {
  RegisterOp("stablehlo.all_reduce", CollectiveIdentity);
  RegisterOp("stablehlo.all_gather", CollectiveIdentity);
  RegisterOp("stablehlo.all_to_all", CollectiveIdentity);
  RegisterOp("stablehlo.reduce_scatter", CollectiveIdentity);
  RegisterOp("stablehlo.collective_broadcast", CollectiveIdentity);
  RegisterOp("stablehlo.collective_permute", CollectiveIdentity);
  RegisterOp("stablehlo.partition_id", RankId);
  RegisterOp("stablehlo.replica_id", RankId);
}

}
