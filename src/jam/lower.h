#ifndef MCCL_JAX_SRC_JAM_LOWER_H_
#define MCCL_JAX_SRC_JAM_LOWER_H_

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "src/jam/jam.h"

namespace mccl_jax::jam {

// Walk a StableHLO module's @main and emit an MPSGraph. Null + sets `error` on failure.
// num_processes = cluster size (1 ⇒ collectives are identity, elided without a segment boundary).
std::unique_ptr<CompiledProgram> Lower(mlir::ModuleOp module, std::string& error, int num_processes = 1);

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_LOWER_H_
