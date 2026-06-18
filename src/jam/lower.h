#ifndef MCCL_JAX_SRC_JAM_LOWER_H_
#define MCCL_JAX_SRC_JAM_LOWER_H_

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "src/jam/jam.h"

namespace mccl_jax::jam {

std::unique_ptr<CompiledProgram> Lower(mlir::ModuleOp module, std::string& error, int num_processes = 1);

}

#endif
