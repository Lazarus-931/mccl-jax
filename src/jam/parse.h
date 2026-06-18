#ifndef MCCL_JAX_SRC_JAM_PARSE_H_
#define MCCL_JAX_SRC_JAM_PARSE_H_

#include <cstddef>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mccl_jax::jam {

// An MLIRContext with the dialects a StableHLO artifact references loaded.
std::unique_ptr<mlir::MLIRContext> MakeContext();

// Deserialize a StableHLO portable artifact into a module. Null + sets `error` on failure.
mlir::OwningOpRef<mlir::ModuleOp> Parse(const char* bytecode, std::size_t size,
                                        mlir::MLIRContext& ctx, std::string& error);

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_PARSE_H_
