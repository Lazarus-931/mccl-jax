#ifndef MCCL_JAX_SRC_JAM_PARSE_H_
#define MCCL_JAX_SRC_JAM_PARSE_H_

#include <cstddef>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mccl_jax::jam {

std::unique_ptr<mlir::MLIRContext> MakeContext();

mlir::OwningOpRef<mlir::ModuleOp> Parse(const char* bytecode, std::size_t size,
                                        mlir::MLIRContext& ctx, std::string& error);

}

#endif
