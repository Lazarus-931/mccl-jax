#include "src/jam/jam.h"

#include "src/jam/lower.h"
#include "src/jam/parse.h"

namespace mccl_jax::jam {

CompileResult Compile(const char* stablehlo_bytecode, std::size_t size, int num_processes) {
  CompileResult result;

  auto ctx = MakeContext();
  std::string error;
  auto module = Parse(stablehlo_bytecode, size, *ctx, error);
  if (!module) {
    result.error = error;
    return result;
  }

  result.program = Lower(module.get(), error, num_processes);
  if (!result.program) {
    result.error = error.empty() ? "jam: lowering failed" : error;
  }
  return result;
}

}  // namespace mccl_jax::jam
