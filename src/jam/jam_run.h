#ifndef MCCL_JAX_SRC_JAM_JAM_RUN_H_
#define MCCL_JAX_SRC_JAM_JAM_RUN_H_

// Execution shim: run a compiled jam program's MPSGraph on the Metal GPU. Pure-C++ API; body in jam_run.mm.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "src/jam/jam.h"

namespace mccl_jax::jam {

// An input device array: the opaque id<MTLBuffer> handle + its byte size.
struct RunInput {
  void* handle = nullptr;
  std::size_t nbytes = 0;
};

// An output device array: a freshly-allocated Metal UMA buffer; caller owns handle/data.
struct RunOutput {
  void* data = nullptr;
  void* handle = nullptr;
  std::size_t nbytes = 0;
  std::vector<int64_t> dims;
  DType dtype = DType::kInvalid;
};

// Runs one collective: reads `send_count` elements from `send`, writes `recv_count` to `recv`
// (send == recv for in-place all_reduce/broadcast). `root` is the broadcast source; `pairs` is the
// collective_permute source->target routing (empty otherwise). Returns "" on success.
using CollectiveFn = std::function<std::string(CollectiveOp, ReduceKind, DType,
                                               const void* send, void* recv,
                                               std::size_t send_count, std::size_t recv_count,
                                               int root,
                                               const std::vector<std::pair<int, int>>& pairs)>;

// Runs prog on the default Metal device (inputs in @main order, outputs in return order).
// Returns "" on success or an error message (outputs untouched on error).
std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs);

// Segmented overload: same as above, but routes collective steps through `collective`. If the
// program has no collective steps it behaves identically to the single-graph Run.
std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs, const CollectiveFn& collective);

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_JAM_RUN_H_
