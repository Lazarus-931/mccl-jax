#ifndef MCCL_JAX_SRC_JAM_JAM_RUN_H_
#define MCCL_JAX_SRC_JAM_JAM_RUN_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "src/jam/jam.h"

namespace mccl_jax::jam {

struct RunInput {
  void* handle = nullptr;
  std::size_t nbytes = 0;
};

struct RunOutput {
  void* data = nullptr;
  void* handle = nullptr;
  std::size_t nbytes = 0;
  std::vector<int64_t> dims;
  DType dtype = DType::kInvalid;
};

using CollectiveFn = std::function<std::string(CollectiveOp, ReduceKind, DType,
                                               const void* send, void* recv,
                                               std::size_t send_count, std::size_t recv_count,
                                               int root,
                                               const std::vector<std::pair<int, int>>& pairs)>;

std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs);

std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs, const CollectiveFn& collective);

void RecycleDeviceBuffer(void* handle, void* data, std::size_t nbytes);

}

#endif
