#ifndef MCCL_JAX_SRC_EXECUTION_METAL_BUFFER_H_
#define MCCL_JAX_SRC_EXECUTION_METAL_BUFFER_H_

#include <cstddef>

namespace mccl_jax::metal {

struct Allocation {
  void* data = nullptr;
  void* handle = nullptr;
};

Allocation Allocate(size_t nbytes);
void Release(void* handle);

}

#endif
