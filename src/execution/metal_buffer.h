#ifndef MCCL_JAX_SRC_EXECUTION_METAL_BUFFER_H_
#define MCCL_JAX_SRC_EXECUTION_METAL_BUFFER_H_

#include <cstddef>

// Unified-memory device allocations (shared-storage MTLBuffer). Pure C++ surface.

namespace mccl_jax::metal {

struct Allocation {
  void* data = nullptr;    // host-addressable UMA pointer (MTLBuffer contents), or null
  void* handle = nullptr;  // opaque retained id<MTLBuffer>; pass to Release()
};

Allocation Allocate(size_t nbytes);  // {nullptr,nullptr} on failure or nbytes == 0
void Release(void* handle);          // safe with nullptr

}  // namespace mccl_jax::metal

#endif  // MCCL_JAX_SRC_EXECUTION_METAL_BUFFER_H_
