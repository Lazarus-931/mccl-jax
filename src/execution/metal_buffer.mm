#import <Metal/Metal.h>

#include "src/execution/metal_buffer.h"

namespace mccl_jax::metal {
namespace {
id<MTLDevice> DefaultDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}
}

Allocation Allocate(size_t nbytes) {
  if (nbytes == 0) return {};
  id<MTLDevice> device = DefaultDevice();
  if (device == nil) return {};
  id<MTLBuffer> buffer = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  if (buffer == nil) return {};
  Allocation a;
  a.data = [buffer contents];
  a.handle = (__bridge_retained void*)buffer;
  return a;
}

void Release(void* handle) {
  if (handle == nullptr) return;
  id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle;
  (void)buffer;
}

}
