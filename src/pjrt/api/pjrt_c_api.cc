// mccl-jax PJRT plugin — implementation of the PJRT C API.
// Implemented entry points translate C args to/from the internal model; the rest are stubs.

#include "src/pjrt/api/pjrt_c_api.h"
#include "src/execution/metal_buffer.h"

#ifdef MCCL_JAX_WITH_JAM
// jam (StableHLO -> MPSGraph) + the execution shim; compiled in only when built with jam.
#include "src/jam/jam.h"
#include "src/jam/jam_run.h"
// mccl collective layer: the multi-host all_reduce backend.
#include "src/mccl/collective/comm.h"
#include "src/mccl/collective/collectives.h"
#endif

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct PJRT_Error {
  PJRT_Error_Code code = PJRT_Error_Code_UNKNOWN;
  std::string message;
};

struct PJRT_DeviceDescription {
  int id = 0;
  int process_index = 0;
  std::string kind = "Metal";
  std::string debug_string;
  std::string to_string;
};

struct PJRT_Device {
  PJRT_DeviceDescription description;
  PJRT_Client* client = nullptr;
  int local_hardware_id = 0;
  bool addressable = true;                // false for remote ranks in a multi-host mesh
  std::vector<PJRT_Memory*> memories;     // MPS + CPU
  PJRT_Memory* default_memory = nullptr;  // MPS
};

// MPS (Metal GPU) and CPU (host) memory spaces; both UMA on Apple Silicon.
struct PJRT_Memory {
  int id = 0;
  std::string kind;  // "device" (MPS) | "pinned_host" (CPU)
  int kind_id = 0;
  std::string debug_string;
  std::string to_string;
  std::vector<PJRT_Device*> devices;
};

struct PJRT_TopologyDescription {
  std::string platform_name = "metal";
  std::string platform_version = "mccl-jax 0.0.1 (PJRT 0.55)";
  std::vector<std::unique_ptr<PJRT_DeviceDescription>> owned;  // for standalone Create
  std::vector<PJRT_DeviceDescription*> descriptions;
};

struct PJRT_Client {
  std::string platform_name = "metal";
  std::string platform_version = "mccl-jax 0.0.1 (PJRT 0.55)";
  int process_index = 0;                  // this host's rank (node_id)
  int num_processes = 1;                   // cluster size (num_nodes)
  std::vector<std::unique_ptr<PJRT_Device>> owned_devices;
  std::vector<PJRT_Device*> devices;              // all ranks' devices (the global mesh)
  std::vector<PJRT_Device*> addressable_devices;  // just this host's
  std::vector<std::unique_ptr<PJRT_Memory>> owned_memories;
  std::vector<PJRT_Memory*> memories;             // this host's (addressable) memories
  std::unique_ptr<PJRT_TopologyDescription> topology;
#ifdef MCCL_JAX_WITH_JAM
  std::unique_ptr<mccl_collective::Comm> comm;    // mccl communicator (null when num_processes == 1)
#endif
};

// Async completion future; synchronous ops hand back an already-ready event.
struct PJRT_Event {
  std::mutex mu;
  std::condition_variable cv;
  bool ready = false;
  PJRT_Error* error = nullptr;  // owned; nullptr == success; valid once ready
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> on_ready;
};

struct PJRT_ExecuteContext {};

// Device array backed by a shared-storage MTLBuffer (UMA). External refs defer the free past Delete.
struct PJRT_Buffer {
  PJRT_Client* client = nullptr;
  PJRT_Device* device = nullptr;
  PJRT_Memory* memory = nullptr;
  PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  size_t nbytes = 0;
  void* data = nullptr;    // MTLBuffer contents (host-addressable UMA)
  void* handle = nullptr;  // opaque id<MTLBuffer>
  int external_refs = 0;
  bool delete_requested = false;
  bool deleted = false;
};

#ifdef MCCL_JAX_WITH_JAM
// PJRT_Executable is the introspection view; PJRT_LoadedExecutable is the runnable handle.
// Both share the CompiledProgram.
struct PJRT_Executable {
  std::shared_ptr<mccl_jax::jam::CompiledProgram> program;
  std::string name = "jam.main";
  std::vector<PJRT_Buffer_Type> out_types;
  std::vector<int64_t> out_dims_flat;     // all outputs' dims concatenated
  std::vector<size_t> out_dim_sizes;      // rank per output
  std::vector<std::string> out_mem_kinds;
  std::vector<const char*> out_mem_kind_ptrs;
  std::vector<size_t> out_mem_kind_sizes;
};
struct PJRT_LoadedExecutable {
  PJRT_Client* client = nullptr;
  std::shared_ptr<mccl_jax::jam::CompiledProgram> program;
  std::vector<PJRT_Device*> devices;
  bool deleted = false;
};
#endif

namespace {

PJRT_Error* MakeUnimplemented(const char* name) {
  return new PJRT_Error{PJRT_Error_Code_UNIMPLEMENTED,
                        std::string(name) + " is not implemented in mccl-jax yet"};
}

// Build the client for one host of a `num_nodes`-host cluster (one Metal device per host;
// device k belongs to rank k, only this host's is addressable).
PJRT_Client* CreateClient(int num_nodes, int node_id) {
  auto client = std::make_unique<PJRT_Client>();
  client->process_index = node_id;
  client->num_processes = num_nodes;

  for (int k = 0; k < num_nodes; ++k) {
    const std::string ks = std::to_string(k);
    auto device = std::make_unique<PJRT_Device>();
    device->client = client.get();
    device->addressable = (k == node_id);
    device->local_hardware_id = device->addressable ? 0 : -1;
    device->description.id = k;
    device->description.process_index = k;
    device->description.to_string = "MetalDevice(id=" + ks + ")";
    device->description.debug_string = "metal:" + ks;
    PJRT_Device* dev = device.get();

    auto mps = std::make_unique<PJRT_Memory>();
    mps->id = k * 2; mps->kind = "device"; mps->kind_id = 0;
    mps->debug_string = "mps:" + ks; mps->to_string = "MpsMemory(id=" + ks + ")";
    mps->devices.push_back(dev);
    auto cpu = std::make_unique<PJRT_Memory>();
    cpu->id = k * 2 + 1; cpu->kind = "pinned_host"; cpu->kind_id = 1;
    cpu->debug_string = "cpu:" + ks; cpu->to_string = "CpuMemory(id=" + ks + ")";
    cpu->devices.push_back(dev);

    dev->memories = {mps.get(), cpu.get()};
    dev->default_memory = mps.get();
    if (dev->addressable) {
      client->memories = {mps.get(), cpu.get()};  // addressable memories = this host's
      client->addressable_devices.push_back(dev);
    }
    client->owned_memories.push_back(std::move(mps));
    client->owned_memories.push_back(std::move(cpu));
    client->devices.push_back(dev);
    client->owned_devices.push_back(std::move(device));
  }

  auto topo = std::make_unique<PJRT_TopologyDescription>();
  for (PJRT_Device* d : client->devices) topo->descriptions.push_back(&d->description);
  client->topology = std::move(topo);
  return client.release();
}

void ReleaseStorage(PJRT_Buffer* b) {
  if (b->handle) {
    mccl_jax::metal::Release(b->handle);
    b->handle = nullptr;
    b->data = nullptr;
  }
}

// ---- errors ----
void ErrorDestroy(PJRT_Error_Destroy_Args* a) { delete a->error; }
void ErrorMessage(PJRT_Error_Message_Args* a) {
  a->message = a->error->message.c_str();
  a->message_size = a->error->message.size();
}
PJRT_Error* ErrorGetCode(PJRT_Error_GetCode_Args* a) {
  a->code = a->error->code;
  return nullptr;
}

// ---- events ----  (returned PJRT_Error*s are independent copies the caller frees)
PJRT_Event* MakeReadyEvent(PJRT_Error* error = nullptr) {
  auto* e = new PJRT_Event();
  e->ready = true;
  e->error = error;
  return e;
}
PJRT_Error* EventDestroy(PJRT_Event_Destroy_Args* a) {
  delete a->event->error;
  delete a->event;
  return nullptr;
}
PJRT_Error* EventIsReady(PJRT_Event_IsReady_Args* a) {
  std::lock_guard<std::mutex> lock(a->event->mu);
  a->is_ready = a->event->ready;
  return nullptr;
}
PJRT_Error* EventError(PJRT_Event_Error_Args* a) {
  std::lock_guard<std::mutex> lock(a->event->mu);
  return a->event->error ? new PJRT_Error(*a->event->error) : nullptr;
}
PJRT_Error* EventAwait(PJRT_Event_Await_Args* a) {
  std::unique_lock<std::mutex> lock(a->event->mu);
  a->event->cv.wait(lock, [&] { return a->event->ready; });
  return a->event->error ? new PJRT_Error(*a->event->error) : nullptr;
}
PJRT_Error* EventOnReady(PJRT_Event_OnReady_Args* a) {
  std::unique_lock<std::mutex> lock(a->event->mu);
  if (a->event->ready) {
    PJRT_Error* err = a->event->error ? new PJRT_Error(*a->event->error) : nullptr;
    lock.unlock();
    a->callback(err, a->user_arg);
  } else {
    a->event->on_ready.emplace_back(a->callback, a->user_arg);
  }
  return nullptr;
}

// ---- plugin ----
PJRT_Error* PluginInitialize(PJRT_Plugin_Initialize_Args*) { return nullptr; }
PJRT_Error* PluginAttributes(PJRT_Plugin_Attributes_Args* a) {
  a->attributes = nullptr;
  a->num_attributes = 0;
  return nullptr;
}

// ---- client ----
PJRT_Error* ClientCreate(PJRT_Client_Create_Args* a) {
  // node_id / num_nodes come from create_options (jax.distributed.initialize); absent => single host.
  int node_id = 0, num_nodes = 1;
  for (size_t i = 0; i < a->num_options; ++i) {
    const PJRT_NamedValue& nv = a->create_options[i];
    const std::string name(nv.name, nv.name_size);
    if (nv.type == PJRT_NamedValue_kInt64) {
      if (name == "node_id") node_id = static_cast<int>(nv.int64_value);
      else if (name == "num_nodes") num_nodes = static_cast<int>(nv.int64_value);
    }
  }
  if (num_nodes < 1) num_nodes = 1;
  if (node_id < 0 || node_id >= num_nodes) node_id = 0;
  PJRT_Client* client = CreateClient(num_nodes, node_id);
#ifdef MCCL_JAX_WITH_JAM
  // Multi-host: do the mccl rendezvous now (blocks until all ranks; reads MCCL_BOOTSTRAP_IP/PORT).
  if (num_nodes > 1) {
    std::string err;
    client->comm = mccl_collective::Comm::Create(num_nodes, node_id, &err);
    if (!client->comm) {
      delete client;
      return new PJRT_Error{PJRT_Error_Code_INTERNAL, "mccl comm init failed: " + err};
    }
  }
#endif
  a->client = client;
  return nullptr;
}
PJRT_Error* ClientDestroy(PJRT_Client_Destroy_Args* a) {
  delete a->client;
  return nullptr;
}
PJRT_Error* ClientPlatformName(PJRT_Client_PlatformName_Args* a) {
  a->platform_name = a->client->platform_name.c_str();
  a->platform_name_size = a->client->platform_name.size();
  return nullptr;
}
PJRT_Error* ClientPlatformVersion(PJRT_Client_PlatformVersion_Args* a) {
  a->platform_version = a->client->platform_version.c_str();
  a->platform_version_size = a->client->platform_version.size();
  return nullptr;
}
PJRT_Error* ClientProcessIndex(PJRT_Client_ProcessIndex_Args* a) {
  a->process_index = a->client->process_index;
  return nullptr;
}
PJRT_Error* ClientDevices(PJRT_Client_Devices_Args* a) {
  a->devices = a->client->devices.data();
  a->num_devices = a->client->devices.size();
  return nullptr;
}
PJRT_Error* ClientAddressableDevices(PJRT_Client_AddressableDevices_Args* a) {
  a->addressable_devices = a->client->addressable_devices.data();
  a->num_addressable_devices = a->client->addressable_devices.size();
  return nullptr;
}
PJRT_Error* ClientAddressableMemories(PJRT_Client_AddressableMemories_Args* a) {
  a->addressable_memories = a->client->memories.data();
  a->num_addressable_memories = a->client->memories.size();
  return nullptr;
}
PJRT_Error* ClientLookupDevice(PJRT_Client_LookupDevice_Args* a) {
  for (PJRT_Device* d : a->client->devices)
    if (d->description.id == a->id) { a->device = d; return nullptr; }
  return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "no device with that id"};
}
PJRT_Error* ClientLookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* a) {
  for (PJRT_Device* d : a->client->devices)
    if (d->local_hardware_id == a->local_hardware_id) { a->addressable_device = d; return nullptr; }
  return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "no addressable device with that local hardware id"};
}
PJRT_Error* ClientDefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* a) {
  const size_t n = static_cast<size_t>(a->num_replicas) * static_cast<size_t>(a->num_partitions);
  for (size_t i = 0; i < n && i < a->default_assignment_size; ++i) a->default_assignment[i] = 0;
  return nullptr;
}
PJRT_Error* ClientTopologyDescription(PJRT_Client_TopologyDescription_Args* a) {
  a->topology = a->client->topology.get();
  return nullptr;
}

// ---- device ----
PJRT_Error* DeviceGetDescription(PJRT_Device_GetDescription_Args* a) {
  a->device_description = &a->device->description;
  return nullptr;
}
PJRT_Error* DeviceIsAddressable(PJRT_Device_IsAddressable_Args* a) {
  a->is_addressable = a->device->addressable;
  return nullptr;
}
PJRT_Error* DeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args* a) {
  a->local_hardware_id = a->device->local_hardware_id;
  return nullptr;
}
PJRT_Error* DeviceAddressableMemories(PJRT_Device_AddressableMemories_Args* a) {
  a->memories = a->device->memories.data();
  a->num_memories = a->device->memories.size();
  return nullptr;
}
PJRT_Error* DeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* a) {
  a->memory = a->device->default_memory;
  return nullptr;
}
PJRT_Error* DeviceMemoryStats(PJRT_Device_MemoryStats_Args* a) {
  // We don't track allocator stats; report 0 in-use and leave the rest unset.
  a->bytes_in_use = 0;
  a->peak_bytes_in_use_is_set = false;
  a->num_allocs_is_set = false;
  a->largest_alloc_size_is_set = false;
  a->bytes_limit_is_set = false;
  a->bytes_reserved_is_set = false;
  a->peak_bytes_reserved_is_set = false;
  a->bytes_reservable_limit_is_set = false;
  a->largest_free_block_bytes_is_set = false;
  a->pool_bytes_is_set = false;
  a->peak_pool_bytes_is_set = false;
  return nullptr;
}

// ---- device description ----
PJRT_Error* DescId(PJRT_DeviceDescription_Id_Args* a) {
  a->id = a->device_description->id;
  return nullptr;
}
PJRT_Error* DescProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* a) {
  a->process_index = a->device_description->process_index;
  return nullptr;
}
PJRT_Error* DescAttributes(PJRT_DeviceDescription_Attributes_Args* a) {
  a->num_attributes = 0;
  a->attributes = nullptr;
  return nullptr;
}
PJRT_Error* DescKind(PJRT_DeviceDescription_Kind_Args* a) {
  a->device_kind = a->device_description->kind.c_str();
  a->device_kind_size = a->device_description->kind.size();
  return nullptr;
}
PJRT_Error* DescDebugString(PJRT_DeviceDescription_DebugString_Args* a) {
  a->debug_string = a->device_description->debug_string.c_str();
  a->debug_string_size = a->device_description->debug_string.size();
  return nullptr;
}
PJRT_Error* DescToString(PJRT_DeviceDescription_ToString_Args* a) {
  a->to_string = a->device_description->to_string.c_str();
  a->to_string_size = a->device_description->to_string.size();
  return nullptr;
}

// ---- memory spaces ----
PJRT_Error* MemoryId(PJRT_Memory_Id_Args* a) { a->id = a->memory->id; return nullptr; }
PJRT_Error* MemoryKind(PJRT_Memory_Kind_Args* a) {
  a->kind = a->memory->kind.c_str();
  a->kind_size = a->memory->kind.size();
  return nullptr;
}
PJRT_Error* MemoryKindId(PJRT_Memory_Kind_Id_Args* a) { a->kind_id = a->memory->kind_id; return nullptr; }
PJRT_Error* MemoryDebugString(PJRT_Memory_DebugString_Args* a) {
  a->debug_string = a->memory->debug_string.c_str();
  a->debug_string_size = a->memory->debug_string.size();
  return nullptr;
}
PJRT_Error* MemoryToString(PJRT_Memory_ToString_Args* a) {
  a->to_string = a->memory->to_string.c_str();
  a->to_string_size = a->memory->to_string.size();
  return nullptr;
}
PJRT_Error* MemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args* a) {
  a->devices = a->memory->devices.data();
  a->num_devices = a->memory->devices.size();
  return nullptr;
}

// ---- topology ----
PJRT_Error* TopologyCreate(PJRT_TopologyDescription_Create_Args* a) {
  auto* t = new PJRT_TopologyDescription();
  auto d = std::make_unique<PJRT_DeviceDescription>();
  d->to_string = "MetalDevice(id=0)";
  d->debug_string = "metal:0";
  t->descriptions.push_back(d.get());
  t->owned.push_back(std::move(d));
  a->topology = t;
  return nullptr;
}
PJRT_Error* TopologyDestroy(PJRT_TopologyDescription_Destroy_Args* a) {
  delete a->topology;
  return nullptr;
}
PJRT_Error* TopologyPlatformName(PJRT_TopologyDescription_PlatformName_Args* a) {
  a->platform_name = a->topology->platform_name.c_str();
  a->platform_name_size = a->topology->platform_name.size();
  return nullptr;
}
PJRT_Error* TopologyPlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* a) {
  a->platform_version = a->topology->platform_version.c_str();
  a->platform_version_size = a->topology->platform_version.size();
  return nullptr;
}
PJRT_Error* TopologyGetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* a) {
  a->descriptions = a->topology->descriptions.data();
  a->num_descriptions = a->topology->descriptions.size();
  return nullptr;
}
PJRT_Error* TopologyAttributes(PJRT_TopologyDescription_Attributes_Args* a) {
  a->attributes = nullptr;
  a->num_attributes = 0;
  return nullptr;
}

// ---- execute context ----
PJRT_Error* ExecuteContextCreate(PJRT_ExecuteContext_Create_Args* a) {
  a->context = new PJRT_ExecuteContext();
  return nullptr;
}
PJRT_Error* ExecuteContextDestroy(PJRT_ExecuteContext_Destroy_Args* a) {
  delete a->context;
  return nullptr;
}

// ---- buffers (Metal unified memory; see src/execution) ----
size_t ByteWidth(PJRT_Buffer_Type t) {
  switch (t) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_S8: case PJRT_Buffer_Type_U8:
    case PJRT_Buffer_Type_F8E5M2: case PJRT_Buffer_Type_F8E4M3FN:
    case PJRT_Buffer_Type_F8E4M3B11FNUZ: case PJRT_Buffer_Type_F8E5M2FNUZ:
    case PJRT_Buffer_Type_F8E4M3FNUZ: case PJRT_Buffer_Type_F8E4M3: case PJRT_Buffer_Type_F8E3M4:
      return 1;
    case PJRT_Buffer_Type_S16: case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16: case PJRT_Buffer_Type_BF16:
      return 2;
    case PJRT_Buffer_Type_S32: case PJRT_Buffer_Type_U32: case PJRT_Buffer_Type_F32:
      return 4;
    case PJRT_Buffer_Type_S64: case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64: case PJRT_Buffer_Type_C64:
      return 8;
    case PJRT_Buffer_Type_C128:
      return 16;
    default:
      return 0;
  }
}

PJRT_Buffer* AllocBufferLike(PJRT_Buffer* src, PJRT_Device* device, PJRT_Memory* memory) {
  mccl_jax::metal::Allocation alloc = mccl_jax::metal::Allocate(src->nbytes);
  if (src->nbytes != 0 && alloc.data == nullptr) return nullptr;
  if (src->nbytes != 0) std::memcpy(alloc.data, src->data, src->nbytes);
  auto* b = new PJRT_Buffer();
  b->client = src->client;
  b->device = device;
  b->memory = memory;
  b->type = src->type;
  b->dims = src->dims;
  b->nbytes = src->nbytes;
  b->data = alloc.data;
  b->handle = alloc.handle;
  return b;
}

PJRT_Error* ClientBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* a) {
  const size_t width = ByteWidth(a->type);
  if (width == 0)
    return new PJRT_Error{PJRT_Error_Code_UNIMPLEMENTED, "BufferFromHostBuffer: unsupported element type"};
  size_t count = 1;
  for (size_t i = 0; i < a->num_dims; ++i) count *= static_cast<size_t>(a->dims[i]);
  const size_t nbytes = count * width;

  mccl_jax::metal::Allocation alloc = mccl_jax::metal::Allocate(nbytes);
  if (nbytes != 0 && alloc.data == nullptr)
    return new PJRT_Error{PJRT_Error_Code_RESOURCE_EXHAUSTED, "BufferFromHostBuffer: Metal allocation failed"};
  if (nbytes != 0) std::memcpy(alloc.data, a->data, nbytes);

  // JAX may target a memory space (device null) or a device (memory null).
  PJRT_Device* device = a->device;
  PJRT_Memory* memory = a->memory;
  if (device == nullptr && memory != nullptr && !memory->devices.empty()) device = memory->devices.front();
  if (memory == nullptr && device != nullptr) memory = device->default_memory;

  auto* buf = new PJRT_Buffer();
  buf->client = a->client;
  buf->device = device;
  buf->memory = memory;
  buf->type = a->type;
  buf->dims.assign(a->dims, a->dims + a->num_dims);
  buf->nbytes = nbytes;
  buf->data = alloc.data;
  buf->handle = alloc.handle;
  a->done_with_host_buffer = MakeReadyEvent();
  a->buffer = buf;
  return nullptr;
}

PJRT_Error* BufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* a) {
  if (a->dst == nullptr) {  // size query
    a->dst_size = a->src->nbytes;
    a->event = MakeReadyEvent();
    return nullptr;
  }
  if (a->dst_size < a->src->nbytes)
    return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "ToHostBuffer: dst_size too small"};
  if (a->src->nbytes != 0) std::memcpy(a->dst, a->src->data, a->src->nbytes);
  a->event = MakeReadyEvent();
  return nullptr;
}

PJRT_Error* BufferCopyToMemory(PJRT_Buffer_CopyToMemory_Args* a) {
  if (a->buffer->deleted) return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "CopyToMemory: deleted buffer"};
  PJRT_Memory* mem = a->dst_memory;
  PJRT_Device* dev = (mem && !mem->devices.empty()) ? mem->devices.front() : a->buffer->device;
  PJRT_Buffer* nb = AllocBufferLike(a->buffer, dev, mem);
  if (!nb) return new PJRT_Error{PJRT_Error_Code_RESOURCE_EXHAUSTED, "CopyToMemory: allocation failed"};
  a->dst_buffer = nb;
  return nullptr;
}
PJRT_Error* BufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args* a) {
  if (a->buffer->deleted) return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "CopyToDevice: deleted buffer"};
  PJRT_Device* dev = a->dst_device;
  PJRT_Buffer* nb = AllocBufferLike(a->buffer, dev, dev ? dev->default_memory : nullptr);
  if (!nb) return new PJRT_Error{PJRT_Error_Code_RESOURCE_EXHAUSTED, "CopyToDevice: allocation failed"};
  a->dst_buffer = nb;
  return nullptr;
}

PJRT_Error* BufferDestroy(PJRT_Buffer_Destroy_Args* a) {
  ReleaseStorage(a->buffer);
  delete a->buffer;
  return nullptr;
}
PJRT_Error* BufferDelete(PJRT_Buffer_Delete_Args* a) {
  a->buffer->deleted = true;
  if (a->buffer->external_refs == 0) ReleaseStorage(a->buffer);
  else a->buffer->delete_requested = true;  // keep alive for the external holder (mccl)
  return nullptr;
}
PJRT_Error* BufferIsDeleted(PJRT_Buffer_IsDeleted_Args* a) {
  a->is_deleted = a->buffer->deleted;
  return nullptr;
}
PJRT_Error* BufferElementType(PJRT_Buffer_ElementType_Args* a) {
  a->type = a->buffer->type;
  return nullptr;
}
PJRT_Error* BufferDimensions(PJRT_Buffer_Dimensions_Args* a) {
  a->dims = a->buffer->dims.data();
  a->num_dims = a->buffer->dims.size();
  return nullptr;
}
PJRT_Error* BufferUnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* a) {
  a->unpadded_dims = a->buffer->dims.data();
  a->num_dims = a->buffer->dims.size();
  return nullptr;
}
PJRT_Error* BufferDynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* a) {
  a->dynamic_dim_indices = nullptr;
  a->num_dynamic_dims = 0;
  return nullptr;
}
PJRT_Error* BufferOnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* a) {
  a->on_device_size_in_bytes = a->buffer->nbytes;
  return nullptr;
}
PJRT_Error* BufferDevice(PJRT_Buffer_Device_Args* a) {
  a->device = a->buffer->device;
  return nullptr;
}
PJRT_Error* BufferMemory(PJRT_Buffer_Memory_Args* a) {
  a->memory = a->buffer->memory;
  return nullptr;
}
PJRT_Error* BufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* a) {
  a->is_on_cpu = false;
  return nullptr;
}
PJRT_Error* BufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* a) {
  a->event = MakeReadyEvent();
  return nullptr;
}
PJRT_Error* BufferUnsafePointer(PJRT_Buffer_UnsafePointer_Args* a) {
  a->buffer_pointer = reinterpret_cast<uintptr_t>(a->buffer->data);  // UMA host pointer
  return nullptr;
}
PJRT_Error* BufferOpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* a) {
  a->device_memory_ptr = a->buffer->data;  // mccl zero-copy seam
  return nullptr;
}
PJRT_Error* BufferIncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* a) {
  ++a->buffer->external_refs;
  return nullptr;
}
PJRT_Error* BufferDecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* a) {
  if (a->buffer->external_refs > 0) --a->buffer->external_refs;
  if (a->buffer->external_refs == 0 && a->buffer->delete_requested) ReleaseStorage(a->buffer);
  return nullptr;
}

#ifdef MCCL_JAX_WITH_JAM
// ---- compile + execute (jam) ------------------------------------------------
PJRT_Buffer_Type JamPjrtType(mccl_jax::jam::DType d) {
  using DT = mccl_jax::jam::DType;
  switch (d) {
    case DT::kPred: return PJRT_Buffer_Type_PRED;
    case DT::kI8:   return PJRT_Buffer_Type_S8;
    case DT::kU8:   return PJRT_Buffer_Type_U8;
    case DT::kI16:  return PJRT_Buffer_Type_S16;
    case DT::kU16:  return PJRT_Buffer_Type_U16;
    case DT::kF16:  return PJRT_Buffer_Type_F16;
    case DT::kBF16: return PJRT_Buffer_Type_BF16;
    case DT::kI32:  case DT::kI64: return PJRT_Buffer_Type_S32;  // i64 narrowed on device
    case DT::kU32:  case DT::kU64: return PJRT_Buffer_Type_U32;
    default:        return PJRT_Buffer_Type_F32;  // f32 / f64
  }
}

// jam DType / ReduceKind -> mccl collective vocabulary (for all_reduce).
mccl_collective::DType ToMcclDType(mccl_jax::jam::DType d) {
  using DT = mccl_jax::jam::DType;
  switch (d) {
    case DT::kI8:   return mccl_collective::DType::kInt8;
    case DT::kU8:   return mccl_collective::DType::kUint8;
    case DT::kI32:  case DT::kI64: return mccl_collective::DType::kInt32;   // i64 narrowed on device
    case DT::kU32:  case DT::kU64: return mccl_collective::DType::kUint32;
    case DT::kF16:  return mccl_collective::DType::kFloat16;
    case DT::kBF16: return mccl_collective::DType::kBfloat16;
    default:        return mccl_collective::DType::kFloat32;  // f32 / f64 (f64 narrowed)
  }
}
mccl_collective::ReduceOp ToMcclReduce(mccl_jax::jam::ReduceKind r) {
  using RK = mccl_jax::jam::ReduceKind;
  switch (r) {
    case RK::kProd: return mccl_collective::ReduceOp::kProd;
    case RK::kMax:  return mccl_collective::ReduceOp::kMax;
    case RK::kMin:  return mccl_collective::ReduceOp::kMin;
    case RK::kAvg:  return mccl_collective::ReduceOp::kAvg;
    default:        return mccl_collective::ReduceOp::kSum;
  }
}

void CacheExecutableOutputs(PJRT_Executable* e) {
  for (const auto& s : e->program->outputs()) {
    e->out_types.push_back(JamPjrtType(s.dtype));
    e->out_dim_sizes.push_back(s.dims.size());
    for (int64_t d : s.dims) e->out_dims_flat.push_back(d);
    e->out_mem_kinds.push_back("device");
  }
  for (const auto& k : e->out_mem_kinds) {
    e->out_mem_kind_ptrs.push_back(k.c_str());
    e->out_mem_kind_sizes.push_back(k.size());
  }
}

PJRT_Error* ClientCompile(PJRT_Client_Compile_Args* a) {
  const PJRT_Program* p = a->program;
  // Optional: dump the StableHLO artifact the plugin actually receives (post SPMD partitioning).
  if (const char* dump = std::getenv("JAM_DUMP_ARTIFACT")) {
    if (FILE* f = std::fopen(dump, "wb")) { std::fwrite(p->code, 1, p->code_size, f); std::fclose(f); }
  }
  mccl_jax::jam::CompileResult r = mccl_jax::jam::Compile(p->code, p->code_size, a->client->num_processes);
  if (!r.program)
    return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "jam compile: " + r.error};
  auto le = std::make_unique<PJRT_LoadedExecutable>();
  le->client = a->client;
  le->program = std::shared_ptr<mccl_jax::jam::CompiledProgram>(std::move(r.program));
  // Executable runs on this host's addressable device(s) only; in a multi-host SPMD run each
  // process executes its own shard (the cross-host reduction is done by mccl in the collective).
  le->devices = a->client->addressable_devices;
  a->executable = le.release();
  return nullptr;
}

PJRT_Error* LoadedGetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* a) {
  auto* e = new PJRT_Executable();
  e->program = a->loaded_executable->program;
  CacheExecutableOutputs(e);
  a->executable = e;
  return nullptr;
}
PJRT_Error* ExecutableDestroy(PJRT_Executable_Destroy_Args* a) { delete a->executable; return nullptr; }
PJRT_Error* ExecutableName(PJRT_Executable_Name_Args* a) {
  a->executable_name = a->executable->name.c_str();
  a->executable_name_size = a->executable->name.size();
  return nullptr;
}
PJRT_Error* ExecutableNumReplicas(PJRT_Executable_NumReplicas_Args* a) { a->num_replicas = 1; return nullptr; }
PJRT_Error* ExecutableNumPartitions(PJRT_Executable_NumPartitions_Args* a) { a->num_partitions = 1; return nullptr; }
PJRT_Error* ExecutableNumOutputs(PJRT_Executable_NumOutputs_Args* a) {
  a->num_outputs = a->executable->program->outputs().size();
  return nullptr;
}
PJRT_Error* ExecutableOutputElementTypes(PJRT_Executable_OutputElementTypes_Args* a) {
  a->output_types = a->executable->out_types.data();
  a->num_output_types = a->executable->out_types.size();
  return nullptr;
}
PJRT_Error* ExecutableOutputDimensions(PJRT_Executable_OutputDimensions_Args* a) {
  auto* e = a->executable;
  a->num_outputs = e->out_dim_sizes.size();
  a->dims = e->out_dims_flat.data();
  a->dim_sizes = e->out_dim_sizes.data();
  return nullptr;
}
PJRT_Error* ExecutableOutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* a) {
  auto* e = a->executable;
  a->num_outputs = e->out_mem_kind_ptrs.size();
  a->memory_kinds = e->out_mem_kind_ptrs.data();
  a->memory_kind_sizes = e->out_mem_kind_sizes.data();
  return nullptr;
}
PJRT_Error* LoadedDestroy(PJRT_LoadedExecutable_Destroy_Args* a) { delete a->executable; return nullptr; }
PJRT_Error* LoadedDelete(PJRT_LoadedExecutable_Delete_Args* a) {
  a->executable->deleted = true;
  a->executable->program.reset();
  return nullptr;
}
PJRT_Error* LoadedIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* a) {
  a->is_deleted = a->executable->deleted;
  return nullptr;
}
PJRT_Error* LoadedAddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* a) {
  a->addressable_devices = a->executable->devices.data();
  a->num_addressable_devices = a->executable->devices.size();
  return nullptr;
}

// Run the program once per device; outputs written to output_lists[device][out] as fresh buffers.
PJRT_Error* LoadedExecute(PJRT_LoadedExecutable_Execute_Args* a) {
  PJRT_LoadedExecutable* le = a->executable;
  if (!le->program) return new PJRT_Error{PJRT_Error_Code_INVALID_ARGUMENT, "Execute: deleted executable"};

  // Route collective steps to mccl over the client's comm. Single-host (no comm) => identity.
  mccl_collective::Comm* comm = le->client ? le->client->comm.get() : nullptr;
  mccl_jax::jam::CollectiveFn collective =
      [comm](mccl_jax::jam::CollectiveOp op, mccl_jax::jam::ReduceKind r,
             mccl_jax::jam::DType dt, const void* send, void* recv,
             std::size_t send_count, std::size_t recv_count, int root,
             const std::vector<std::pair<int, int>>& pairs) -> std::string {
    using CO = mccl_jax::jam::CollectiveOp;
    mccl_collective::DType mdt = ToMcclDType(dt);
    if (comm == nullptr) {  // 1 rank: every collective is identity (copy for shape-changing ops)
      if (send != recv && recv_count != 0)
        std::memcpy(recv, send, recv_count * mccl_collective::DTypeSize(mdt));
      return "";
    }
    mccl_collective::ReduceOp mop = ToMcclReduce(r);
    mccl_collective::Status s;
    switch (op) {
      case CO::kAllReduce:
        s = mccl_collective::AllReduce(*comm, send, recv, send_count, mdt, mop); break;
      case CO::kAllGather:  // send_count = this rank's contribution; recv = send_count * n_ranks
        s = mccl_collective::AllGather(*comm, send, recv, send_count, mdt); break;
      case CO::kReduceScatter:  // recv_count = this rank's slice; send = recv_count * n_ranks
        s = mccl_collective::ReduceScatter(*comm, send, recv, recv_count, mdt, mop); break;
      case CO::kBroadcast:
        s = mccl_collective::Broadcast(*comm, send, recv, send_count, mdt, root); break;
      case CO::kAllToAll: {  // mccl wants the per-peer count, jam carries the total
        int n = comm->n_ranks();
        s = mccl_collective::AllToAll(*comm, send, recv, n > 0 ? send_count / n : send_count, mdt);
        break;
      }
      case CO::kCollectivePermute: {  // resolve this rank's target/source from the routing table
        int me = comm->rank(), target = -1, source = -1;
        for (const auto& p : pairs) {
          if (p.first == me) target = p.second;
          if (p.second == me) source = p.first;
        }
        if (source < 0 && recv_count != 0)  // unsent ranks read zeros
          std::memset(recv, 0, recv_count * mccl_collective::DTypeSize(mdt));
        if (target < 0 && source < 0) { s = mccl_collective::Status::Ok(); break; }
        s = mccl_collective::Permute(*comm, send, recv, send_count, mdt, target, source);
        break;
      }
    }
    return s.ok ? std::string() : s.message;
  };

  for (size_t d = 0; d < a->num_devices; ++d) {
    std::vector<mccl_jax::jam::RunInput> ins;
    ins.reserve(a->num_args);
    for (size_t i = 0; i < a->num_args; ++i) {
      PJRT_Buffer* b = a->argument_lists[d][i];
      ins.push_back({b->handle, b->nbytes});
    }
    std::vector<mccl_jax::jam::RunOutput> outs;
    std::string err = mccl_jax::jam::Run(*le->program, ins, outs, collective);
    if (!err.empty()) return new PJRT_Error{PJRT_Error_Code_INTERNAL, err};
    PJRT_Device* dev = a->execute_device;
    if (dev == nullptr && !le->devices.empty()) dev = le->devices[d % le->devices.size()];
    for (size_t i = 0; i < outs.size(); ++i) {
      auto* ob = new PJRT_Buffer();
      ob->client = le->client;
      ob->device = dev;
      ob->memory = dev ? dev->default_memory : nullptr;
      ob->type = JamPjrtType(outs[i].dtype);
      ob->dims = outs[i].dims;
      ob->nbytes = outs[i].nbytes;
      ob->data = outs[i].data;
      ob->handle = outs[i].handle;
      a->output_lists[d][i] = ob;
    }
    if (a->device_complete_events) a->device_complete_events[d] = MakeReadyEvent();
  }
  return nullptr;
}
#endif  // MCCL_JAX_WITH_JAM

// ---- not-yet-implemented surface --------------------------------------------
// Every PJRT_Api entry point not implemented above; to graduate one, write+bind a handler and
// delete it here. Without jam the compile/execute surface is stubbed too (loader-only plugin).
#ifdef MCCL_JAX_WITH_JAM
#define MCCL_JAX_UNIMPLEMENTED(X)                                               \
  X(Executable_SizeOfGeneratedCodeInBytes) X(Executable_GetCostAnalysis)       \
  X(Executable_OptimizedProgram) X(Executable_Serialize)                       \
  X(Executable_DeserializeAndLoad) X(LoadedExecutable_Fingerprint)             \
  X(Buffer_GetMemoryLayout)                                                    \
  X(CopyToDeviceStream_Destroy) X(CopyToDeviceStream_AddChunk)                 \
  X(CopyToDeviceStream_TotalBytes) X(CopyToDeviceStream_GranuleSize)           \
  X(CopyToDeviceStream_CurrentBytes)                                           \
  X(TopologyDescription_Serialize)                                             \
  X(Compile)                                                                   \
  X(Client_CreateViewOfDeviceBuffer)                                           \
  X(Executable_Fingerprint)                                                    \
  X(Executable_GetCompiledMemoryStats)
#else
#define MCCL_JAX_UNIMPLEMENTED(X)                                               \
  X(Client_Compile)                                                            \
  X(Executable_Destroy) X(Executable_Name) X(Executable_NumReplicas)           \
  X(Executable_NumPartitions) X(Executable_NumOutputs)                         \
  X(Executable_SizeOfGeneratedCodeInBytes) X(Executable_GetCostAnalysis)       \
  X(Executable_OutputMemoryKinds) X(Executable_OptimizedProgram)               \
  X(Executable_Serialize)                                                      \
  X(LoadedExecutable_Destroy) X(LoadedExecutable_GetExecutable)                \
  X(LoadedExecutable_AddressableDevices) X(LoadedExecutable_Delete)            \
  X(LoadedExecutable_IsDeleted) X(LoadedExecutable_Execute)                    \
  X(Executable_DeserializeAndLoad) X(LoadedExecutable_Fingerprint)             \
  X(Buffer_GetMemoryLayout)                                                    \
  X(CopyToDeviceStream_Destroy) X(CopyToDeviceStream_AddChunk)                 \
  X(CopyToDeviceStream_TotalBytes) X(CopyToDeviceStream_GranuleSize)           \
  X(CopyToDeviceStream_CurrentBytes)                                           \
  X(TopologyDescription_Serialize)                                             \
  X(Compile)                                                                   \
  X(Executable_OutputElementTypes) X(Executable_OutputDimensions)             \
  X(Client_CreateViewOfDeviceBuffer)                                           \
  X(Executable_Fingerprint)                                                    \
  X(Executable_GetCompiledMemoryStats)
#endif

#define MCCL_JAX_DEFINE_STUB(name)                                            \
  PJRT_Error* name##_Stub(PJRT_##name##_Args*) { return MakeUnimplemented(#name); }
MCCL_JAX_UNIMPLEMENTED(MCCL_JAX_DEFINE_STUB)
#undef MCCL_JAX_DEFINE_STUB

}  // namespace

extern "C" const PJRT_Api* GetPjrtApi() {
  static PJRT_Api api = [] {
    PJRT_Api a;
    std::memset(&a, 0, sizeof(a));
    a.struct_size = sizeof(PJRT_Api);
    a.pjrt_api_version.struct_size = sizeof(PJRT_Api_Version);
    a.pjrt_api_version.major_version = PJRT_API_MAJOR;
    a.pjrt_api_version.minor_version = PJRT_API_MINOR;

    a.PJRT_Error_Destroy = ErrorDestroy;
    a.PJRT_Error_Message = ErrorMessage;
    a.PJRT_Error_GetCode = ErrorGetCode;

    a.PJRT_Event_Destroy = EventDestroy;
    a.PJRT_Event_IsReady = EventIsReady;
    a.PJRT_Event_Error = EventError;
    a.PJRT_Event_Await = EventAwait;
    a.PJRT_Event_OnReady = EventOnReady;

    a.PJRT_Plugin_Initialize = PluginInitialize;
    a.PJRT_Plugin_Attributes = PluginAttributes;

    a.PJRT_Client_Create = ClientCreate;
    a.PJRT_Client_Destroy = ClientDestroy;
    a.PJRT_Client_PlatformName = ClientPlatformName;
    a.PJRT_Client_PlatformVersion = ClientPlatformVersion;
    a.PJRT_Client_ProcessIndex = ClientProcessIndex;
    a.PJRT_Client_Devices = ClientDevices;
    a.PJRT_Client_AddressableDevices = ClientAddressableDevices;
    a.PJRT_Client_AddressableMemories = ClientAddressableMemories;
    a.PJRT_Client_LookupDevice = ClientLookupDevice;
    a.PJRT_Client_LookupAddressableDevice = ClientLookupAddressableDevice;
    a.PJRT_Client_DefaultDeviceAssignment = ClientDefaultDeviceAssignment;
    a.PJRT_Client_TopologyDescription = ClientTopologyDescription;
    a.PJRT_Client_BufferFromHostBuffer = ClientBufferFromHostBuffer;

    a.PJRT_Device_GetDescription = DeviceGetDescription;
    a.PJRT_Device_IsAddressable = DeviceIsAddressable;
    a.PJRT_Device_LocalHardwareId = DeviceLocalHardwareId;
    a.PJRT_Device_AddressableMemories = DeviceAddressableMemories;
    a.PJRT_Device_DefaultMemory = DeviceDefaultMemory;
    a.PJRT_Device_MemoryStats = DeviceMemoryStats;

    a.PJRT_DeviceDescription_Id = DescId;
    a.PJRT_DeviceDescription_ProcessIndex = DescProcessIndex;
    a.PJRT_DeviceDescription_Attributes = DescAttributes;
    a.PJRT_DeviceDescription_Kind = DescKind;
    a.PJRT_DeviceDescription_DebugString = DescDebugString;
    a.PJRT_DeviceDescription_ToString = DescToString;

    a.PJRT_Memory_Id = MemoryId;
    a.PJRT_Memory_Kind = MemoryKind;
    a.PJRT_Memory_Kind_Id = MemoryKindId;
    a.PJRT_Memory_DebugString = MemoryDebugString;
    a.PJRT_Memory_ToString = MemoryToString;
    a.PJRT_Memory_AddressableByDevices = MemoryAddressableByDevices;

    a.PJRT_TopologyDescription_Create = TopologyCreate;
    a.PJRT_TopologyDescription_Destroy = TopologyDestroy;
    a.PJRT_TopologyDescription_PlatformName = TopologyPlatformName;
    a.PJRT_TopologyDescription_PlatformVersion = TopologyPlatformVersion;
    a.PJRT_TopologyDescription_GetDeviceDescriptions = TopologyGetDeviceDescriptions;
    a.PJRT_TopologyDescription_Attributes = TopologyAttributes;

    a.PJRT_ExecuteContext_Create = ExecuteContextCreate;
    a.PJRT_ExecuteContext_Destroy = ExecuteContextDestroy;

    a.PJRT_Buffer_Destroy = BufferDestroy;
    a.PJRT_Buffer_Delete = BufferDelete;
    a.PJRT_Buffer_IsDeleted = BufferIsDeleted;
    a.PJRT_Buffer_ElementType = BufferElementType;
    a.PJRT_Buffer_Dimensions = BufferDimensions;
    a.PJRT_Buffer_UnpaddedDimensions = BufferUnpaddedDimensions;
    a.PJRT_Buffer_DynamicDimensionIndices = BufferDynamicDimensionIndices;
    a.PJRT_Buffer_OnDeviceSizeInBytes = BufferOnDeviceSizeInBytes;
    a.PJRT_Buffer_Device = BufferDevice;
    a.PJRT_Buffer_Memory = BufferMemory;
    a.PJRT_Buffer_IsOnCpu = BufferIsOnCpu;
    a.PJRT_Buffer_ReadyEvent = BufferReadyEvent;
    a.PJRT_Buffer_ToHostBuffer = BufferToHostBuffer;
    a.PJRT_Buffer_CopyToMemory = BufferCopyToMemory;
    a.PJRT_Buffer_CopyToDevice = BufferCopyToDevice;
    a.PJRT_Buffer_UnsafePointer = BufferUnsafePointer;
    a.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = BufferOpaqueDeviceMemoryDataPointer;
    a.PJRT_Buffer_IncreaseExternalReferenceCount = BufferIncreaseExternalReferenceCount;
    a.PJRT_Buffer_DecreaseExternalReferenceCount = BufferDecreaseExternalReferenceCount;

#define MCCL_JAX_BIND_STUB(name) a.PJRT_##name = name##_Stub;
    MCCL_JAX_UNIMPLEMENTED(MCCL_JAX_BIND_STUB)
#undef MCCL_JAX_BIND_STUB

#ifdef MCCL_JAX_WITH_JAM
    a.PJRT_Client_Compile = ClientCompile;
    a.PJRT_Executable_Destroy = ExecutableDestroy;
    a.PJRT_Executable_Name = ExecutableName;
    a.PJRT_Executable_NumReplicas = ExecutableNumReplicas;
    a.PJRT_Executable_NumPartitions = ExecutableNumPartitions;
    a.PJRT_Executable_NumOutputs = ExecutableNumOutputs;
    a.PJRT_Executable_OutputElementTypes = ExecutableOutputElementTypes;
    a.PJRT_Executable_OutputDimensions = ExecutableOutputDimensions;
    a.PJRT_Executable_OutputMemoryKinds = ExecutableOutputMemoryKinds;
    a.PJRT_LoadedExecutable_Destroy = LoadedDestroy;
    a.PJRT_LoadedExecutable_GetExecutable = LoadedGetExecutable;
    a.PJRT_LoadedExecutable_AddressableDevices = LoadedAddressableDevices;
    a.PJRT_LoadedExecutable_Delete = LoadedDelete;
    a.PJRT_LoadedExecutable_IsDeleted = LoadedIsDeleted;
    a.PJRT_LoadedExecutable_Execute = LoadedExecute;
#endif

    return a;
  }();
  return &api;
}
