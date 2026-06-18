// jam_run.mm — runs a compiled jam program (its MPSGraph) on the Metal GPU.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/jam/jam_run.h"
#include "src/jam/program_impl.h"
#include "src/execution/metal_buffer.h"

namespace mccl_jax::jam {
namespace {

MPSDataType MpsType(DType d) {
  switch (d) {
    case DType::kF16:  return MPSDataTypeFloat16;
    case DType::kBF16: return MPSDataTypeBFloat16;
    case DType::kI8:   return MPSDataTypeInt8;
    case DType::kU8:   return MPSDataTypeUInt8;
    case DType::kPred: return MPSDataTypeBool;
    case DType::kI16:  return MPSDataTypeInt16;
    case DType::kU16:  return MPSDataTypeUInt16;
    case DType::kI32:
    case DType::kI64:  return MPSDataTypeInt32;   // i64 narrowed on device
    case DType::kU32:
    case DType::kU64:  return MPSDataTypeUInt32;
    default:           return MPSDataTypeFloat32;  // f32 / f64
  }
}

// Bytes per element on device (matches MpsType's narrowing of i64/u64/f64).
std::size_t Width(DType d) {
  switch (d) {
    case DType::kI8: case DType::kU8: case DType::kPred: return 1;
    case DType::kF16: case DType::kBF16: case DType::kI16: case DType::kU16: return 2;
    default: return 4;
  }
}

std::size_t NumElems(const std::vector<int64_t>& dims) {
  std::size_t n = 1;
  for (int64_t d : dims) n *= static_cast<std::size_t>(d < 0 ? 0 : d);
  return n;
}

NSArray<NSNumber*>* Shape(const std::vector<int64_t>& dims) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d : dims) [a addObject:@(d)];
  if (a.count == 0) [a addObject:@1];  // rank-0 modeled as [1] (matches the lowering)
  return a;
}

}  // namespace

namespace {

// A device-buffer slot threaded between steps. `owned` slots are freed by the runner; program
// inputs/outputs handed to the caller are not owned here.
struct Slot {
  void* data = nullptr;
  void* handle = nullptr;
  std::size_t nbytes = 0;
  bool owned = false;
};

std::size_t SpecBytes(const IoSpec& s) { return NumElems(s.dims) * Width(s.dtype); }

// One command queue reused across runs (Execute is serialized; creating a queue per call costs
// ~200us). Process-lifetime, like the default device.
id<MTLCommandQueue> SharedQueue() {
  static id<MTLCommandQueue> q = [MTLCreateSystemDefaultDevice() newCommandQueue];
  return q;
}

// A process-lifetime free-list of device buffers keyed by exact byte size. Training runs the same
// program repeatedly with identical shapes, so the run-internal intermediate/gradient buffers recur
// every step; recycling them removes ~2-5us/buffer of newBufferWithLength churn per step (tens of
// buffers per segmented step). Buffers handed to the caller as program outputs leave the pool (the
// caller frees them via metal::Release) — only run-internal intermediates are recycled here.
class BufferPool {
 public:
  BufferPool() : enabled_(getenv("MCCL_JAX_NO_POOL") == nullptr) {}

  mccl_jax::metal::Allocation Acquire(std::size_t nbytes) {
    if (nbytes == 0) return {};
    if (enabled_) {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = free_.find(nbytes);
      if (it != free_.end() && !it->second.empty()) {
        mccl_jax::metal::Allocation a = it->second.back();
        it->second.pop_back();
        return a;
      }
    }
    return mccl_jax::metal::Allocate(nbytes);
  }
  void Recycle(void* handle, void* data, std::size_t nbytes) {
    if (handle == nullptr) return;
    if (enabled_ && nbytes != 0) {
      std::lock_guard<std::mutex> lk(mu_);
      auto& v = free_[nbytes];
      if (v.size() < kCapPerSize) { v.push_back({data, handle}); return; }
    }
    mccl_jax::metal::Release(handle);  // disabled, size 0, or this size's free-list is full → free to OS
  }

 private:
  static constexpr std::size_t kCapPerSize = 64;  // bounds memory if many distinct shapes appear
  const bool enabled_;
  std::mutex mu_;
  std::unordered_map<std::size_t, std::vector<mccl_jax::metal::Allocation>> free_;
};

BufferPool& Pool() {
  static BufferPool p;
  return p;
}

// Optional per-step timing (MCCL_JAX_TIME=1). The MPSGraph run and the collective call are both
// synchronous, so wall time around each is its GPU/collective cost. Prints a cumulative breakdown
// every 200 runs to stderr. Off by default (one getenv at first use).
struct RunTimer {
  const bool on = getenv("MCCL_JAX_TIME") != nullptr;
  long runs = 0;
  double compute_ms = 0, collective_ms = 0, copy_ms = 0;
  double n_compute = 0, n_collective = 0;
  bool wall_set = false;
  std::chrono::high_resolution_clock::time_point wall_start;  // first Run, for GPU-utilization
};
RunTimer& Timer() { static RunTimer t; return t; }
using TClock = std::chrono::high_resolution_clock;
double MsSince(TClock::time_point t0) {
  return std::chrono::duration<double, std::milli>(TClock::now() - t0).count();
}


}  // namespace

// ---- segmented runner: compute steps run MPSGraphs over slots; collective steps run in place ----
static std::string RunSegmentedImpl(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                                    std::vector<RunOutput>& outputs, const CollectiveFn& collective) {
  auto* impl = prog.impl();
  const std::vector<IoSpec>& in_specs = prog.inputs();
  const std::vector<IoSpec>& out_specs = prog.outputs();
  if (inputs.size() != in_specs.size()) return "jam run: wrong number of inputs";

  id<MTLCommandQueue> queue = SharedQueue();
  if (queue == nil) return "jam run: no Metal device";

  std::vector<Slot> slots(impl->num_slots);
  auto freeOwned = [&]() {
    for (auto& s : slots) if (s.owned && s.handle) Pool().Recycle(s.handle, s.data, s.nbytes);
  };

  // One-shot structural dump (MCCL_JAX_DUMP=1): the ordered step list with each step's slot
  // reads/writes — reveals why deferred collectives flush (e.g. an update segment consuming a
  // reduced gradient before the next reduction).
  static bool dumped = false;
  if (!dumped && getenv("MCCL_JAX_DUMP") != nullptr) {
    dumped = true;
    fprintf(stderr, "[jam-dump] %zu steps, %d slots:\n", impl->steps.size(), impl->num_slots);
    int si = 0;
    for (const auto& st : impl->steps) {
      if (st.kind == CompiledProgram::Impl::Step::kCollective) {
        const auto& c = st.collective;
        fprintf(stderr, "  %2d COLL op=%d reduce=%d send_slot=%d recv_slot=%d count=%lld\n",
                si, (int)c.op, (int)c.reduce, c.send_slot, c.recv_slot, (long long)c.send_count);
      } else {
        const auto& c = st.compute;
        std::string in, out;
        for (int s : c.input_slots) in += std::to_string(s) + " ";
        for (int s : c.output_slots) out += std::to_string(s) + " ";
        fprintf(stderr, "  %2d COMPUTE in=[%s] out=[%s]\n", si, in.c_str(), out.c_str());
      }
      ++si;
    }
  }

  // Only inputs whose slot a collective writes in place need a private copy (so the caller's buffer
  // isn't mutated); every other input aliases the caller's buffer directly, avoiding a per-step copy.
  std::vector<char> collective_target(impl->num_slots, 0);
  for (const auto& step : impl->steps)
    if (step.kind == CompiledProgram::Impl::Step::kCollective)
      collective_target[step.collective.recv_slot] = 1;  // slot a collective writes in place

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::size_t want = SpecBytes(in_specs[i]);
    if (inputs[i].nbytes != want) { freeOwned(); return "jam run: input size mismatch"; }
    int slot = impl->input_slots[i];
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)inputs[i].handle;
    if (collective_target[slot]) {
      mccl_jax::metal::Allocation a = Pool().Acquire(want);
      if (want != 0 && a.data == nullptr) { freeOwned(); return "jam run: input slot alloc failed"; }
      if (want != 0) std::memcpy(a.data, src.contents, want);
      slots[slot] = {a.data, a.handle, want, true};
    } else {
      slots[slot] = {want ? src.contents : nullptr, inputs[i].handle, want, false};
    }
  }

  // Size-aware deferred collective fusion (DEFAULT ON; disable with MCCL_JAX_NO_FUSE=1): small in-place
  // collectives of identical config (biases, scalars — the per-call-overhead-bound ones) are batched
  // into one transfer (memcpy gather → single collective → scatter), flushed before any step that
  // reads/overwrites a pending slot. ONLY small buffers are fused (see kFuseMaxBytes): all buffers the
  // cluster reduces are < mccl's METAL_MIN_BYTES (256MB) so mccl reduces them on the CPU, whose result
  // is fusion-invariant → BYTE-EXACT (verified: checksum unchanged). Big weight all_reduces stay
  // separate (their gather/scatter memcpy outweighs the saved call, and HoistCollectives already
  // clusters them). Net: fewer mccl calls (e.g. 7→4 for dp_train), byte-exact.
  using Coll = CompiledProgram::Impl::Collective;
  const bool fuse_enabled = getenv("MCCL_JAX_NO_FUSE") == nullptr;
  std::vector<const Coll*> pending;
  auto pendingHas = [&](int slot) {
    for (auto* c : pending) if (c->send_slot == slot) return true;
    return false;
  };
  auto sameCfg = [](const Coll& a, const Coll& b) {
    return a.op == b.op && a.reduce == b.reduce && a.dtype == b.dtype && a.root == b.root &&
           a.pairs == b.pairs;
  };

  auto runOneInPlace = [&](const Coll& c) -> std::string {
    void* p = slots[c.send_slot].data;
    TClock::time_point _ct = Timer().on ? TClock::now() : TClock::time_point{};
    std::string e = collective(c.op, c.reduce, c.dtype, p, p, (std::size_t)c.send_count,
                               (std::size_t)c.send_count, c.root, c.pairs);
    if (Timer().on) { Timer().collective_ms += MsSince(_ct); Timer().n_collective += 1; }
    return e;
  };
  auto flush = [&]() -> std::string {
    if (pending.empty()) return "";
    if (pending.size() == 1) { std::string e = runOneInPlace(*pending[0]); pending.clear(); return e; }
    std::size_t width = Width(pending[0]->dtype), total = 0;
    for (auto* c : pending) total += (std::size_t)c->send_count;
    mccl_jax::metal::Allocation scratch = Pool().Acquire(total * width);
    if (total != 0 && scratch.data == nullptr) { pending.clear(); return "jam run: fuse scratch alloc failed"; }
    std::size_t off = 0;
    for (auto* c : pending) {  // gather each piece's local data into the contiguous scratch
      std::size_t n = (std::size_t)c->send_count * width;
      if (n) std::memcpy((char*)scratch.data + off, slots[c->send_slot].data, n);
      off += n;
    }
    TClock::time_point _ct = Timer().on ? TClock::now() : TClock::time_point{};
    std::string e = collective(pending[0]->op, pending[0]->reduce, pending[0]->dtype, scratch.data,
                               scratch.data, total, total, pending[0]->root, pending[0]->pairs);
    if (Timer().on) { Timer().collective_ms += MsSince(_ct); Timer().n_collective += 1; }
    if (e.empty()) {
      off = 0;
      for (auto* c : pending) {  // scatter the reduced result back to each piece's slot
        std::size_t n = (std::size_t)c->send_count * width;
        if (n) std::memcpy(slots[c->send_slot].data, (char*)scratch.data + off, n);
        off += n;
      }
    }
    Pool().Recycle(scratch.handle, scratch.data, total * width);
    pending.clear();
    return e;
  };

  for (const auto& step : impl->steps) {
    if (step.kind == CompiledProgram::Impl::Step::kCollective) {
      const auto& c = step.collective;
      bool inplace = (c.recv_slot == c.send_slot);
      // Size-aware: defer only SMALL in-place collectives (biases, scalars). Their per-call mccl
      // overhead dominates, and the gather/scatter memcpy to fuse them is cheap. Big buffers (weights)
      // run separately — fusing them was measured slower (their memcpy outweighs the saved call).
      constexpr std::size_t kFuseMaxBytes = 256 * 1024;
      bool small = (std::size_t)c.send_count * Width(c.dtype) <= kFuseMaxBytes;
      if (fuse_enabled && inplace && small && !pendingHas(c.send_slot) &&
          (pending.empty() || sameCfg(c, *pending[0]))) {
        pending.push_back(&c);  // defer; flushed before its first consumer or at end of run
        continue;
      }
      // Run this one separately. Flush the deferred small batch only if THIS op touches a pending slot
      // (independent big collectives don't — so the small batch keeps accumulating across them).
      if (pendingHas(c.send_slot) || (c.recv_slot != c.send_slot && pendingHas(c.recv_slot)))
        if (std::string fe = flush(); !fe.empty()) { freeOwned(); return fe; }
      void* send = slots[c.send_slot].data;
      void* recv = send;
      if (c.recv_slot != c.send_slot) {  // shape-changing op: give it a fresh recv buffer
        std::size_t nbytes = (std::size_t)c.recv_count * Width(c.dtype);
        Slot& rs = slots[c.recv_slot];
        if (rs.owned && rs.handle) Pool().Recycle(rs.handle, rs.data, rs.nbytes);
        mccl_jax::metal::Allocation a = Pool().Acquire(nbytes);
        if (nbytes != 0 && a.data == nullptr) { freeOwned(); return "jam run: collective recv alloc failed"; }
        rs = {a.data, a.handle, nbytes, true};
        recv = rs.data;
      }
      TClock::time_point _ct = Timer().on ? TClock::now() : TClock::time_point{};
      std::string e = collective(c.op, c.reduce, c.dtype, send, recv,
                                 (std::size_t)c.send_count, (std::size_t)c.recv_count, c.root, c.pairs);
      if (Timer().on) { Timer().collective_ms += MsSince(_ct); Timer().n_collective += 1; }
      if (!e.empty()) { freeOwned(); return e; }
      continue;
    }
    @autoreleasepool {
      const auto& c = step.compute;
      if (c.outputs.count == 0) continue;  // empty (no-op) segment
      if (!pending.empty()) {  // flush deferred collectives this segment depends on, before running it
        bool dep = false;
        for (int s : c.input_slots) if (pendingHas(s)) { dep = true; break; }
        if (!dep) for (int s : c.output_slots) if (pendingHas(s)) { dep = true; break; }
        if (dep) { if (std::string fe = flush(); !fe.empty()) { freeOwned(); return fe; } }
      }
      NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
      for (std::size_t i = 0; i < c.input_slots.size(); ++i) {
        Slot& s = slots[c.input_slots[i]];
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)s.handle;
        if (buf == nil) { freeOwned(); return "jam run: compute reads an empty slot"; }
        MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                        shape:Shape(c.input_specs[i].dims)
                                                                     dataType:MpsType(c.input_specs[i].dtype)];
        feeds[(MPSGraphTensor*)c.inputs[i]] = td;
      }
      // Pre-allocate the output buffers and have MPSGraph write directly into them (resultsDictionary
      // form), avoiding a per-output readBytes copy — the dominant cost for many-output segments.
      std::vector<mccl_jax::metal::Allocation> outAllocs(c.output_slots.size());
      NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* resultsDict =
          [NSMutableDictionary dictionary];
      for (std::size_t i = 0; i < c.output_slots.size(); ++i) {
        std::size_t nbytes = SpecBytes(c.output_specs[i]);
        mccl_jax::metal::Allocation a = Pool().Acquire(nbytes);
        if (nbytes != 0 && a.data == nullptr) { freeOwned(); return "jam run: output slot alloc failed"; }
        outAllocs[i] = a;
        if (nbytes != 0)
          resultsDict[(MPSGraphTensor*)c.outputs[i]] = [[MPSGraphTensorData alloc]
              initWithMTLBuffer:(__bridge id<MTLBuffer>)a.handle
                          shape:Shape(c.output_specs[i].dims)
                       dataType:MpsType(c.output_specs[i].dtype)];
      }
      TClock::time_point _gt = Timer().on ? TClock::now() : TClock::time_point{};
      [c.graph runWithMTLCommandQueue:queue feeds:feeds targetOperations:nil resultsDictionary:resultsDict];
      if (Timer().on) { Timer().compute_ms += MsSince(_gt); Timer().n_compute += 1; }
      for (std::size_t i = 0; i < c.output_slots.size(); ++i) {
        int slot = c.output_slots[i];
        if (slots[slot].owned && slots[slot].handle)
          Pool().Recycle(slots[slot].handle, slots[slot].data, slots[slot].nbytes);
        slots[slot] = {outAllocs[i].data, outAllocs[i].handle, SpecBytes(c.output_specs[i]), true};
      }
    }
  }

  if (std::string fe = flush(); !fe.empty()) { freeOwned(); return fe; }  // any collectives whose result feeds an output

  // Hand out program outputs from their slots, transferring ownership (clear `owned` so freeOwned skips).
  std::vector<RunOutput> out(out_specs.size());
  for (std::size_t i = 0; i < out_specs.size(); ++i) {
    int slot = impl->output_slots[i];
    Slot& s = slots[slot];
    std::size_t nbytes = SpecBytes(out_specs[i]);
    if (s.owned) {
      out[i] = {s.data, s.handle, nbytes, out_specs[i].dims, out_specs[i].dtype};
      s.owned = false;  // ownership moves to the output
    } else {
      // Slot aliases a non-owned buffer (shouldn't happen with copy-in inputs); copy it out.
      mccl_jax::metal::Allocation a = Pool().Acquire(nbytes);
      if (nbytes != 0 && a.data == nullptr) { freeOwned(); return "jam run: output copy alloc failed"; }
      if (nbytes != 0) std::memcpy(a.data, s.data, nbytes);
      out[i] = {a.data, a.handle, nbytes, out_specs[i].dims, out_specs[i].dtype};
    }
  }
  freeOwned();
  outputs = std::move(out);
  if (Timer().on) {
    RunTimer& t = Timer();
    if (++t.runs % 200 == 0)
      fprintf(stderr, "[jam-time] runs=%ld compute=%.3fms/run (%.1f segs) collective=%.3fms/run (%.1f ops)\n",
              t.runs, t.compute_ms / t.runs, t.n_compute / t.runs,
              t.collective_ms / t.runs, t.n_collective / t.runs);
  }
  return "";
}

std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs, const CollectiveFn& collective) {
  if (!prog.impl()->steps.empty()) {
    // Top-level pool drains any per-Run autoreleased Metal/MPS objects (the per-step pools only
    // cover compute steps); bounds memory growth over long training runs.
    std::string err;
    @autoreleasepool { err = RunSegmentedImpl(prog, inputs, outputs, collective); }
    return err;
  }
  return Run(prog, inputs, outputs);
}

std::string Run(const CompiledProgram& prog, const std::vector<RunInput>& inputs,
                std::vector<RunOutput>& outputs) {
  @autoreleasepool {
    auto* impl = prog.impl();
    const std::vector<IoSpec>& in_specs = prog.inputs();
    const std::vector<IoSpec>& out_specs = prog.outputs();
    if (inputs.size() != in_specs.size()) return "jam run: wrong number of inputs";

    id<MTLCommandQueue> queue = SharedQueue();
    if (queue == nil) return "jam run: no Metal device";

    TClock::time_point _t0 = Timer().on ? TClock::now() : TClock::time_point{};
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      std::size_t want = NumElems(in_specs[i].dims) * Width(in_specs[i].dtype);
      if (inputs[i].nbytes != want) return "jam run: input size mismatch";
      id<MTLBuffer> buf = (__bridge id<MTLBuffer>)inputs[i].handle;
      if (buf == nil) return "jam run: null input buffer";
      MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                      shape:Shape(in_specs[i].dims)
                                                                   dataType:MpsType(in_specs[i].dtype)];
      feeds[(MPSGraphTensor*)impl->inputs[i]] = td;
    }

    // Pre-allocate outputs and let MPSGraph write directly into them (no readBytes copy).
    std::vector<RunOutput> out(out_specs.size());
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* resultsDict =
        [NSMutableDictionary dictionary];
    for (std::size_t i = 0; i < out_specs.size(); ++i) {
      std::size_t nbytes = NumElems(out_specs[i].dims) * Width(out_specs[i].dtype);
      mccl_jax::metal::Allocation alloc = mccl_jax::metal::Allocate(nbytes);
      if (nbytes != 0 && alloc.data == nullptr) {
        for (auto& o : out) if (o.handle) mccl_jax::metal::Release(o.handle);
        return "jam run: output allocation failed";
      }
      out[i] = {alloc.data, alloc.handle, nbytes, out_specs[i].dims, out_specs[i].dtype};
      if (nbytes != 0)
        resultsDict[(MPSGraphTensor*)impl->outputs[i]] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:(__bridge id<MTLBuffer>)alloc.handle
                        shape:Shape(out_specs[i].dims)
                     dataType:MpsType(out_specs[i].dtype)];
    }
    TClock::time_point _t1 = Timer().on ? TClock::now() : TClock::time_point{};
    [impl->graph runWithMTLCommandQueue:queue feeds:feeds targetOperations:nil resultsDictionary:resultsDict];
    if (Timer().on) {
      RunTimer& t = Timer();
      if (!t.wall_set) { t.wall_start = _t0; t.wall_set = true; }
      t.copy_ms += std::chrono::duration<double, std::milli>(_t1 - _t0).count();  // feeds + output alloc
      t.compute_ms += MsSince(_t1);                                               // GPU run
      if (++t.runs % 50 == 0) {
        double wall = std::chrono::duration<double, std::milli>(TClock::now() - t.wall_start).count();
        fprintf(stderr, "[jam-time single] runs=%ld setup=%.3f gpu=%.3fms/run | in-jam %.0f%% of wall (gap=JAX/Python)\n",
                t.runs, t.copy_ms / t.runs, t.compute_ms / t.runs,
                100.0 * (t.compute_ms + t.copy_ms) / wall);
      }
    }
    outputs = std::move(out);
  }
  return "";
}

}  // namespace mccl_jax::jam
