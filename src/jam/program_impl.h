#ifndef MCCL_JAX_SRC_JAM_PROGRAM_IMPL_H_
#define MCCL_JAX_SRC_JAM_PROGRAM_IMPL_H_

// Internal (ObjC++) — the body of CompiledProgram.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "src/jam/jam.h"

namespace mccl_jax::jam {

struct CompiledProgram::Impl {
  // Single-segment path (no collectives): one MPSGraph over @main's args -> returns.
  MPSGraph* graph = nil;
  NSArray<MPSGraphTensor*>* inputs = nil;   // ordered by @main argument order
  NSArray<MPSGraphTensor*>* outputs = nil;  // ordered by return order
  std::vector<IoSpec> input_specs;
  std::vector<IoSpec> output_specs;

  // Segmented path (collectives present): an ordered step list over a device-buffer-slot table.
  // A compute step runs an MPSGraph reading/writing slots; a collective step runs in place on a slot.
  struct Compute {
    MPSGraph* graph = nil;
    NSArray<MPSGraphTensor*>* inputs = nil;
    NSArray<MPSGraphTensor*>* outputs = nil;
    std::vector<int> input_slots;     // slot feeding each input placeholder
    std::vector<int> output_slots;    // slot each output writes
    std::vector<IoSpec> input_specs;
    std::vector<IoSpec> output_specs;
  };
  struct Collective {
    CollectiveOp op = CollectiveOp::kAllReduce;
    ReduceKind reduce = ReduceKind::kSum;
    DType dtype = DType::kF32;
    int send_slot = -1;               // input buffer slot
    int recv_slot = -1;               // output slot (== send_slot for in-place all_reduce/broadcast)
    std::int64_t send_count = 0;      // elements sent
    std::int64_t recv_count = 0;      // elements received (allocated when recv_slot != send_slot)
    int root = 0;                     // broadcast source rank
    std::vector<std::pair<int, int>> pairs;  // collective_permute source->target routing
  };
  struct Step {
    enum Kind { kCompute, kCollective } kind = kCompute;
    Compute compute;
    Collective collective;
  };
  std::vector<Step> steps;            // empty => use the single-segment graph above
  int num_slots = 0;
  std::vector<int> input_slots;       // buffer slot for each @main arg
  std::vector<int> output_slots;      // buffer slot for each @main return
};

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_PROGRAM_IMPL_H_
