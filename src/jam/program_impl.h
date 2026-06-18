#ifndef MCCL_JAX_SRC_JAM_PROGRAM_IMPL_H_
#define MCCL_JAX_SRC_JAM_PROGRAM_IMPL_H_

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "src/jam/jam.h"

namespace mccl_jax::jam {

struct CompiledProgram::Impl {

  MPSGraph* graph = nil;
  NSArray<MPSGraphTensor*>* inputs = nil;
  NSArray<MPSGraphTensor*>* outputs = nil;
  std::vector<IoSpec> input_specs;
  std::vector<IoSpec> output_specs;

  struct Compute {
    MPSGraph* graph = nil;
    NSArray<MPSGraphTensor*>* inputs = nil;
    NSArray<MPSGraphTensor*>* outputs = nil;
    std::vector<int> input_slots;
    std::vector<int> output_slots;
    std::vector<IoSpec> input_specs;
    std::vector<IoSpec> output_specs;
  };
  struct Collective {
    CollectiveOp op = CollectiveOp::kAllReduce;
    ReduceKind reduce = ReduceKind::kSum;
    DType dtype = DType::kF32;
    int send_slot = -1;
    int recv_slot = -1;
    std::int64_t send_count = 0;
    std::int64_t recv_count = 0;
    int root = 0;
    std::vector<std::pair<int, int>> pairs;
  };
  struct Step {
    enum Kind { kCompute, kCollective } kind = kCompute;
    Compute compute;
    Collective collective;
  };
  std::vector<Step> steps;
  int num_slots = 0;
  std::vector<int> input_slots;
  std::vector<int> output_slots;
};

}

#endif
