#ifndef MCCL_JAX_SRC_JAM_LOWERING_INTERNAL_H_
#define MCCL_JAX_SRC_JAM_LOWERING_INTERNAL_H_

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "src/jam/jam.h"

namespace mlir { class Block; }

namespace mccl_jax::jam {

struct Segment {
  MPSGraph* graph = nil;
  std::vector<MPSGraphTensor*> in_tensors;
  std::vector<int> in_slots;
  std::vector<IoSpec> in_specs;
  std::vector<MPSGraphTensor*> out_tensors;
  std::vector<int> out_slots;
  std::vector<IoSpec> out_specs;
};

struct PendingCollective {
  CollectiveOp op = CollectiveOp::kAllReduce;
  int send_slot = -1;
  int recv_slot = -1;
  ReduceKind reduce = ReduceKind::kSum;
  DType dtype = DType::kF32;
  std::int64_t send_count = 0;
  std::int64_t recv_count = 0;
  int root = 0;
  std::vector<std::pair<int, int>> pairs;
};

class Lowering {
 public:
  explicit Lowering(MPSGraph* graph) {
    segments_.push_back({});
    segments_[0].graph = graph;
  }

  MPSGraph* graph() const { return segments_[cur_].graph; }

  MPSGraphTensor* value(mlir::Value v);
  void bind(mlir::Value v, MPSGraphTensor* t);

  void substitute(mlir::Value param, mlir::Value with) { subst_[param] = with; }
  mlir::Value resolve(mlir::Value v) const {
    auto it = subst_.find(v);
    return it == subst_.end() ? v : resolve(it->second);
  }

  void bindTuple(mlir::Value tuple, mlir::ValueRange elems) {
    tuples_[tuple].assign(elems.begin(), elems.end());
  }
  mlir::Value tupleElement(mlir::Value tuple, int64_t index) const {
    auto it = tuples_.find(tuple);
    if (it == tuples_.end() || index < 0 || index >= (int64_t)it->second.size())
      return mlir::Value();
    return it->second[index];
  }

  void fail(std::string message);
  bool ok() const { return error_.empty(); }
  const std::string& error() const { return error_; }

  void declareArg(mlir::Value arg, MPSGraphTensor* placeholder, int slot);

  void retypeArg(mlir::Value arg, const IoSpec& shard, MPSGraphTensor* shardPlaceholder);

  const IoSpec* argSpecOverride(mlir::Value arg) const;

  void setOutputShard(mlir::Value v, const IoSpec& shard) { output_shard_[v] = shard; }
  const IoSpec* outputShard(mlir::Value v) const {
    auto it = output_shard_.find(v);
    return it == output_shard_.end() ? nullptr : &it->second;
  }

  int materialize(mlir::Value v);

  MPSGraphTensor* startSegmentAfterCollective(int slot, const IoSpec& spec);
  int allocSlot() { return next_slot_++; }
  void assignSlot(mlir::Value v, int slot) { slot_of_[v] = slot; }
  int numSlots() const { return next_slot_; }
  int curSegment() const { return cur_; }
  std::vector<Segment>& segments() { return segments_; }

  std::vector<PendingCollective>& pending() { return collectives_; }

  int nRanks() const { return n_ranks_; }
  void setNRanks(int n) { n_ranks_ = n; }

  static MPSDataType MpsDType(mlir::Type element_type);
  static NSArray<NSNumber*>* MpsShape(mlir::Type tensor_type);
  static mlir::Type ElementType(mlir::Type tensor_type);
  static IoSpec SpecOf(mlir::Value v);

 private:
  struct Tensor { void* t = nullptr; int seg = 0; };

  MPSGraphTensor* slotPlaceholder(int seg, int slot, const IoSpec& spec);

  std::vector<Segment> segments_;
  std::vector<PendingCollective> collectives_;
  int cur_ = 0;
  int next_slot_ = 0;
  int n_ranks_ = 1;
  llvm::DenseMap<mlir::Value, Tensor> map_;
  llvm::DenseMap<mlir::Value, mlir::Value> subst_;
  llvm::DenseMap<mlir::Value, std::vector<mlir::Value>> tuples_;
  llvm::DenseMap<mlir::Value, int> slot_of_;
  std::map<std::pair<int,int>, void*> seg_slot_ph_;
  llvm::DenseMap<mlir::Value, IoSpec> arg_override_;
  llvm::DenseMap<mlir::Value, IoSpec> output_shard_;
  std::string error_;
};

using OpHandler = void (*)(Lowering& L, mlir::Operation* op);

void RegisterOp(const char* op_name, OpHandler handler);
OpHandler LookupOp(llvm::StringRef op_name);
void RegisterAllOps();
std::vector<std::string> RegisteredOpNames();

void WalkBlock(Lowering& L, mlir::ModuleOp module, mlir::Block& block,
               std::vector<mlir::Value>& returns);

void RegisterElementwise();
void RegisterUnary();
void RegisterBinary();
void RegisterReduce();
void RegisterShape();
void RegisterLinalg();
void RegisterGatherScatter();
void RegisterStructural();
void RegisterControlFlow();
void RegisterRng();
void RegisterNorm();
void RegisterDynamic();
void RegisterCollectives();
void RegisterHardLimit();

}

#endif
