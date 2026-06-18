#ifndef MCCL_JAX_SRC_JAM_LOWERING_INTERNAL_H_
#define MCCL_JAX_SRC_JAM_LOWERING_INTERNAL_H_

// Internal (ObjC++) — per-compile lowering state and the op-handler registry.

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

// One MPSGraph + the slot wiring it reads/writes. Segments are split at collectives:
// segment K's graph runs, then a collective runs in place on a slot, then segment K+1 runs.
struct Segment {
  MPSGraph* graph = nil;
  std::vector<MPSGraphTensor*> in_tensors;   // placeholders this segment feeds from slots
  std::vector<int> in_slots;                 // slot for each in placeholder
  std::vector<IoSpec> in_specs;
  std::vector<MPSGraphTensor*> out_tensors;  // tensors this segment writes (lazily accumulated)
  std::vector<int> out_slots;                // slot each out tensor writes
  std::vector<IoSpec> out_specs;
};

// A pending collective that runs between two segments: reads `send_slot`, writes `recv_slot`
// (recv_slot == send_slot for in-place ops: all_reduce, broadcast). For shape-changing ops
// (all_gather, reduce_scatter, all_to_all) recv_slot is a fresh slot of recv_count elements.
struct PendingCollective {
  CollectiveOp op = CollectiveOp::kAllReduce;
  int send_slot = -1;
  int recv_slot = -1;
  ReduceKind reduce = ReduceKind::kSum;
  DType dtype = DType::kF32;
  std::int64_t send_count = 0;
  std::int64_t recv_count = 0;
  int root = 0;
  std::vector<std::pair<int, int>> pairs;  // collective_permute source->target routing
};

// Per-compile state threaded through every op handler.
class Lowering {
 public:
  explicit Lowering(MPSGraph* graph) {
    segments_.push_back({});
    segments_[0].graph = graph;
  }

  // Current segment's graph; handlers emit into it.
  MPSGraph* graph() const { return segments_[cur_].graph; }

  MPSGraphTensor* value(mlir::Value v);          // operand lookup (fails if unbound)
  void bind(mlir::Value v, MPSGraphTensor* t);   // record an op result (in the current segment)

  // Map an inlined callee block arg / passthrough result to its source; resolve() chases the chain.
  void substitute(mlir::Value param, mlir::Value with) { subst_[param] = with; }
  mlir::Value resolve(mlir::Value v) const {
    auto it = subst_.find(v);
    return it == subst_.end() ? v : resolve(it->second);
  }

  // Tuple tracking: record element values; tupleElement() resolves get_tuple_element.
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

  // ---- segmentation (used by lower.mm; collectives drive it) ----
  // Record a @main arg: give it a slot and create its placeholder in segment 0.
  void declareArg(mlir::Value arg, MPSGraphTensor* placeholder, int slot);
  // Re-type an arg to its per-rank shard shape (manual SPMD: JAX feeds the shard, not the full
  // tensor). Replaces the arg's segment-0 placeholder + records the shard spec for input_specs.
  void retypeArg(mlir::Value arg, const IoSpec& shard, MPSGraphTensor* shardPlaceholder);
  // Shard spec recorded by retypeArg for `arg`, or nullptr.
  const IoSpec* argSpecOverride(mlir::Value arg) const;
  // Record that a @main return value is a per-rank shard (SPMDShardToFullShape with shard != full):
  // the plugin returns the shard buffer, JAX reassembles the full array from each rank's shard.
  void setOutputShard(mlir::Value v, const IoSpec& shard) { output_shard_[v] = shard; }
  const IoSpec* outputShard(mlir::Value v) const {
    auto it = output_shard_.find(v);
    return it == output_shard_.end() ? nullptr : &it->second;
  }
  // Ensure `v` lives in a slot (assign one, mark its producing segment to write it). Returns slot.
  int materialize(mlir::Value v);
  // Close the current segment and open a fresh one whose graph holds a placeholder for `slot`.
  // Returns the new segment's placeholder for that slot (so the caller can bind a result to it).
  MPSGraphTensor* startSegmentAfterCollective(int slot, const IoSpec& spec);
  int allocSlot() { return next_slot_++; }
  void assignSlot(mlir::Value v, int slot) { slot_of_[v] = slot; }
  int numSlots() const { return next_slot_; }
  int curSegment() const { return cur_; }
  std::vector<Segment>& segments() { return segments_; }
  // Collectives recorded in walk order; pending()[i] runs after segments_[i] (one per split).
  std::vector<PendingCollective>& pending() { return collectives_; }

  // Cluster size (1 ⇒ single device: collectives are identity and can be elided without a segment).
  int nRanks() const { return n_ranks_; }
  void setNRanks(int n) { n_ranks_ = n; }

  // MLIR tensor/element type → MPSGraph.
  static MPSDataType MpsDType(mlir::Type element_type);
  static NSArray<NSNumber*>* MpsShape(mlir::Type tensor_type);
  static mlir::Type ElementType(mlir::Type tensor_type);
  static IoSpec SpecOf(mlir::Value v);

 private:
  struct Tensor { void* t = nullptr; int seg = 0; };  // (__bridge) MPSGraphTensor* + producing segment
  // The placeholder + slot for a value already routed into the current segment, keyed (seg,slot).
  MPSGraphTensor* slotPlaceholder(int seg, int slot, const IoSpec& spec);

  std::vector<Segment> segments_;
  std::vector<PendingCollective> collectives_;  // collectives_[i] follows segments_[i]
  int cur_ = 0;
  int next_slot_ = 0;
  int n_ranks_ = 1;
  llvm::DenseMap<mlir::Value, Tensor> map_;         // value → tensor + producing segment
  llvm::DenseMap<mlir::Value, mlir::Value> subst_;  // inlined block arg / passthrough → source value
  llvm::DenseMap<mlir::Value, std::vector<mlir::Value>> tuples_;  // tuple result → elements
  llvm::DenseMap<mlir::Value, int> slot_of_;        // value → its slot (once routed through one)
  std::map<std::pair<int,int>, void*> seg_slot_ph_; // (segment,slot) → (__bridge) placeholder tensor
  llvm::DenseMap<mlir::Value, IoSpec> arg_override_;  // arg → per-rank shard spec (manual SPMD)
  llvm::DenseMap<mlir::Value, IoSpec> output_shard_;  // return value → per-rank shard spec (manual SPMD)
  std::string error_;
};

// One handler per StableHLO op. Reads operands via L.value(), binds the result via L.bind().
using OpHandler = void (*)(Lowering& L, mlir::Operation* op);

void RegisterOp(const char* op_name, OpHandler handler);
OpHandler LookupOp(llvm::StringRef op_name);
void RegisterAllOps();  // calls each category registrar exactly once
std::vector<std::string> RegisteredOpNames();  // all registered handler names (for coverage)

// Walk one block, emitting MPSGraph nodes; `returns` collects its yielded values.
void WalkBlock(Lowering& L, mlir::ModuleOp module, mlir::Block& block,
               std::vector<mlir::Value>& returns);

// Category registrars (one per ops/*.mm file).
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

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_LOWERING_INTERNAL_H_
