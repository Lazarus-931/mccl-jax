#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/jam/lower.h"
#include "src/jam/lowering_internal.h"
#include "src/jam/program_impl.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {

static MPSDataType MpsTypeOf(DType d) {
  switch (d) {
    case DType::kF16:  return MPSDataTypeFloat16;
    case DType::kBF16: return MPSDataTypeBFloat16;
    case DType::kPred: return MPSDataTypeBool;
    case DType::kI8:   return MPSDataTypeInt8;
    case DType::kU8:   return MPSDataTypeUInt8;
    case DType::kI16:  return MPSDataTypeInt16;
    case DType::kU16:  return MPSDataTypeUInt16;
    case DType::kI32:  case DType::kI64: return MPSDataTypeInt32;
    case DType::kU32:  case DType::kU64: return MPSDataTypeUInt32;
    default:           return MPSDataTypeFloat32;
  }
}
static NSArray<NSNumber*>* MpsShapeOf(const IoSpec& s) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d : s.dims) [a addObject:@(d)];
  if (a.count == 0) [a addObject:@1];
  return a;
}

static std::map<std::string, OpHandler>& Registry() {
  static auto* registry = new std::map<std::string, OpHandler>();
  return *registry;
}
void RegisterOp(const char* name, OpHandler handler) { Registry()[name] = handler; }
OpHandler LookupOp(llvm::StringRef name) {
  auto it = Registry().find(name.str());
  return it == Registry().end() ? nullptr : it->second;
}
void RegisterAllOps() {
  static bool done = false;
  if (done) return;
  done = true;
  RegisterElementwise();
  RegisterUnary();
  RegisterBinary();
  RegisterReduce();
  RegisterShape();
  RegisterLinalg();
  RegisterGatherScatter();
  RegisterStructural();
  RegisterControlFlow();
  RegisterRng();
  RegisterNorm();
  RegisterDynamic();
  RegisterCollectives();
  RegisterHardLimit();
}

std::vector<std::string> RegisteredOpNames() {
  RegisterAllOps();
  std::vector<std::string> names;
  for (auto& kv : Registry()) names.push_back(kv.first);
  return names;
}

MPSGraphTensor* Lowering::value(mlir::Value v) {
  auto it = map_.find(v);
  if (it == map_.end()) {
    fail("jam: referenced an unbound SSA value");
    return nil;
  }
  if (it->second.seg == cur_) return (__bridge MPSGraphTensor*)it->second.t;
  int slot = materialize(v);
  return slotPlaceholder(cur_, slot, SpecOf(v));
}
void Lowering::bind(mlir::Value v, MPSGraphTensor* t) {
  map_[v] = Tensor{(__bridge void*)t, cur_};
}
void Lowering::fail(std::string message) {
  if (error_.empty()) error_ = std::move(message);
}

void Lowering::declareArg(mlir::Value arg, MPSGraphTensor* placeholder, int slot) {
  map_[arg] = Tensor{(__bridge void*)placeholder, 0};
  slot_of_[arg] = slot;
  seg_slot_ph_[{0, slot}] = (__bridge void*)placeholder;
  Segment& s0 = segments_[0];
  s0.in_tensors.push_back(placeholder);
  s0.in_slots.push_back(slot);
  s0.in_specs.push_back(SpecOf(arg));
}

void Lowering::retypeArg(mlir::Value arg, const IoSpec& shard, MPSGraphTensor* shardPlaceholder) {
  auto si = slot_of_.find(arg);
  if (si == slot_of_.end()) return;
  map_[arg] = Tensor{(__bridge void*)shardPlaceholder, 0};
  seg_slot_ph_[{0, si->second}] = (__bridge void*)shardPlaceholder;
  arg_override_[arg] = shard;
  Segment& s0 = segments_[0];
  for (size_t i = 0; i < s0.in_slots.size(); ++i) {
    if (s0.in_slots[i] == si->second) { s0.in_tensors[i] = shardPlaceholder; s0.in_specs[i] = shard; }
  }
}

const IoSpec* Lowering::argSpecOverride(mlir::Value arg) const {
  auto it = arg_override_.find(arg);
  return it == arg_override_.end() ? nullptr : &it->second;
}

MPSGraphTensor* Lowering::slotPlaceholder(int seg, int slot, const IoSpec& spec) {
  auto key = std::make_pair(seg, slot);
  auto it = seg_slot_ph_.find(key);
  if (it != seg_slot_ph_.end()) return (__bridge MPSGraphTensor*)it->second;
  Segment& s = segments_[seg];
  MPSGraphTensor* ph = [s.graph placeholderWithShape:MpsShapeOf(spec)
                                            dataType:MpsTypeOf(spec.dtype)
                                                name:nil];
  s.in_tensors.push_back(ph);
  s.in_slots.push_back(slot);
  s.in_specs.push_back(spec);
  seg_slot_ph_[key] = (__bridge void*)ph;
  return ph;
}

int Lowering::materialize(mlir::Value v) {
  mlir::Value r = resolve(v);
  for (mlir::Value c : {v, r}) {
    auto si = slot_of_.find(c);
    if (si != slot_of_.end()) { slot_of_[v] = si->second; return si->second; }
  }
  auto mi = map_.find(r);
  if (mi == map_.end()) { fail("jam: materialize of an unbound value"); return -1; }
  int seg = mi->second.seg;
  int slot = allocSlot();
  slot_of_[v] = slot;
  slot_of_[r] = slot;
  Segment& s = segments_[seg];
  s.out_tensors.push_back((__bridge MPSGraphTensor*)mi->second.t);
  s.out_slots.push_back(slot);
  s.out_specs.push_back(SpecOf(r));
  return slot;
}

MPSGraphTensor* Lowering::startSegmentAfterCollective(int slot, const IoSpec& spec) {
  segments_.push_back({});
  cur_ = (int)segments_.size() - 1;
  segments_[cur_].graph = [MPSGraph new];
  return slotPlaceholder(cur_, slot, spec);
}

mlir::Type Lowering::ElementType(mlir::Type t) {
  if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t)) return rt.getElementType();
  return t;
}
MPSDataType Lowering::MpsDType(mlir::Type elem) {
  if (elem.isF32()) return MPSDataTypeFloat32;
  if (elem.isF16()) return MPSDataTypeFloat16;
  if (elem.isBF16()) return MPSDataTypeBFloat16;
  if (elem.isInteger(1)) return MPSDataTypeBool;
  if (elem.isInteger(8)) return MPSDataTypeInt8;
  if (elem.isInteger(16)) return MPSDataTypeInt16;
  if (elem.isInteger(32)) return MPSDataTypeInt32;
  if (elem.isInteger(64)) return MPSDataTypeInt32;
  return MPSDataTypeFloat32;
}
NSArray<NSNumber*>* Lowering::MpsShape(mlir::Type t) {
  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t))
    for (int64_t d : rt.getShape()) [shape addObject:@(d)];
  if (shape.count == 0) [shape addObject:@1];
  return shape;
}
IoSpec Lowering::SpecOf(mlir::Value v) {
  IoSpec spec;
  if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType()))
    for (int64_t d : rt.getShape()) spec.dims.push_back(d);
  mlir::Type e = ElementType(v.getType());
  if (e.isF32()) spec.dtype = DType::kF32;
  else if (e.isF16()) spec.dtype = DType::kF16;
  else if (e.isBF16()) spec.dtype = DType::kBF16;
  else if (e.isInteger(1)) spec.dtype = DType::kPred;
  else if (e.isInteger(8)) spec.dtype = DType::kI8;
  else if (e.isInteger(16)) spec.dtype = DType::kI16;
  else if (e.isInteger(32)) spec.dtype = DType::kI32;
  else if (e.isInteger(64)) spec.dtype = DType::kI64;
  else spec.dtype = DType::kF32;
  return spec;
}

void LowerAllReduce(Lowering& L, mlir::Operation* op);
void LowerAllGather(Lowering& L, mlir::Operation* op);
void LowerReduceScatter(Lowering& L, mlir::Operation* op);
void LowerAllToAll(Lowering& L, mlir::Operation* op);
void LowerCollectiveBroadcast(Lowering& L, mlir::Operation* op);
void LowerCollectivePermute(Lowering& L, mlir::Operation* op);

void WalkBlock(Lowering& L, mlir::ModuleOp module, mlir::Block& block,
               std::vector<mlir::Value>& returns) {
  for (mlir::Operation& op : block) {
    if (!L.ok()) return;
    llvm::StringRef name = op.getName().getStringRef();

    if (name == "func.return" || name == "stablehlo.return") {
      for (mlir::Value v : op.getOperands()) returns.push_back(v);
      continue;
    }

    if (name == "stablehlo.optimization_barrier") {
      for (auto [res, in] : llvm::zip(op.getResults(), op.getOperands())) {
        L.bind(res, L.value(in));
        L.substitute(res, in);
      }
      continue;
    }
    if (name == "stablehlo.after_all" || name == "stablehlo.create_token") {

      L.bind(op.getResult(0), [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeFloat32]);
      continue;
    }
    if (name == "stablehlo.tuple") {

      L.bindTuple(op.getResult(0), op.getOperands());
      continue;
    }
    if (name == "stablehlo.get_tuple_element") {
      auto gte = op.getOperand(0);
      int64_t index = op.getAttrOfType<mlir::IntegerAttr>("index").getInt();
      mlir::Value elem = L.tupleElement(gte, index);
      if (!elem) { L.fail("jam: get_tuple_element from non-tuple value"); return; }
      L.bind(op.getResult(0), L.value(elem));
      L.substitute(op.getResult(0), elem);
      continue;
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(&op)) {
      auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (!callee || callee.getBody().empty()) {
        L.fail("jam: func.call to unknown/external callee '" + call.getCallee().str() + "'");
        return;
      }
      mlir::Block& body = callee.getBody().front();
      for (auto [param, arg] : llvm::zip(body.getArguments(), call.getArgOperands())) {
        L.bind(param, L.value(arg));
        L.substitute(param, arg);
      }
      std::vector<mlir::Value> calleeReturns;
      WalkBlock(L, module, body, calleeReturns);
      if (!L.ok()) return;
      if (calleeReturns.size() != call.getNumResults()) {
        L.fail("jam: func.call result count mismatch");
        return;
      }
      for (auto [result, ret] : llvm::zip(call.getResults(), calleeReturns)) {
        L.bind(result, L.value(ret));
        L.substitute(result, ret);
      }
      continue;
    }

    if (name == "stablehlo.custom_call") {
      auto cc = mlir::cast<mlir::stablehlo::CustomCallOp>(&op);
      llvm::StringRef target = cc.getCallTargetName();
      if (target == "SPMDFullToShardShape") {
        mlir::Value in = op.getOperand(0), out = op.getResult(0);
        mlir::Value src = L.resolve(in);
        IoSpec shard = Lowering::SpecOf(out);
        if (mlir::isa<mlir::BlockArgument>(src) &&
            Lowering::SpecOf(in).dims != shard.dims) {
          MPSGraphTensor* ph = [L.graph() placeholderWithShape:MpsShapeOf(shard)
                                                      dataType:MpsTypeOf(shard.dtype)
                                                          name:nil];
          L.retypeArg(src, shard, ph);
          L.bind(out, ph);
        } else {
          L.bind(out, L.value(in));
        }
        L.substitute(out, in);
        continue;
      }
      if (target == "SPMDShardToFullShape") {

        mlir::Value in = op.getOperand(0), out = op.getResult(0);
        L.bind(out, L.value(in));
        L.substitute(out, in);
        IoSpec shard = Lowering::SpecOf(in), full = Lowering::SpecOf(out);
        if (shard.dims != full.dims) L.setOutputShard(out, shard);
        continue;
      }
    }

    if (name == "stablehlo.all_reduce")           { LowerAllReduce(L, &op);    continue; }
    if (name == "stablehlo.all_gather")           { LowerAllGather(L, &op);    continue; }
    if (name == "stablehlo.reduce_scatter")       { LowerReduceScatter(L, &op); continue; }
    if (name == "stablehlo.all_to_all")           { LowerAllToAll(L, &op);     continue; }
    if (name == "stablehlo.collective_broadcast") { LowerCollectiveBroadcast(L, &op); continue; }
    if (name == "stablehlo.collective_permute")   { LowerCollectivePermute(L, &op);   continue; }
    OpHandler handler = LookupOp(name);
    if (!handler) {
      L.fail("jam: unhandled op '" + name.str() + "'");
      return;
    }
    handler(L, &op);
  }
}

static bool ReduceKindOf(mlir::Operation* op, ReduceKind* out) {
  if (op->getNumRegions() < 1) return false;
  mlir::Region& region = op->getRegion(0);
  if (region.empty()) return false;
  for (mlir::Operation& inner : region.front()) {
    llvm::StringRef n = inner.getName().getStringRef();
    if (n == "stablehlo.add")      { *out = ReduceKind::kSum;  return true; }
    if (n == "stablehlo.maximum")  { *out = ReduceKind::kMax;  return true; }
    if (n == "stablehlo.minimum")  { *out = ReduceKind::kMin;  return true; }
    if (n == "stablehlo.multiply") { *out = ReduceKind::kProd; return true; }
  }
  return false;
}

static std::int64_t ElemCount(const IoSpec& s) {
  std::int64_t n = 1;
  for (int64_t d : s.dims) n *= d;
  return n;
}

static void EmitCollective(Lowering& L, mlir::Operation* op, CollectiveOp cop, ReduceKind reduce,
                           bool in_place, int root,
                           std::vector<std::pair<int, int>> pairs = {}) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    L.fail("jam: variadic collectives not yet supported");
    return;
  }
  mlir::Value operand = op->getOperand(0), result = op->getResult(0);
  IoSpec in = Lowering::SpecOf(operand), out = Lowering::SpecOf(result);

  if (L.nRanks() <= 1 && in.dims == out.dims && cop != CollectiveOp::kCollectivePermute) {
    MPSGraphTensor* v = L.value(operand);
    if (!L.ok()) return;
    L.bind(result, v);
    L.substitute(result, operand);
    return;
  }

  int send_slot = L.materialize(operand);
  if (!L.ok()) return;

  bool can_inplace = in_place && operand.hasOneUse();
  int recv_slot = can_inplace ? send_slot : L.allocSlot();

  PendingCollective pc;
  pc.op = cop;
  pc.send_slot = send_slot;
  pc.recv_slot = recv_slot;
  pc.reduce = reduce;
  pc.dtype = out.dtype;
  pc.send_count = ElemCount(in);
  pc.recv_count = ElemCount(out);
  pc.root = root;
  pc.pairs = std::move(pairs);
  L.pending().push_back(std::move(pc));

  MPSGraphTensor* res = L.startSegmentAfterCollective(recv_slot, out);
  L.bind(result, res);
  L.assignSlot(result, recv_slot);
}

static bool LeadingDim(mlir::Operation* op, const char* attr) {
  auto a = op->getAttrOfType<mlir::IntegerAttr>(attr);
  return !a || a.getInt() == 0;
}

void LowerAllReduce(Lowering& L, mlir::Operation* op) {
  ReduceKind reduce;
  if (!ReduceKindOf(op, &reduce)) {
    L.fail("jam: all_reduce reducer is not a recognized add/max/min/multiply");
    return;
  }
  EmitCollective(L, op, CollectiveOp::kAllReduce, reduce, true, 0);
}

void LowerAllGather(Lowering& L, mlir::Operation* op) {
  if (!LeadingDim(op, "all_gather_dim")) {
    L.fail("jam: all_gather only supported along dim 0");
    return;
  }
  EmitCollective(L, op, CollectiveOp::kAllGather, ReduceKind::kSum, false, 0);
}

void LowerReduceScatter(Lowering& L, mlir::Operation* op) {
  if (!LeadingDim(op, "scatter_dimension")) {
    L.fail("jam: reduce_scatter only supported along dim 0");
    return;
  }
  ReduceKind reduce;
  if (!ReduceKindOf(op, &reduce)) {
    L.fail("jam: reduce_scatter reducer is not a recognized add/max/min/multiply");
    return;
  }
  EmitCollective(L, op, CollectiveOp::kReduceScatter, reduce, false, 0);
}

void LowerAllToAll(Lowering& L, mlir::Operation* op) {
  if (!LeadingDim(op, "split_dimension") || !LeadingDim(op, "concat_dimension")) {
    L.fail("jam: all_to_all only supported along dim 0");
    return;
  }
  EmitCollective(L, op, CollectiveOp::kAllToAll, ReduceKind::kSum, false, 0);
}

void LowerCollectiveBroadcast(Lowering& L, mlir::Operation* op) {
  int root = 0;
  if (auto g = op->getAttrOfType<mlir::DenseIntElementsAttr>("replica_groups"))
    if (g.getNumElements() > 0) root = (int)(*g.value_begin<llvm::APInt>()).getSExtValue();
  EmitCollective(L, op, CollectiveOp::kBroadcast, ReduceKind::kSum, true, root);
}

void LowerCollectivePermute(Lowering& L, mlir::Operation* op) {
  std::vector<std::pair<int, int>> pairs;
  if (auto a = op->getAttrOfType<mlir::DenseIntElementsAttr>("source_target_pairs")) {
    std::vector<int> flat;
    for (const llvm::APInt& v : a.getValues<llvm::APInt>()) flat.push_back((int)v.getSExtValue());
    for (size_t i = 0; i + 1 < flat.size(); i += 2) pairs.push_back({flat[i], flat[i + 1]});
  }
  EmitCollective(L, op, CollectiveOp::kCollectivePermute, ReduceKind::kSum,
                 false, 0, std::move(pairs));
}

static void HoistCollectives(CompiledProgram::Impl* impl) {
  using Step = CompiledProgram::Impl::Step;
  auto touches = [](const Step& s, int slot) -> bool {
    if (s.kind == Step::kCollective)
      return s.collective.send_slot == slot || s.collective.recv_slot == slot;
    for (int v : s.compute.input_slots) if (v == slot) return true;
    for (int v : s.compute.output_slots) if (v == slot) return true;
    return false;
  };
  auto& steps = impl->steps;
  for (std::size_t i = 0; i < steps.size(); ++i) {
    if (steps[i].kind != Step::kCollective) continue;
    if (steps[i].collective.recv_slot != steps[i].collective.send_slot) continue;
    int slot = steps[i].collective.send_slot;
    std::size_t j = i;
    while (j > 0 && !touches(steps[j - 1], slot)) {
      std::swap(steps[j - 1], steps[j]);
      --j;
    }
  }
}

std::unique_ptr<CompiledProgram> Lower(mlir::ModuleOp module, std::string& error, int num_processes) {
  RegisterAllOps();

  auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    error = "jam: module has no @main function";
    return nullptr;
  }

  MPSGraph* graph = [MPSGraph new];
  Lowering lowering(graph);
  lowering.setNRanks(num_processes);
  auto impl = std::make_unique<CompiledProgram::Impl>();
  impl->graph = graph;

  NSMutableArray<MPSGraphTensor*>* inputs = [NSMutableArray array];
  std::vector<mlir::BlockArgument> args;
  for (mlir::BlockArgument arg : main.getArguments()) {
    MPSGraphTensor* placeholder =
        [graph placeholderWithShape:Lowering::MpsShape(arg.getType())
                           dataType:Lowering::MpsDType(Lowering::ElementType(arg.getType()))
                               name:nil];
    int slot = lowering.allocSlot();
    lowering.declareArg(arg, placeholder, slot);
    impl->input_slots.push_back(slot);
    [inputs addObject:placeholder];
    args.push_back(arg);
  }

  std::vector<mlir::Value> returns;
  WalkBlock(lowering, module, main.getBody().front(), returns);
  if (!lowering.ok()) {
    error = lowering.error();
    return nullptr;
  }

  std::vector<IoSpec> in_specs;
  for (mlir::BlockArgument arg : args) {
    const IoSpec* ov = lowering.argSpecOverride(arg);
    in_specs.push_back(ov ? *ov : Lowering::SpecOf(arg));
  }

  if (lowering.pending().empty()) {
    NSMutableArray<MPSGraphTensor*>* outputs = [NSMutableArray array];
    std::vector<IoSpec> out_specs;
    for (mlir::Value v : returns) {
      MPSGraphTensor* t = lowering.value(v);
      if (!lowering.ok()) { error = lowering.error(); return nullptr; }
      [outputs addObject:t];
      const IoSpec* shard = lowering.outputShard(v);
      out_specs.push_back(shard ? *shard : Lowering::SpecOf(v));
    }
    impl->inputs = inputs;
    impl->outputs = outputs;
    impl->input_specs = std::move(in_specs);
    impl->output_specs = std::move(out_specs);
    return std::make_unique<CompiledProgram>(std::move(impl));
  }

  for (mlir::Value v : returns) {
    int slot = lowering.materialize(v);
    if (!lowering.ok()) { error = lowering.error(); return nullptr; }
    impl->output_slots.push_back(slot);
    const IoSpec* shard = lowering.outputShard(v);
    impl->output_specs.push_back(shard ? *shard : Lowering::SpecOf(v));
  }
  impl->input_specs = std::move(in_specs);
  impl->num_slots = lowering.numSlots();

  auto& segs = lowering.segments();
  auto& cols = lowering.pending();
  for (size_t k = 0; k < segs.size(); ++k) {
    CompiledProgram::Impl::Step step;
    step.kind = CompiledProgram::Impl::Step::kCompute;
    auto& c = step.compute;
    c.graph = segs[k].graph;
    NSMutableArray<MPSGraphTensor*>* ins = [NSMutableArray array];
    for (MPSGraphTensor* t : segs[k].in_tensors) [ins addObject:t];
    NSMutableArray<MPSGraphTensor*>* outs = [NSMutableArray array];
    for (MPSGraphTensor* t : segs[k].out_tensors) [outs addObject:t];
    c.inputs = ins;
    c.outputs = outs;
    c.input_slots = segs[k].in_slots;
    c.output_slots = segs[k].out_slots;
    c.input_specs = segs[k].in_specs;
    c.output_specs = segs[k].out_specs;
    impl->steps.push_back(std::move(step));

    if (k < cols.size()) {
      CompiledProgram::Impl::Step cstep;
      cstep.kind = CompiledProgram::Impl::Step::kCollective;
      cstep.collective.op = cols[k].op;
      cstep.collective.reduce = cols[k].reduce;
      cstep.collective.dtype = cols[k].dtype;
      cstep.collective.send_slot = cols[k].send_slot;
      cstep.collective.recv_slot = cols[k].recv_slot;
      cstep.collective.send_count = cols[k].send_count;
      cstep.collective.recv_count = cols[k].recv_count;
      cstep.collective.root = cols[k].root;
      cstep.collective.pairs = cols[k].pairs;
      impl->steps.push_back(std::move(cstep));
    }
  }

  if (getenv("MCCL_JAX_NO_HOIST") == nullptr) HoistCollectives(impl.get());
  return std::make_unique<CompiledProgram>(std::move(impl));
}

}
