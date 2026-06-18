// ops/control_flow.mm — while / if / case via MPSGraph control-flow nodes.
// Region blocks are re-walked with WalkBlock; case folds to a chain of nested ifs.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {

void WalkBlock(Lowering& L, mlir::ModuleOp module, mlir::Block& block,
               std::vector<mlir::Value>& returns);

namespace {

mlir::ModuleOp ModuleOf(mlir::Operation* op) {
  return op->getParentOfType<mlir::ModuleOp>();
}

constexpr int64_t kMaxUnroll = 2048;  // cap on unrolled trip count (else fail cleanly, never OOM/crash)

// A scalar integer stablehlo.constant value (after seeing through convert), else false.
bool ConstInt(mlir::Value v, int64_t* out) {
  mlir::Operation* def = v.getDefiningOp();
  while (def && mlir::isa<mlir::stablehlo::ConvertOp>(def)) def = def->getOperand(0).getDefiningOp();
  auto c = def ? mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def) : nullptr;
  if (!c) return false;
  auto attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(c.getValue());
  if (!attr || attr.getNumElements() != 1) return false;
  *out = (*attr.begin()).getSExtValue();
  return true;
}

// MPSGraph's `while` SIGSEGVs when the body contains dynamic_slice / dynamic_update_slice (the scan
// stacking pattern). Detect that so we can unroll instead.
bool BodyHasDynamicIndexing(mlir::Region& body) {
  bool found = false;
  body.walk([&](mlir::Operation* o) {
    llvm::StringRef n = o->getName().getStringRef();
    if (n == "stablehlo.dynamic_update_slice" || n == "stablehlo.dynamic_slice") found = true;
  });
  return found;
}

// Static trip count of a canonical counted loop (counter: const init, cond `counter </<= constN`,
// body returns `counter + constStep`), else -1. This is exactly what jax.lax.scan emits.
int64_t StaticTripCount(mlir::stablehlo::WhileOp wh) {
  mlir::Block& cond = wh.getCond().front();
  mlir::Block& body = wh.getBody().front();
  mlir::Operation* cret = cond.getTerminator();
  if (!cret || cret->getNumOperands() != 1) return -1;
  auto cmp = cret->getOperand(0).getDefiningOp<mlir::stablehlo::CompareOp>();
  if (!cmp) return -1;
  auto lhs = mlir::dyn_cast<mlir::BlockArgument>(cmp.getLhs());
  if (!lhs || lhs.getOwner() != &cond) return -1;
  unsigned idx = lhs.getArgNumber();
  int64_t bound, init, step = 0;
  if (!ConstInt(cmp.getRhs(), &bound)) return -1;
  if (idx >= wh->getNumOperands() || !ConstInt(wh->getOperand(idx), &init)) return -1;
  mlir::Operation* bret = body.getTerminator();
  if (!bret || idx >= bret->getNumOperands()) return -1;
  auto add = bret->getOperand(idx).getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add) return -1;
  auto a0 = mlir::dyn_cast<mlir::BlockArgument>(add.getLhs());
  auto a1 = mlir::dyn_cast<mlir::BlockArgument>(add.getRhs());
  if (a0 && a0.getArgNumber() == idx && ConstInt(add.getRhs(), &step)) {}
  else if (a1 && a1.getArgNumber() == idx && ConstInt(add.getLhs(), &step)) {}
  else return -1;
  if (step <= 0) return -1;
  using D = mlir::stablehlo::ComparisonDirection;
  D dir = cmp.getComparisonDirection();
  int64_t n;
  if (dir == D::LT) n = (bound - init + step - 1) / step;
  else if (dir == D::LE) n = (bound - init) / step + 1;
  else return -1;
  return n < 0 ? 0 : n;
}

// Bind op's results to the MPSGraph tensors in an NSArray.
void BindResults(Lowering& L, mlir::Operation* op, NSArray<MPSGraphTensor*>* results) {
  unsigned n = op->getNumResults();
  for (unsigned i = 0; i < n && i < results.count; ++i) L.bind(op->getResult(i), results[i]);
}

// Walk a single-block region, returning the MPSGraph tensors for its yielded (return) values.
NSArray<MPSGraphTensor*>* WalkRegion(Lowering& L, mlir::Operation* op, mlir::Region& region) {
  std::vector<mlir::Value> rets;
  WalkBlock(L, ModuleOf(op), region.front(), rets);
  NSMutableArray<MPSGraphTensor*>* out = [NSMutableArray array];
  for (mlir::Value v : rets) {
    MPSGraphTensor* t = L.value(v);
    if (t) [out addObject:t];
  }
  return out;
}

// if(pred) then region0 else region1.  pred is a scalar bool tensor.
void If(Lowering& L, mlir::Operation* op) {
  auto ifOp = mlir::cast<mlir::stablehlo::IfOp>(op);
  MPSGraphTensor* pred = L.value(op->getOperand(0));
  mlir::Region& thenR = ifOp.getTrueBranch();
  mlir::Region& elseR = ifOp.getFalseBranch();

  NSArray<MPSGraphTensor*>* results = [L.graph()
      ifWithPredicateTensor:pred
                  thenBlock:^NSArray<MPSGraphTensor*>* { return WalkRegion(L, op, thenR); }
                  elseBlock:^NSArray<MPSGraphTensor*>* { return WalkRegion(L, op, elseR); }
                       name:nil];
  if (!L.ok()) return;
  BindResults(L, op, results);
}

// case(index): clamped to [0, n-1] (XLA), built as a right-folded chain of `if (index == k)`.
void Case(Lowering& L, mlir::Operation* op) {
  auto caseOp = mlir::cast<mlir::stablehlo::CaseOp>(op);
  MPSGraphTensor* index = L.value(op->getOperand(0));
  index = Casted(L, index, MPSDataTypeInt32);
  unsigned n = caseOp.getBranches().size();
  if (n == 0) { L.fail("jam: case with no branches"); return; }

  // Walk the last branch as the base of the fold (covers index >= n-1, the clamp).
  __block std::vector<mlir::Region*> regions;
  for (mlir::Region& r : caseOp.getBranches()) regions.push_back(&r);

  // Build nested ifs from the last branch backward.
  NSArray<MPSGraphTensor*>* acc = WalkRegion(L, op, *regions[n - 1]);
  if (!L.ok()) return;
  for (int k = (int)n - 2; k >= 0; --k) {
    MPSGraphTensor* kConst = [L.graph() constantWithScalar:(double)k dataType:MPSDataTypeInt32];
    MPSGraphTensor* isK = [L.graph() equalWithPrimaryTensor:index secondaryTensor:kConst name:nil];
    mlir::Region* branch = regions[k];
    NSArray<MPSGraphTensor*>* prev = acc;
    acc = [L.graph()
        ifWithPredicateTensor:isK
                    thenBlock:^NSArray<MPSGraphTensor*>* { return WalkRegion(L, op, *branch); }
                    elseBlock:^NSArray<MPSGraphTensor*>* { return prev; }
                         name:nil];
    if (!L.ok()) return;
  }
  BindResults(L, op, acc);
}

// while(carry): cond region yields a scalar bool; body region yields the next carry.
void While(Lowering& L, mlir::Operation* op) {
  auto wh = mlir::cast<mlir::stablehlo::WhileOp>(op);
  mlir::Region& condR = wh.getCond();
  mlir::Region& bodyR = wh.getBody();
  unsigned nCarry = op->getNumOperands();
  mlir::Block& condBlock = condR.front();
  mlir::Block& bodyBlock = bodyR.front();

  // MPSGraph's native `while` SIGSEGVs on bodies with dynamic_slice/dynamic_update_slice (the
  // jax.lax.scan stacking pattern). For a statically counted loop, unroll instead: the dynamic ops
  // then live in the main graph (where they run fine) and there is no MPSGraph while. fori-style
  // loops (no dynamic indexing) keep the native while below.
  if (BodyHasDynamicIndexing(bodyR)) {
    int64_t trip = StaticTripCount(wh);
    if (trip < 0 || trip > kMaxUnroll) {
      L.fail("jam: while with dynamic slice/update needs a static trip count <= 2048 to unroll "
             "(MPSGraph cannot run dynamic indexing inside a loop)");
      return;
    }
    std::vector<MPSGraphTensor*> carry;
    for (mlir::Value v : op->getOperands()) carry.push_back(L.value(v));
    if (!L.ok()) return;
    for (int64_t k = 0; k < trip; ++k) {
      for (unsigned i = 0; i < nCarry; ++i) L.bind(bodyBlock.getArgument(i), carry[i]);
      std::vector<mlir::Value> rets;
      WalkBlock(L, ModuleOf(op), bodyBlock, rets);
      if (!L.ok()) return;
      std::vector<MPSGraphTensor*> next;
      for (mlir::Value v : rets) {
        MPSGraphTensor* t = L.value(v);
        if (!t) { L.fail("jam: unrolled while body produced an unbound value"); return; }
        next.push_back(t);
      }
      carry.swap(next);
    }
    for (unsigned i = 0; i < nCarry && i < op->getNumResults(); ++i) L.bind(op->getResult(i), carry[i]);
    return;
  }

  // Initial carry = the while's operands.
  NSMutableArray<MPSGraphTensor*>* init = [NSMutableArray array];
  for (mlir::Value v : op->getOperands()) [init addObject:L.value(v)];
  if (!L.ok()) return;

  NSArray<MPSGraphTensor*>* results = [L.graph()
      whileWithInitialInputs:init
      before:^(NSArray<MPSGraphTensor*>* inputs, NSMutableArray<MPSGraphTensor*>* out) {
        // bind carry args, evaluate cond → predicate, forward the carry to the after-block.
        for (unsigned i = 0; i < nCarry; ++i)
          L.bind(condBlock.getArgument(i), inputs[i]);
        std::vector<mlir::Value> condRets;
        WalkBlock(L, ModuleOf(op), condBlock, condRets);
        MPSGraphTensor* pred = condRets.empty() ? nil : L.value(condRets[0]);
        for (MPSGraphTensor* t : inputs) [out addObject:t];  // forward carry unchanged
        return pred;
      }
      after:^NSArray<MPSGraphTensor*>*(NSArray<MPSGraphTensor*>* bodyArgs) {
        for (unsigned i = 0; i < nCarry; ++i)
          L.bind(bodyBlock.getArgument(i), bodyArgs[i]);
        std::vector<mlir::Value> bodyRets;
        WalkBlock(L, ModuleOf(op), bodyBlock, bodyRets);
        NSMutableArray<MPSGraphTensor*>* next = [NSMutableArray array];
        for (mlir::Value v : bodyRets) {
          MPSGraphTensor* t = L.value(v);
          if (!t) { L.fail("jam: while body produced an unbound value"); return bodyArgs; }  // keep carry size; bail after
          [next addObject:t];
        }
        return next;
      }
      name:nil];
  if (!L.ok()) return;
  BindResults(L, op, results);
}

}  // namespace

void RegisterControlFlow() {
  RegisterOp("stablehlo.if", If);
  RegisterOp("stablehlo.case", Case);
  RegisterOp("stablehlo.while", While);
}

}  // namespace mccl_jax::jam
