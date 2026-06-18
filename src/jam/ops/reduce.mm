#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

mlir::Operation* FirstBodyOp(mlir::Region& region) {
  if (region.empty()) return nullptr;
  for (mlir::Operation& inner : region.front()) {
    if (inner.getName().getStringRef() == "stablehlo.return") continue;
    return &inner;
  }
  return nullptr;
}

bool DetectArgReduce(mlir::stablehlo::ReduceOp red, bool* isMax, int64_t* axis) {
  if (red->getNumOperands() != 4 || red->getNumResults() != 2) return false;
  llvm::ArrayRef<int64_t> dims = red.getDimensions();
  if (dims.size() != 1) return false;
  for (mlir::Operation& inner : red.getBody().front()) {
    if (auto cmp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&inner)) {
      using Dir = mlir::stablehlo::ComparisonDirection;
      Dir d = cmp.getComparisonDirection();
      if (d == Dir::GT) { *isMax = true; *axis = dims[0]; return true; }
      if (d == Dir::LT) { *isMax = false; *axis = dims[0]; return true; }
      return false;
    }
  }
  return false;
}

void Reduce(Lowering& L, mlir::Operation* op) {
  auto red = mlir::cast<mlir::stablehlo::ReduceOp>(op);

  bool isMax = false;
  int64_t argAxis = 0;
  if (DetectArgReduce(red, &isMax, &argAxis)) {
    MPSGraphTensor* a = L.value(op->getOperand(0));

    MPSDataType uty;
    bool uns = UnsignedIntOperand(op, 0, uty);
    MPSGraphTensor* msb = nil;
    if (uns) {
      unsigned w = (uty == MPSDataTypeUInt8) ? 8 : (uty == MPSDataTypeUInt16) ? 16 : 32;
      msb = [L.graph() constantWithScalar:-(double)(1ULL << (w - 1)) dataType:a.dataType];
      a = [L.graph() bitwiseXORWithPrimaryTensor:a secondaryTensor:msb name:nil];
    }
    MPSGraphTensor* val = isMax
        ? [L.graph() reductionMaximumWithTensor:a axis:argAxis name:nil]
        : [L.graph() reductionMinimumWithTensor:a axis:argAxis name:nil];
    MPSGraphTensor* idx = isMax
        ? [L.graph() reductionArgMaximumWithTensor:a axis:argAxis name:nil]
        : [L.graph() reductionArgMinimumWithTensor:a axis:argAxis name:nil];

    auto vt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (uns) val = [L.graph() bitwiseXORWithPrimaryTensor:val secondaryTensor:msb name:nil];
    Set(L, op, Reshaped(L, val, vt.getShape()));

    auto it = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
    idx = Casted(L, idx, Lowering::MpsDType(it.getElementType()));
    idx = Reshaped(L, idx, it.getShape());
    L.bind(op->getResult(1), idx);
    return;
  }

  mlir::Operation* body = FirstBodyOp(red.getBody());
  if (!body) { L.fail("jam: reduce with empty reducer body"); return; }
  llvm::StringRef rk = body->getName().getStringRef();

  enum { kAdd, kMax, kMin, kMul, kAnd, kOr } kind;
  if (rk == "stablehlo.add") kind = kAdd;
  else if (rk == "stablehlo.maximum") kind = kMax;
  else if (rk == "stablehlo.minimum") kind = kMin;
  else if (rk == "stablehlo.multiply") kind = kMul;
  else if (rk == "stablehlo.and") kind = kAnd;
  else if (rk == "stablehlo.or") kind = kOr;
  else { L.fail("jam: unsupported reduce body op '" + rk.str() + "'"); return; }

  MPSGraphTensor* acc = L.value(op->getOperand(0));

  MPSDataType uty;
  bool uns = (kind == kMax || kind == kMin) && UnsignedIntOperand(op, 0, uty);
  if (uns) acc = [L.graph() reinterpretCastTensor:acc toType:uty name:nil];
  for (int64_t dim : red.getDimensions()) {
    switch (kind) {
      case kAdd: acc = [L.graph() reductionSumWithTensor:acc axis:dim name:nil]; break;
      case kMax: acc = [L.graph() reductionMaximumWithTensor:acc axis:dim name:nil]; break;
      case kMin: acc = [L.graph() reductionMinimumWithTensor:acc axis:dim name:nil]; break;
      case kMul: acc = [L.graph() reductionProductWithTensor:acc axis:dim name:nil]; break;
      case kAnd: acc = [L.graph() reductionAndWithTensor:acc axis:dim name:nil]; break;
      case kOr:  acc = [L.graph() reductionOrWithTensor:acc axis:dim name:nil]; break;
    }
  }
  if (uns) acc = [L.graph() reinterpretCastTensor:acc toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType())) name:nil];
  Set(L, op, Reshaped(L, acc, OutShape(op)));
}

static bool AllOnes(std::optional<llvm::ArrayRef<int64_t>> a) {
  if (!a.has_value()) return true;
  for (int64_t v : *a) if (v != 1) return false;
  return true;
}

static bool TryCumulative(Lowering& L, mlir::Operation* op, mlir::stablehlo::ReduceWindowOp rw,
                          bool isAdd, bool isMul, bool isMax, bool isMin) {
  if (!isAdd && !isMul && !isMax && !isMin) return false;
  if (!AllOnes(rw.getWindowStrides()) || !AllOnes(rw.getBaseDilations()) ||
      !AllOnes(rw.getWindowDilations()))
    return false;
  auto padAttr = rw.getPadding();
  if (!padAttr.has_value()) return false;

  llvm::ArrayRef<int64_t> inShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  int64_t rank = static_cast<int64_t>(inShape.size());
  llvm::ArrayRef<int64_t> wd = rw.getWindowDimensions();
  std::vector<int64_t> padVals;
  for (const llvm::APInt& v : padAttr->getValues<llvm::APInt>()) padVals.push_back(v.getSExtValue());
  if (static_cast<int64_t>(padVals.size()) != rank * 2 || static_cast<int64_t>(wd.size()) != rank)
    return false;

  int64_t axis = -1;
  bool reverse = false, ok = true;
  for (int64_t i = 0; ok && i < rank; ++i) {
    int64_t lo = padVals[i * 2], hi = padVals[i * 2 + 1];
    if (wd[i] == 1 && lo == 0 && hi == 0) continue;
    if (wd[i] == inShape[i] && axis == -1) {
      if (lo == inShape[i] - 1 && hi == 0) { axis = i; reverse = false; continue; }
      if (hi == inShape[i] - 1 && lo == 0) { axis = i; reverse = true; continue; }
    }
    ok = false;
  }
  if (!ok || axis == -1) return false;

  MPSGraphTensor* a = L.value(op->getOperand(0));
  if (isAdd)
    Set(L, op, [L.graph() cumulativeSumWithTensor:a axis:axis exclusive:NO reverse:reverse name:nil]);
  else if (isMul)
    Set(L, op, [L.graph() cumulativeProductWithTensor:a axis:axis exclusive:NO reverse:reverse name:nil]);
  else if (isMax)
    Set(L, op, [L.graph() cumulativeMaximumWithTensor:a axis:axis exclusive:NO reverse:reverse name:nil]);
  else
    Set(L, op, [L.graph() cumulativeMinimumWithTensor:a axis:axis exclusive:NO reverse:reverse name:nil]);
  return true;
}

static bool TryPool2D(Lowering& L, mlir::Operation* op, mlir::stablehlo::ReduceWindowOp rw,
                      bool isAdd, bool isMax) {
  if (!isAdd && !isMax) return false;
  llvm::ArrayRef<int64_t> wd = rw.getWindowDimensions();
  if (wd.size() != 4) return false;
  if (!AllOnes(rw.getBaseDilations()) || !AllOnes(rw.getWindowDilations())) return false;

  std::vector<int> poolDims;
  for (int i = 0; i < 4; ++i) if (wd[i] != 1) poolDims.push_back(i);
  if (poolDims.empty()) { poolDims = {1, 2}; }
  if (poolDims.size() != 2 || poolDims[0] != 1 || poolDims[1] != 2) return false;

  int64_t kh = wd[1], kw = wd[2];
  int64_t sh = 1, sw = 1;
  if (auto s = rw.getWindowStrides()) {
    auto sr = *s;
    if (sr.size() == 4) { if (sr[0] != 1 || sr[3] != 1) return false; sh = sr[1]; sw = sr[2]; }
  }
  int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
  if (auto padAttr = rw.getPadding()) {
    std::vector<int64_t> pv;
    for (const llvm::APInt& v : padAttr->getValues<llvm::APInt>()) pv.push_back(v.getSExtValue());
    if (pv.size() != 8) return false;
    if (pv[0] != 0 || pv[1] != 0 || pv[6] != 0 || pv[7] != 0) return false;
    padTop = pv[2]; padBottom = pv[3]; padLeft = pv[4]; padRight = pv[5];
  }

  MPSGraphPooling2DOpDescriptor* desc =
      [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kw
                                                  kernelHeight:(NSUInteger)kh
                                                     strideInX:(NSUInteger)sw
                                                     strideInY:(NSUInteger)sh
                                               dilationRateInX:1
                                               dilationRateInY:1
                                                   paddingLeft:(NSUInteger)padLeft
                                                  paddingRight:(NSUInteger)padRight
                                                    paddingTop:(NSUInteger)padTop
                                                 paddingBottom:(NSUInteger)padBottom
                                                  paddingStyle:MPSGraphPaddingStyleExplicit
                                                    dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
  MPSGraphTensor* a = L.value(op->getOperand(0));
  if (isMax) {
    Set(L, op, [L.graph() maxPooling2DWithSourceTensor:a descriptor:desc name:nil]);
  } else {
    desc.includeZeroPadToAverage = YES;
    MPSGraphTensor* avg = [L.graph() avgPooling2DWithSourceTensor:a descriptor:desc name:nil];
    MPSGraphTensor* area = [L.graph() constantWithScalar:(double)(kh * kw) dataType:avg.dataType];
    Set(L, op, [L.graph() multiplicationWithPrimaryTensor:avg secondaryTensor:area name:nil]);
  }
  return true;
}

void ReduceWindow(Lowering& L, mlir::Operation* op) {
  auto rw = mlir::cast<mlir::stablehlo::ReduceWindowOp>(op);
  if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
    L.fail("jam: reduce_window: only single-input forms supported");
    return;
  }
  mlir::Operation* body = FirstBodyOp(rw.getBody());
  if (!body) { L.fail("jam: reduce_window with empty body"); return; }
  llvm::StringRef bn = body->getName().getStringRef();
  bool isAdd = (bn == "stablehlo.add");
  bool isMul = (bn == "stablehlo.multiply");
  bool isMax = (bn == "stablehlo.maximum");
  bool isMin = (bn == "stablehlo.minimum");

  if (TryCumulative(L, op, rw, isAdd, isMul, isMax, isMin)) return;
  if (TryPool2D(L, op, rw, isAdd, isMax)) return;
  L.fail("jam: reduce_window: only cumulative (cumsum/cumprod/cummax/cummin) and 2D max/sum pooling supported");
}

}

void RegisterReduce() {
  RegisterOp("stablehlo.reduce", Reduce);
  RegisterOp("stablehlo.reduce_window", ReduceWindow);
}

}
