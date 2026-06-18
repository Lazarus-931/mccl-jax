// ops/shape.mm — shape/layout ops: reshape, transpose, slice, dynamic_(update_)slice,
// broadcast_in_dim, reverse, pad, iota, concatenate, bitcast_convert, convert.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

void Reshape(Lowering& L, mlir::Operation* op) {
  if (op->getOperand(0).getType() == op->getResult(0).getType()) {  // no-op reshape: pass through
    L.bind(op->getResult(0), A(L, op));
    L.substitute(op->getResult(0), op->getOperand(0));
    return;
  }
  Set(L, op, [L.graph() reshapeTensor:A(L, op) withShape:ShapeArray(OutShape(op)) name:nil]);
}

void Transpose(Lowering& L, mlir::Operation* op) {
  auto t = mlir::cast<mlir::stablehlo::TransposeOp>(op);
  auto perm = t.getPermutation();
  bool identity = true;
  for (size_t i = 0; i < perm.size(); ++i)
    if (perm[i] != (int64_t)i) { identity = false; break; }
  if (identity) {  // identity permutation: pass through
    L.bind(op->getResult(0), A(L, op));
    L.substitute(op->getResult(0), op->getOperand(0));
    return;
  }
  Set(L, op, [L.graph() transposeTensor:A(L, op) permutation:IntArray(t.getPermutation()) name:nil]);
}

void Slice(Lowering& L, mlir::Operation* op) {
  auto s = mlir::cast<mlir::stablehlo::SliceOp>(op);
  Set(L, op, [L.graph() sliceTensor:A(L, op)
                              starts:IntArray(s.getStartIndices())
                                ends:IntArray(s.getLimitIndices())
                             strides:IntArray(s.getStrides())
                                name:nil]);
}

// Read a constant scalar start index, clamped to [0, maxStart]; fails the lowering otherwise.
bool ConstStart(Lowering& L, mlir::Value idx, int64_t maxStart, const char* what, int64_t* out) {
  auto cstOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(idx.getDefiningOp());
  if (!cstOp) { L.fail(std::string("jam: ") + what + " with non-constant start index"); return false; }
  auto ints = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cstOp.getValue());
  if (!ints || ints.getNumElements() < 1) {
    L.fail(std::string("jam: ") + what + " start index not an integer constant");
    return false;
  }
  int64_t s = (*ints.value_begin<llvm::APInt>()).getSExtValue();
  if (s < 0) s = 0;
  if (s > maxStart) s = maxStart;  // clamp in-bounds per StableHLO
  *out = s;
  return true;
}

// Is the value a compile-time-constant scalar integer?
bool IsConstStart(mlir::Value idx) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(idx.getDefiningOp());
  if (!cst) return false;
  return (bool)mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue());
}

// Build a runtime 1-D i32 start tensor from the scalar start operands, each clamped to [0, maxStart_i].
MPSGraphTensor* RuntimeStarts(Lowering& L, mlir::Operation* op, unsigned firstIdxOperand,
                              llvm::ArrayRef<int64_t> maxStart) {
  NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray array];
  MPSGraphTensor* zero = [L.graph() constantWithScalar:0.0 dataType:MPSDataTypeInt32];
  for (unsigned i = 0; i < maxStart.size(); ++i) {
    MPSGraphTensor* s = L.value(op->getOperand(firstIdxOperand + i));
    s = [L.graph() castTensor:s toType:MPSDataTypeInt32 name:nil];
    s = [L.graph() reshapeTensor:s withShape:@[ @1 ] name:nil];
    MPSGraphTensor* hi = [L.graph() constantWithScalar:(double)maxStart[i] dataType:MPSDataTypeInt32];
    s = [L.graph() clampWithTensor:s minValueTensor:zero maxValueTensor:hi name:nil];
    [parts addObject:s];
  }
  return [L.graph() concatTensors:parts dimension:0 name:nil];
}

// dynamic_slice(operand, start...): constant starts → static slice, else tensor-indexed slice.
void DynamicSlice(Lowering& L, mlir::Operation* op) {
  auto ds = mlir::cast<mlir::stablehlo::DynamicSliceOp>(op);
  llvm::ArrayRef<int64_t> inShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  llvm::ArrayRef<int64_t> sizes = ds.getSliceSizes();

  bool allConst = true;
  for (unsigned i = 0; i < sizes.size(); ++i)
    if (!IsConstStart(op->getOperand(1 + i))) { allConst = false; break; }

  if (allConst) {
    NSMutableArray<NSNumber*>* st = [NSMutableArray array];
    NSMutableArray<NSNumber*>* en = [NSMutableArray array];
    NSMutableArray<NSNumber*>* sr = [NSMutableArray array];
    for (unsigned i = 0; i < sizes.size(); ++i) {
      int64_t s0;
      if (!ConstStart(L, op->getOperand(1 + i), inShape[i] - sizes[i], "dynamic_slice", &s0)) return;
      [st addObject:@(s0)]; [en addObject:@(s0 + sizes[i])]; [sr addObject:@1];
    }
    Set(L, op, [L.graph() sliceTensor:A(L, op) starts:st ends:en strides:sr name:nil]);
    return;
  }

  std::vector<int64_t> maxStart, sizeVec;
  for (unsigned i = 0; i < sizes.size(); ++i) { maxStart.push_back(inShape[i] - sizes[i]); sizeVec.push_back(sizes[i]); }
  MPSGraphTensor* starts = RuntimeStarts(L, op, 1, maxStart);
  NSMutableData* sd = [NSMutableData dataWithLength:sizeVec.size() * sizeof(int32_t)];
  int32_t* sp = (int32_t*)[sd mutableBytes];
  for (size_t i = 0; i < sizeVec.size(); ++i) sp[i] = (int32_t)sizeVec[i];
  MPSGraphTensor* sizeT = [L.graph() constantWithData:sd shape:@[ @((NSInteger)sizeVec.size()) ]
                                             dataType:MPSDataTypeInt32];
  Set(L, op, [L.graph() sliceTensor:A(L, op) startTensor:starts sizeTensor:sizeT squeezeMask:0 name:nil]);
}

// dynamic_update_slice(operand, update, start...): constant starts → static, else runtime slice-update.
void DynamicUpdateSlice(Lowering& L, mlir::Operation* op) {
  auto inTy = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto upTy = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  llvm::ArrayRef<int64_t> inShape = inTy.getShape();
  llvm::ArrayRef<int64_t> upShape = upTy.getShape();
  int64_t rank = inTy.getRank();

  bool allConst = true;
  for (int64_t i = 0; i < rank; ++i)
    if (!IsConstStart(op->getOperand(2 + i))) { allConst = false; break; }

  if (allConst) {
    NSMutableArray<NSNumber*>* st = [NSMutableArray array];
    NSMutableArray<NSNumber*>* en = [NSMutableArray array];
    NSMutableArray<NSNumber*>* sr = [NSMutableArray array];
    for (int64_t i = 0; i < rank; ++i) {
      int64_t s0;
      if (!ConstStart(L, op->getOperand(2 + i), inShape[i] - upShape[i], "dynamic_update_slice", &s0)) return;
      [st addObject:@(s0)]; [en addObject:@(s0 + upShape[i])]; [sr addObject:@1];
    }
    Set(L, op, [L.graph() sliceUpdateDataTensor:A(L, op) updateTensor:B(L, op)
                                          starts:st ends:en strides:sr name:nil]);
    return;
  }

  std::vector<int64_t> maxStart;
  for (int64_t i = 0; i < rank; ++i) maxStart.push_back(inShape[i] - upShape[i]);
  MPSGraphTensor* starts = RuntimeStarts(L, op, 2, maxStart);
  // ends = starts + update_shape (computed at runtime); strides = 1.
  NSMutableData* ud = [NSMutableData dataWithLength:rank * sizeof(int32_t)];
  int32_t* up = (int32_t*)[ud mutableBytes];
  for (int64_t i = 0; i < rank; ++i) up[i] = (int32_t)upShape[i];
  MPSGraphTensor* upSizeT = [L.graph() constantWithData:ud shape:@[ @((NSInteger)rank) ] dataType:MPSDataTypeInt32];
  MPSGraphTensor* ends = [L.graph() additionWithPrimaryTensor:starts secondaryTensor:upSizeT name:nil];
  NSMutableData* od = [NSMutableData dataWithLength:rank * sizeof(int32_t)];
  int32_t* on = (int32_t*)[od mutableBytes];
  for (int64_t i = 0; i < rank; ++i) on[i] = 1;
  MPSGraphTensor* strides = [L.graph() constantWithData:od shape:@[ @((NSInteger)rank) ] dataType:MPSDataTypeInt32];
  Set(L, op, [L.graph() sliceUpdateDataTensor:A(L, op) updateTensor:B(L, op)
                                 startsTensor:starts endsTensor:ends stridesTensor:strides
                                    startMask:0 endMask:0 squeezeMask:0 name:nil]);
}

// broadcast_in_dim = reshape to output rank (1s in non-mapped dims) then broadcast.
void BroadcastInDim(Lowering& L, mlir::Operation* op) {
  auto b = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(op);
  llvm::ArrayRef<int64_t> bdims = b.getBroadcastDimensions();
  llvm::ArrayRef<int64_t> inShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  llvm::ArrayRef<int64_t> outShape = OutShape(op);
  std::vector<int64_t> reshaped(outShape.size(), 1);
  for (unsigned i = 0; i < bdims.size(); ++i) reshaped[bdims[i]] = inShape[i];
  MPSGraphTensor* r = Reshaped(L, A(L, op), reshaped);
  Set(L, op, Broadcasted(L, r, outShape));
}

void Reverse(Lowering& L, mlir::Operation* op) {
  auto r = mlir::cast<mlir::stablehlo::ReverseOp>(op);
  Set(L, op, [L.graph() reverseTensor:A(L, op) axes:IntArray(r.getDimensions()) name:nil]);
}

// Trace an SSA value back to a scalar splat constant through reshape/slice/broadcast/convert.
bool TraceScalarConstant(Lowering& L, mlir::Value v, double* out, int depth = 0) {
  if (depth > 16) return false;
  v = L.resolve(v);
  mlir::Operation* def = v.getDefiningOp();
  if (!def) return false;
  if (auto cst = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    if (!attr) return false;
    if (auto fp = mlir::dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
      *out = (*fp.value_begin<llvm::APFloat>()).convertToDouble();
      return true;
    }
    if (auto ip = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      *out = static_cast<double>((*ip.value_begin<llvm::APInt>()).getSExtValue());
      return true;
    }
    return false;
  }
  llvm::StringRef n = def->getName().getStringRef();
  if (n == "stablehlo.reshape" || n == "stablehlo.slice" ||
      n == "stablehlo.broadcast_in_dim" || n == "stablehlo.convert")
    return TraceScalarConstant(L, def->getOperand(0), out, depth + 1);
  return false;
}

// pad: constant low/high + interior (dilation) padding. Negative (crop) padding still a gap.
void Pad(Lowering& L, mlir::Operation* op) {
  auto p = mlir::cast<mlir::stablehlo::PadOp>(op);
  llvm::ArrayRef<int64_t> low = p.getEdgePaddingLow();
  llvm::ArrayRef<int64_t> high = p.getEdgePaddingHigh();
  llvm::ArrayRef<int64_t> interior = p.getInteriorPadding();
  for (int64_t v : interior) if (v < 0) { L.fail("jam: pad with negative interior padding unsupported"); return; }
  for (int64_t v : low) if (v < 0) { L.fail("jam: pad with negative (crop) padding unsupported"); return; }
  for (int64_t v : high) if (v < 0) { L.fail("jam: pad with negative (crop) padding unsupported"); return; }
  double fill = 0.0;
  if (!TraceScalarConstant(L, op->getOperand(1), &fill)) {
    L.fail("jam: pad with non-constant fill value");
    return;
  }
  MPSGraphTensor* t = A(L, op);
  bool hasInterior = false;
  for (int64_t v : interior) if (v != 0) hasInterior = true;
  if (hasInterior) {
    std::vector<int64_t> shape(
        mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape());
    t = InteriorDilate(L, t, shape, interior, fill);
  }
  Set(L, op, [L.graph() padTensor:t
                 withPaddingMode:MPSGraphPaddingModeConstant
                     leftPadding:IntArray(low)
                    rightPadding:IntArray(high)
                   constantValue:fill
                            name:nil]);
}

// iota: coordinate ramp along one axis, cast to the result dtype.
void Iota(Lowering& L, mlir::Operation* op) {
  auto io = mlir::cast<mlir::stablehlo::IotaOp>(op);
  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  MPSGraphTensor* ramp = [L.graph() coordinateAlongAxis:static_cast<int64_t>(io.getIotaDimension())
                                               withShape:ShapeArray(rt.getShape())
                                                    name:nil];
  Set(L, op, Casted(L, ramp, Lowering::MpsDType(rt.getElementType())));
}

void Concatenate(Lowering& L, mlir::Operation* op) {
  auto c = mlir::cast<mlir::stablehlo::ConcatenateOp>(op);
  NSMutableArray<MPSGraphTensor*>* arr = [NSMutableArray array];
  for (mlir::Value v : op->getOperands()) [arr addObject:L.value(v)];
  Set(L, op, [L.graph() concatTensors:arr dimension:static_cast<int64_t>(c.getDimension()) name:nil]);
}

void BitcastConvert(Lowering& L, mlir::Operation* op) {  // same-width reinterpret
  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  Set(L, op, [L.graph() reinterpretCastTensor:A(L, op) toType:Lowering::MpsDType(rt.getElementType()) name:nil]);
}

void Convert(Lowering& L, mlir::Operation* op) {  // dtype cast
  mlir::Type srcE = Lowering::ElementType(op->getOperand(0).getType());
  mlir::Type dstE = Lowering::ElementType(op->getResult(0).getType());
  // Device no-op when both element types narrow to the same MPSDataType (e.g. f64->f32, i64->i32,
  // or an identity f32->f32): skip the cast and pass the operand through.
  if (Lowering::MpsDType(srcE) == Lowering::MpsDType(dstE)) {
    L.bind(op->getResult(0), A(L, op));
    L.substitute(op->getResult(0), op->getOperand(0));
    return;
  }
  // UNSIGNED int -> float: jam backs ui with a signed int, so castTensor treats high-bit values as
  // negative (0x80000000 -> -2^31 instead of +2^31). Cast signed, then add 2^width wherever the
  // result came out negative to recover the unsigned magnitude. Robust if the cast was already
  // unsigned-correct (then there are no negatives and the select is a no-op).
  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcE);
  bool dstFloat = dstE.isF32() || dstE.isF16() || dstE.isBF16();
  if (srcInt && srcInt.isUnsigned() && dstFloat) {
    MPSGraphTensor* f = [L.graph() castTensor:A(L, op) toType:Lowering::MpsDType(dstE) name:nil];
    unsigned width = srcInt.getWidth() > 32 ? 32 : srcInt.getWidth();
    MPSGraphTensor* zero = [L.graph() constantWithScalar:0.0 dataType:f.dataType];
    MPSGraphTensor* twoW = [L.graph() constantWithScalar:(double)(1ULL << width) dataType:f.dataType];
    MPSGraphTensor* neg = [L.graph() lessThanWithPrimaryTensor:f secondaryTensor:zero name:nil];
    MPSGraphTensor* up = [L.graph() additionWithPrimaryTensor:f secondaryTensor:twoW name:nil];
    Set(L, op, [L.graph() selectWithPredicateTensor:neg truePredicateTensor:up falsePredicateTensor:f name:nil]);
    return;
  }
  Set(L, op, [L.graph() castTensor:A(L, op) toType:Lowering::MpsDType(dstE) name:nil]);
}

}  // namespace

void RegisterShape() {
  RegisterOp("stablehlo.reshape", Reshape);
  RegisterOp("stablehlo.transpose", Transpose);
  RegisterOp("stablehlo.slice", Slice);
  RegisterOp("stablehlo.dynamic_slice", DynamicSlice);
  RegisterOp("stablehlo.dynamic_update_slice", DynamicUpdateSlice);
  RegisterOp("stablehlo.broadcast_in_dim", BroadcastInDim);
  RegisterOp("stablehlo.reverse", Reverse);
  RegisterOp("stablehlo.pad", Pad);
  RegisterOp("stablehlo.iota", Iota);
  RegisterOp("stablehlo.concatenate", Concatenate);
  RegisterOp("stablehlo.bitcast_convert", BitcastConvert);
  RegisterOp("stablehlo.convert", Convert);
}

}  // namespace mccl_jax::jam
