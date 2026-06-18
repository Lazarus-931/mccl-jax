// ops/dynamic.mm — dynamic-shape family; handlers use the static result shape, dynamic shapes fail.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

bool HasStaticResult(mlir::Operation* op) {
  auto rt = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  return rt && rt.hasStaticShape();
}

// dynamic_reshape(operand, shape) → static reshape to the result shape.
void DynamicReshape(Lowering& L, mlir::Operation* op) {
  if (!HasStaticResult(op)) { L.fail("jam: dynamic_reshape with dynamic result shape unsupported"); return; }
  Set(L, op, Reshaped(L, A(L, op), OutShape(op)));
}

// dynamic_broadcast_in_dim(operand, output_dimensions) → static broadcast_in_dim.
void DynamicBroadcastInDim(Lowering& L, mlir::Operation* op) {
  if (!HasStaticResult(op)) { L.fail("jam: dynamic_broadcast_in_dim with dynamic result unsupported"); return; }
  auto b = mlir::cast<mlir::stablehlo::DynamicBroadcastInDimOp>(op);
  llvm::ArrayRef<int64_t> bdims = b.getBroadcastDimensions();
  llvm::ArrayRef<int64_t> inShape = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  llvm::ArrayRef<int64_t> outShape = OutShape(op);
  std::vector<int64_t> reshaped(outShape.size(), 1);
  for (unsigned i = 0; i < bdims.size(); ++i) reshaped[bdims[i]] = inShape[i];
  MPSGraphTensor* r = Reshaped(L, A(L, op), reshaped);
  Set(L, op, Broadcasted(L, r, outShape));
}

// dynamic_iota(shape) → static iota along the iota dimension.
void DynamicIota(Lowering& L, mlir::Operation* op) {
  if (!HasStaticResult(op)) { L.fail("jam: dynamic_iota with dynamic result unsupported"); return; }
  auto io = mlir::cast<mlir::stablehlo::DynamicIotaOp>(op);
  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  MPSGraphTensor* ramp = [L.graph() coordinateAlongAxis:(int64_t)io.getIotaDimension()
                                               withShape:ShapeArray(rt.getShape()) name:nil];
  Set(L, op, Casted(L, ramp, Lowering::MpsDType(rt.getElementType())));
}

// Read a 1-D integer constant operand into a vector. Returns false if not a constant.
bool ConstIntVec(mlir::Value v, std::vector<int64_t>* out) {
  auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  auto ints = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue());
  if (!ints) return false;
  for (const llvm::APInt& x : ints.getValues<llvm::APInt>()) out->push_back(x.getSExtValue());
  return true;
}

// real_dynamic_slice: lower to a static slice when start/limit/stride are constants.
void RealDynamicSlice(Lowering& L, mlir::Operation* op) {
  std::vector<int64_t> start, limit, stride;
  if (!ConstIntVec(op->getOperand(1), &start) || !ConstIntVec(op->getOperand(2), &limit) ||
      !ConstIntVec(op->getOperand(3), &stride)) {
    L.fail("jam: real_dynamic_slice with non-constant bounds unsupported");
    return;
  }
  Set(L, op, [L.graph() sliceTensor:A(L, op) starts:IntArray(start) ends:IntArray(limit)
                             strides:IntArray(stride) name:nil]);
}

// dynamic_pad: lower to a static constant pad when low/high (>=0) and interior (==0) are constants.
void DynamicPad(Lowering& L, mlir::Operation* op) {
  std::vector<int64_t> low, high, interior;
  if (!ConstIntVec(op->getOperand(2), &low) || !ConstIntVec(op->getOperand(3), &high) ||
      !ConstIntVec(op->getOperand(4), &interior)) {
    L.fail("jam: dynamic_pad with non-constant padding unsupported");
    return;
  }
  for (int64_t v : interior) if (v != 0) { L.fail("jam: dynamic_pad interior padding unsupported"); return; }
  for (int64_t v : low) if (v < 0) { L.fail("jam: dynamic_pad negative padding unsupported"); return; }
  for (int64_t v : high) if (v < 0) { L.fail("jam: dynamic_pad negative padding unsupported"); return; }
  // pad value (operand 1): use 0.0 unless it's a constant.
  double fill = 0.0;
  std::vector<int64_t> fillVec;
  if (auto cst = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op->getOperand(1).getDefiningOp())) {
    if (auto fp = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue()))
      fill = (*fp.value_begin<llvm::APFloat>()).convertToDouble();
    else if (auto ip = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue()))
      fill = (double)(*ip.value_begin<llvm::APInt>()).getSExtValue();
  }
  Set(L, op, [L.graph() padTensor:A(L, op) withPaddingMode:MPSGraphPaddingModeConstant
                   leftPadding:IntArray(low) rightPadding:IntArray(high)
                 constantValue:fill name:nil]);
}

// get_dimension_size(operand) → scalar i32 = static size of dimension `dimension`.
void GetDimensionSize(Lowering& L, mlir::Operation* op) {
  auto g = mlir::cast<mlir::stablehlo::GetDimensionSizeOp>(op);
  auto inTy = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  int64_t dim = g.getDimension();
  if (dim < 0 || dim >= inTy.getRank() || inTy.isDynamicDim(dim)) {
    L.fail("jam: get_dimension_size on a dynamic dimension unsupported");
    return;
  }
  Set(L, op, [L.graph() constantWithScalar:(double)inTy.getDimSize(dim)
                                      shape:@[ @1 ] dataType:MPSDataTypeInt32]);
}

// set_dimension_size: no-op on the data; forward the operand.
void SetDimensionSize(Lowering& L, mlir::Operation* op) {
  Set(L, op, A(L, op));
}

}  // namespace

void RegisterDynamic() {
  RegisterOp("stablehlo.dynamic_reshape", DynamicReshape);
  RegisterOp("stablehlo.dynamic_broadcast_in_dim", DynamicBroadcastInDim);
  RegisterOp("stablehlo.dynamic_iota", DynamicIota);
  RegisterOp("stablehlo.real_dynamic_slice", RealDynamicSlice);
  RegisterOp("stablehlo.dynamic_pad", DynamicPad);
  RegisterOp("stablehlo.get_dimension_size", GetDimensionSize);
  RegisterOp("stablehlo.set_dimension_size", SetDimensionSize);
  // dynamic_slice/gather/conv: handled elsewhere once canonicalize-dynamism folds them.
}

}  // namespace mccl_jax::jam
