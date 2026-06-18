// ops/structural.mm — constant (dense + splat, incl. non-finite ±inf/nan).

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <cstring>

#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

// Encode a float as IEEE half (truncating the mantissa; subnormal-flush + inf/overflow).
uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

uint16_t FloatToBF16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7FFFu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}

void Constant(Lowering& L, mlir::Operation* op) {
  auto cst = mlir::cast<mlir::stablehlo::ConstantOp>(op);
  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Type elem = rt.getElementType();
  MPSDataType mps = Lowering::MpsDType(elem);
  NSArray<NSNumber*>* mshape = ShapeArray(rt.getShape());

  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!attr) { L.fail("jam: constant without dense value attribute"); return; }
  bool isFP = mlir::isa<mlir::FloatType>(elem);

  // Splat: one MPSGraph scalar constant broadcast to the shape (handles ±inf/nan).
  if (attr.isSplat()) {
    double v = isFP ? attr.getSplatValue<llvm::APFloat>().convertToDouble()
                    : static_cast<double>(attr.getSplatValue<llvm::APInt>().getSExtValue());
    Set(L, op, [L.graph() constantWithScalar:v shape:mshape dataType:mps]);
    return;
  }

  // Dense: pack raw little-endian bytes in the device dtype, then constantWithData.
  int64_t n = rt.getNumElements();
  size_t bytes = (mps == MPSDataTypeFloat32 || mps == MPSDataTypeInt32) ? 4
               : (mps == MPSDataTypeFloat16 || mps == MPSDataTypeBFloat16) ? 2
               : 1;
  NSMutableData* nd = [NSMutableData dataWithLength:(NSUInteger)(n * bytes)];
  void* p = [nd mutableBytes];

  if (isFP) {
    int64_t i = 0;
    for (const llvm::APFloat& fv : attr.getValues<llvm::APFloat>()) {
      double dv = fv.convertToDouble();
      if (mps == MPSDataTypeFloat16) reinterpret_cast<uint16_t*>(p)[i] = FloatToHalf((float)dv);
      else if (mps == MPSDataTypeBFloat16) reinterpret_cast<uint16_t*>(p)[i] = FloatToBF16((float)dv);
      else reinterpret_cast<float*>(p)[i] = static_cast<float>(dv);
      ++i;
    }
  } else {
    int64_t i = 0;
    for (const llvm::APInt& iv : attr.getValues<llvm::APInt>()) {
      int64_t dv = elem.isInteger(1) ? (iv.getBoolValue() ? 1 : 0) : iv.getSExtValue();
      if (mps == MPSDataTypeInt8) reinterpret_cast<int8_t*>(p)[i] = static_cast<int8_t>(dv);
      else if (mps == MPSDataTypeBool) reinterpret_cast<uint8_t*>(p)[i] = static_cast<uint8_t>(dv != 0);
      else reinterpret_cast<int32_t*>(p)[i] = static_cast<int32_t>(dv);
      ++i;
    }
  }
  Set(L, op, [L.graph() constantWithData:nd shape:mshape dataType:mps]);
}

// ---- deprecated ops (safety net; normalization usually rewrites these first) ----

// stablehlo.broadcast: prepend `broadcast_sizes` leading dims, then broadcast to out shape.
void Broadcast(Lowering& L, mlir::Operation* op) {
  auto b = mlir::cast<mlir::stablehlo::BroadcastOp>(op);
  llvm::ArrayRef<int64_t> sizes = b.getBroadcastSizes();
  llvm::ArrayRef<int64_t> inShape =
      mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getShape();
  std::vector<int64_t> reshaped(sizes.size(), 1);
  for (int64_t d : inShape) reshaped.push_back(d);
  MPSGraphTensor* r = Reshaped(L, A(L, op), reshaped);
  Set(L, op, Broadcasted(L, r, OutShape(op)));
}

}  // namespace

void RegisterStructural() {
  RegisterOp("stablehlo.constant", Constant);
  RegisterOp("stablehlo.broadcast", Broadcast);
}

}  // namespace mccl_jax::jam
