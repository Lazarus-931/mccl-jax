// ops/binary.mm — atan2, remainder, bitwise/logical and/or/xor, shifts, clamp, compare, select.

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mlir/IR/Operation.h"
#include "src/jam/ops/ops_common.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mccl_jax::jam {
namespace {

bool IsBoolOperand(mlir::Operation* op, unsigned i) {
  return Lowering::ElementType(op->getOperand(i).getType()).isInteger(1);
}

void Atan2(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() atan2WithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Remainder(Lowering& L, mlir::Operation* op) {
  // StableHLO remainder == C fmod: result has the sign of the dividend (signed/float). For UNSIGNED
  // ints do the modulo in the unsigned type (else high-bit values are negative — broke randint).
  MPSDataType uty;
  if (UnsignedIntOperand(op, 0, uty)) {
    MPSGraphTensor* au = [L.graph() reinterpretCastTensor:A(L, op) toType:uty name:nil];
    MPSGraphTensor* bu = [L.graph() reinterpretCastTensor:B(L, op) toType:uty name:nil];
    MPSGraphTensor* r = [L.graph() moduloWithPrimaryTensor:au secondaryTensor:bu name:nil];
    Set(L, op, [L.graph() reinterpretCastTensor:r
                                          toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType()))
                                            name:nil]);
    return;
  }
  Set(L, op, [L.graph() moduloWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}

// bitwise / logical (bool → logical, int → bitwise)
void And(Lowering& L, mlir::Operation* op) {
  if (IsBoolOperand(op, 0))
    Set(L, op, [L.graph() logicalANDWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
  else
    Set(L, op, [L.graph() bitwiseANDWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Or(Lowering& L, mlir::Operation* op) {
  if (IsBoolOperand(op, 0))
    Set(L, op, [L.graph() logicalORWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
  else
    Set(L, op, [L.graph() bitwiseORWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void Xor(Lowering& L, mlir::Operation* op) {
  if (IsBoolOperand(op, 0))
    Set(L, op, [L.graph() logicalXORWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
  else
    Set(L, op, [L.graph() bitwiseXORWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}

// shifts
void ShiftLeft(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() bitwiseLeftShiftWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void ShiftRightArithmetic(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() bitwiseRightShiftWithPrimaryTensor:A(L, op) secondaryTensor:B(L, op) name:nil]);
}
void ShiftRightLogical(Lowering& L, mlir::Operation* op) {
  // reinterpret to uint32, shift, reinterpret back to int32.
  MPSGraphTensor* au = [L.graph() reinterpretCastTensor:A(L, op) toType:MPSDataTypeUInt32 name:nil];
  MPSGraphTensor* bu = [L.graph() reinterpretCastTensor:B(L, op) toType:MPSDataTypeUInt32 name:nil];
  MPSGraphTensor* su = [L.graph() bitwiseRightShiftWithPrimaryTensor:au secondaryTensor:bu name:nil];
  Set(L, op, [L.graph() reinterpretCastTensor:su toType:MPSDataTypeInt32 name:nil]);
}

// clamp(min, operand, max)
void Clamp(Lowering& L, mlir::Operation* op) {
  MPSDataType uty;  // unsigned clamp: reinterpret all three (signed clamp mis-orders high-bit bounds)
  if (UnsignedIntOperand(op, 1, uty)) {
    MPSGraphTensor* r = [L.graph() clampWithTensor:[L.graph() reinterpretCastTensor:B(L, op) toType:uty name:nil]
                                    minValueTensor:[L.graph() reinterpretCastTensor:A(L, op) toType:uty name:nil]
                                    maxValueTensor:[L.graph() reinterpretCastTensor:C(L, op) toType:uty name:nil] name:nil];
    Set(L, op, [L.graph() reinterpretCastTensor:r toType:Lowering::MpsDType(Lowering::ElementType(op->getResult(0).getType())) name:nil]);
    return;
  }
  Set(L, op, [L.graph() clampWithTensor:B(L, op) minValueTensor:A(L, op) maxValueTensor:C(L, op) name:nil]);
}
// select(pred, on_true, on_false)
void Select(Lowering& L, mlir::Operation* op) {
  Set(L, op, [L.graph() selectWithPredicateTensor:A(L, op)
                                truePredicateTensor:B(L, op)
                               falsePredicateTensor:C(L, op)
                                               name:nil]);
}

// compare (→ bool tensor)
void Compare(Lowering& L, mlir::Operation* op) {
  auto cmp = mlir::cast<mlir::stablehlo::CompareOp>(op);
  MPSGraphTensor* a = A(L, op);
  MPSGraphTensor* b = B(L, op);
  using Dir = mlir::stablehlo::ComparisonDirection;
  Dir dir = cmp.getComparisonDirection();
  // Unsigned ordering: jam backs ui32 with signed Int32, and MPSGraph's comparison ops compare as
  // SIGNED even on a uint-typed tensor (verified). Map unsigned order onto signed order by flipping
  // the sign bit (xor MSB): unsigned a<b ⇔ (a^0x80…0) <_signed (b^0x80…0). EQ/NE are bitwise so skip.
  MPSDataType uty;
  bool unsignedType = (cmp.getCompareType() == mlir::stablehlo::ComparisonType::UNSIGNED) || UnsignedIntOperand(op, 0, uty);
  if ((dir == Dir::GT || dir == Dir::GE || dir == Dir::LT || dir == Dir::LE) && unsignedType) {
    int width = Lowering::ElementType(op->getOperand(0).getType()).getIntOrFloatBitWidth();
    if (width > 32) width = 32;  // jam narrows i64/u64 to 32-bit
    double msb = -static_cast<double>(1LL << (width - 1));  // two's-complement MSB pattern as signed
    MPSGraphTensor* flip = [L.graph() constantWithScalar:msb dataType:a.dataType];
    a = [L.graph() bitwiseXORWithPrimaryTensor:a secondaryTensor:flip name:nil];
    b = [L.graph() bitwiseXORWithPrimaryTensor:b secondaryTensor:flip name:nil];
  }
  switch (dir) {
    case Dir::GT: Set(L, op, [L.graph() greaterThanWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
    case Dir::GE: Set(L, op, [L.graph() greaterThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
    case Dir::LT: Set(L, op, [L.graph() lessThanWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
    case Dir::LE: Set(L, op, [L.graph() lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
    case Dir::EQ: Set(L, op, [L.graph() equalWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
    case Dir::NE: Set(L, op, [L.graph() notEqualWithPrimaryTensor:a secondaryTensor:b name:nil]); break;
  }
}

}  // namespace

void RegisterBinary() {
  RegisterOp("stablehlo.atan2", Atan2);
  RegisterOp("stablehlo.remainder", Remainder);
  RegisterOp("stablehlo.and", And);
  RegisterOp("stablehlo.or", Or);
  RegisterOp("stablehlo.xor", Xor);
  RegisterOp("stablehlo.shift_left", ShiftLeft);
  RegisterOp("stablehlo.shift_right_arithmetic", ShiftRightArithmetic);
  RegisterOp("stablehlo.shift_right_logical", ShiftRightLogical);
  RegisterOp("stablehlo.clamp", Clamp);
  RegisterOp("stablehlo.select", Select);
  RegisterOp("stablehlo.compare", Compare);
}

}  // namespace mccl_jax::jam
