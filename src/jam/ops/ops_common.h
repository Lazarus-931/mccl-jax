#ifndef MCCL_JAX_SRC_JAM_OPS_OPS_COMMON_H_
#define MCCL_JAX_SRC_JAM_OPS_OPS_COMMON_H_

// Shared helpers for the ops/*.mm lowering handlers (read operands A/B/C, bind result via Set).

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/lowering_internal.h"

namespace mccl_jax::jam {

// Operands as MPSGraph tensors (nil + L.fail() on an unbound value).
inline MPSGraphTensor* A(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(0)); }
inline MPSGraphTensor* B(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(1)); }
inline MPSGraphTensor* C(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(2)); }

// Bind result 0 (the single-result common case).
inline void Set(Lowering& L, mlir::Operation* op, MPSGraphTensor* t) {
  L.bind(op->getResult(0), t);
}

// True (+ matching unsigned MPSDataType in `uty`) iff operand `i` is a genuine MLIR unsigned integer.
// jam backs unsigned ints with a SIGNED MPSGraph int (MpsDType maps ui32→Int32), so divide/modulo/
// compare/convert on them use signed semantics and need fixing (reinterpret, or a sign-bit flip).
inline bool UnsignedIntOperand(mlir::Operation* op, unsigned i, MPSDataType& uty) {
  auto it = mlir::dyn_cast<mlir::IntegerType>(Lowering::ElementType(op->getOperand(i).getType()));
  if (!it || !it.isUnsigned()) return false;
  switch (it.getWidth()) {
    case 8:  uty = MPSDataTypeUInt8; break;
    case 16: uty = MPSDataTypeUInt16; break;
    default: uty = MPSDataTypeUInt32; break;  // 32, and 64 which jam narrows to 32
  }
  return true;
}

// int64 dims → NSArray<NSNumber*> for MPSGraph shapes (rank-0 modeled as [1]).
inline NSArray<NSNumber*>* ShapeArray(llvm::ArrayRef<int64_t> dims) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d : dims) [a addObject:@(d)];
  if (a.count == 0) [a addObject:@1];
  return a;
}

// Raw int64 list → NSArray<NSNumber*> (no rank-0 fixup; for axes/strides/perms).
inline NSArray<NSNumber*>* IntArray(llvm::ArrayRef<int64_t> v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(x)];
  return a;
}

// Static shape of an op result.
inline llvm::ArrayRef<int64_t> OutShape(mlir::Operation* op, unsigned result = 0) {
  return mlir::cast<mlir::RankedTensorType>(op->getResult(result).getType()).getShape();
}

// Reshape `t` to `dims` ONLY if it isn't already that shape — MPSGraph doesn't fold no-op reshapes,
// so emitting one is a dead kernel. Handlers internally reshape to the result shape (gather, reduce,
// scatter, …) which is frequently already correct; routing those through here drops the dead op.
inline MPSGraphTensor* Reshaped(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> dims) {
  NSArray<NSNumber*>* want = ShapeArray(dims);
  NSArray<NSNumber*>* cur = t.shape;  // static for jam tensors
  if (cur != nil && [cur isEqualToArray:want]) return t;
  return [L.graph() reshapeTensor:t withShape:want name:nil];
}

// Cast `t` to `dt` only if it isn't already that dtype (no-op casts, e.g. an i32 argmax index cast
// to an i32 result, or an i32 iota cast to i32 — MPSGraph doesn't fold them either).
inline MPSGraphTensor* Casted(Lowering& L, MPSGraphTensor* t, MPSDataType dt) {
  if (t.dataType == dt) return t;
  return [L.graph() castTensor:t toType:dt name:nil];
}

// Broadcast `t` to `dims` only if it isn't already that shape (a broadcast to the current shape is a
// no-op MPSGraph won't fold).
inline MPSGraphTensor* Broadcasted(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> dims) {
  NSArray<NSNumber*>* want = ShapeArray(dims);
  NSArray<NSNumber*>* cur = t.shape;
  if (cur != nil && [cur isEqualToArray:want]) return t;
  return [L.graph() broadcastTensor:t toShape:want name:nil];
}

// Transpose `t` by `perm` only if `perm` reorders (identity perm = no-op, e.g. a conv result transpose
// when the output layout is already NHWC).
inline MPSGraphTensor* Transposed(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> perm) {
  for (std::size_t i = 0; i < perm.size(); ++i)
    if (perm[i] != (int64_t)i) return [L.graph() transposeTensor:t permutation:IntArray(perm) name:nil];
  return t;
}

// Interior-dilate `t` (shape `shape`, updated in place): insert `interior[d]` copies of `fill`
// between adjacent elements along each axis d (reshape→pad-the-gap→reshape→trim). Used by both
// stablehlo.pad (interior padding) and convolution (lhs_dilation / transposed conv).
inline MPSGraphTensor* InteriorDilate(Lowering& L, MPSGraphTensor* t, std::vector<int64_t>& shape,
                                      llvm::ArrayRef<int64_t> interior, double fill) {
  int64_t rank = (int64_t)shape.size();
  for (int64_t d = 0; d < rank; ++d) {
    int64_t pi = (d < (int64_t)interior.size()) ? interior[d] : 0, N = shape[d];
    if (pi <= 0 || N <= 1) continue;
    std::vector<int64_t> ins(shape.begin(), shape.end());
    ins.insert(ins.begin() + d + 1, 1);
    t = [L.graph() reshapeTensor:t withShape:IntArray(ins) name:nil];
    std::vector<int64_t> lo(ins.size(), 0), hi(ins.size(), 0);
    hi[d + 1] = pi;
    t = [L.graph() padTensor:t withPaddingMode:MPSGraphPaddingModeConstant
               leftPadding:IntArray(lo) rightPadding:IntArray(hi) constantValue:fill name:nil];
    std::vector<int64_t> col(shape.begin(), shape.end());
    col[d] = N * (1 + pi);
    t = [L.graph() reshapeTensor:t withShape:IntArray(col) name:nil];
    int64_t newN = N + (N - 1) * pi;
    NSMutableArray<NSNumber*>* st = [NSMutableArray array];
    NSMutableArray<NSNumber*>* en = [NSMutableArray array];
    NSMutableArray<NSNumber*>* sr = [NSMutableArray array];
    for (int64_t k = 0; k < rank; ++k) { [st addObject:@0]; [en addObject:@(k == d ? newN : col[k])]; [sr addObject:@1]; }
    t = [L.graph() sliceTensor:t starts:st ends:en strides:sr name:nil];
    shape[d] = newN;
  }
  return t;
}

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_OPS_OPS_COMMON_H_
