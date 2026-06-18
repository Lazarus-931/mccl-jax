#ifndef MCCL_JAX_SRC_JAM_OPS_OPS_COMMON_H_
#define MCCL_JAX_SRC_JAM_OPS_OPS_COMMON_H_

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/jam/lowering_internal.h"

namespace mccl_jax::jam {

inline MPSGraphTensor* A(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(0)); }
inline MPSGraphTensor* B(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(1)); }
inline MPSGraphTensor* C(Lowering& L, mlir::Operation* op) { return L.value(op->getOperand(2)); }

inline void Set(Lowering& L, mlir::Operation* op, MPSGraphTensor* t) {
  L.bind(op->getResult(0), t);
}

inline bool UnsignedIntOperand(mlir::Operation* op, unsigned i, MPSDataType& uty) {
  auto it = mlir::dyn_cast<mlir::IntegerType>(Lowering::ElementType(op->getOperand(i).getType()));
  if (!it || !it.isUnsigned()) return false;
  switch (it.getWidth()) {
    case 8:  uty = MPSDataTypeUInt8; break;
    case 16: uty = MPSDataTypeUInt16; break;
    default: uty = MPSDataTypeUInt32; break;
  }
  return true;
}

inline NSArray<NSNumber*>* ShapeArray(llvm::ArrayRef<int64_t> dims) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t d : dims) [a addObject:@(d)];
  if (a.count == 0) [a addObject:@1];
  return a;
}

inline NSArray<NSNumber*>* IntArray(llvm::ArrayRef<int64_t> v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(x)];
  return a;
}

inline llvm::ArrayRef<int64_t> OutShape(mlir::Operation* op, unsigned result = 0) {
  return mlir::cast<mlir::RankedTensorType>(op->getResult(result).getType()).getShape();
}

inline MPSGraphTensor* Reshaped(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> dims) {
  NSArray<NSNumber*>* want = ShapeArray(dims);
  NSArray<NSNumber*>* cur = t.shape;
  if (cur != nil && [cur isEqualToArray:want]) return t;
  return [L.graph() reshapeTensor:t withShape:want name:nil];
}

inline MPSGraphTensor* Casted(Lowering& L, MPSGraphTensor* t, MPSDataType dt) {
  if (t.dataType == dt) return t;
  return [L.graph() castTensor:t toType:dt name:nil];
}

inline MPSGraphTensor* Broadcasted(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> dims) {
  NSArray<NSNumber*>* want = ShapeArray(dims);
  NSArray<NSNumber*>* cur = t.shape;
  if (cur != nil && [cur isEqualToArray:want]) return t;
  return [L.graph() broadcastTensor:t toShape:want name:nil];
}

inline MPSGraphTensor* Transposed(Lowering& L, MPSGraphTensor* t, llvm::ArrayRef<int64_t> perm) {
  for (std::size_t i = 0; i < perm.size(); ++i)
    if (perm[i] != (int64_t)i) return [L.graph() transposeTensor:t permutation:IntArray(perm) name:nil];
  return t;
}

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

}

#endif
