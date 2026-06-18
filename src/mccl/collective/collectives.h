#pragma once

#include <cstddef>

#include "comm.h"
#include "status.h"

namespace mccl_collective {

enum class DType {
  kInt8,
  kUint8,
  kInt32,
  kUint32,
  kInt64,
  kUint64,
  kFloat16,
  kBfloat16,
  kFloat32,
  kFloat64,
};

enum class ReduceOp {
  kSum,
  kProd,
  kMax,
  kMin,
  kAvg,
};

std::size_t DTypeSize(DType dt);

Status AllReduce(Comm& comm, void* data, std::size_t count, DType dt,
                 ReduceOp op);

Status AllGather(Comm& comm, void* data, std::size_t count, DType dt);

Status ReduceScatter(Comm& comm, void* data, std::size_t count, DType dt,
                     ReduceOp op);

Status Broadcast(Comm& comm, void* data, std::size_t count, DType dt, int root);

Status AllToAll(Comm& comm, void* data, std::size_t count, DType dt);

Status AllReduce(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, ReduceOp op);

Status AllGather(Comm& comm, const void* send, void* recv, std::size_t sendcount, DType dt);

Status ReduceScatter(Comm& comm, const void* send, void* recv, std::size_t recvcount, DType dt, ReduceOp op);

Status Broadcast(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, int root);

Status AllToAll(Comm& comm, const void* send, void* recv, std::size_t count, DType dt);

Status Permute(Comm& comm, const void* send, void* recv, std::size_t count, DType dt,
               int target, int source);

}
