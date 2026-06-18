#pragma once

#include <cstddef>

#include "comm.h"
#include "status.h"

namespace mccl_collective {

// This module's own data-type / reduce-op vocabulary, mapped internally to libmccl's enums.
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

// Size in bytes of one element of `dt`.
std::size_t DTypeSize(DType dt);

// In-place, buffer-agnostic collectives (zero-copy over one device buffer). `count` is in
// elements, not bytes (conventions match mccl). Each returns a Status (no crash on error).
//   - AllReduce / Broadcast: count = elements in the whole buffer.
//   - AllGather: count = elements this rank contributes; recv holds count * n_ranks.
//   - ReduceScatter: count = elements this rank receives; send holds count * n_ranks.
//   - AllToAll: count = total elements (split evenly across ranks).

Status AllReduce(Comm& comm, void* data, std::size_t count, DType dt,
                 ReduceOp op);

Status AllGather(Comm& comm, void* data, std::size_t count, DType dt);

Status ReduceScatter(Comm& comm, void* data, std::size_t count, DType dt,
                     ReduceOp op);

Status Broadcast(Comm& comm, void* data, std::size_t count, DType dt, int root);

Status AllToAll(Comm& comm, void* data, std::size_t count, DType dt);

// Out-of-place forms (send != recv), used where the op changes shape. Counts follow mccl:
//   - AllReduce / Broadcast: count = elements (send and recv both hold count).
//   - AllGather: sendcount = elements this rank sends; recv holds sendcount * n_ranks.
//   - ReduceScatter: recvcount = elements this rank receives; send holds recvcount * n_ranks.
//   - AllToAll: count = elements exchanged per peer; send and recv each hold count * n_ranks.

Status AllReduce(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, ReduceOp op);

Status AllGather(Comm& comm, const void* send, void* recv, std::size_t sendcount, DType dt);

Status ReduceScatter(Comm& comm, const void* send, void* recv, std::size_t recvcount, DType dt, ReduceOp op);

Status Broadcast(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, int root);

Status AllToAll(Comm& comm, const void* send, void* recv, std::size_t count, DType dt);

// collective_permute: send this rank's buffer to `target` and receive into recv from `source`
// (pass -1 to skip the send or the receive). count = elements. Grouped to avoid deadlock.
Status Permute(Comm& comm, const void* send, void* recv, std::size_t count, DType dt,
               int target, int source);

}  // namespace mccl_collective
