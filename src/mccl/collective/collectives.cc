#include "collectives.h"

#include <string>

// libmccl public API.
#include "mccl.h"

namespace mccl_collective {

namespace {

// Map this module's DType -> mccl::mcclDataType.
mccl::mcclDataType ToMccl(DType dt) {
  switch (dt) {
    case DType::kInt8:     return mccl::mcclInt8;
    case DType::kUint8:    return mccl::mcclUint8;
    case DType::kInt32:    return mccl::mcclInt32;
    case DType::kUint32:   return mccl::mcclUint32;
    case DType::kInt64:    return mccl::mcclInt64;
    case DType::kUint64:   return mccl::mcclUint64;
    case DType::kFloat16:  return mccl::mcclFloat16;
    case DType::kBfloat16: return mccl::mcclBfloat16;
    case DType::kFloat32:  return mccl::mcclFloat32;
    case DType::kFloat64:  return mccl::mcclFloat64;
  }
  return mccl::mcclFloat32;  // unreachable; keeps the compiler happy
}

// Map this module's ReduceOp -> mccl::mcclRedOp.
mccl::mcclRedOp ToMccl(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum:  return mccl::mcclSum;
    case ReduceOp::kProd: return mccl::mcclProd;
    case ReduceOp::kMax:  return mccl::mcclMax;
    case ReduceOp::kMin:  return mccl::mcclMin;
    case ReduceOp::kAvg:  return mccl::mcclAvg;
  }
  return mccl::mcclSum;  // unreachable
}

// Turn an mcclResult into a Status (ok on mcclSuccess).
Status FromResult(const char* op, mccl::mcclResult r) {
  if (r == mccl::mcclSuccess) return Status::Ok();
  return Status::Error(std::string(op) + ": " + mccl::mcclResultStr(r) +
                       " (code " + std::to_string(static_cast<int>(r)) + ")");
}

// Common guard shared by all collectives.
Status CheckArgs(const char* op, Comm& comm, const void* data,
                  std::size_t count) {
  if (comm.handle() == nullptr) {
    return Status::Error(std::string(op) + ": comm has no handle");
  }
  if (data == nullptr && count != 0) {
    return Status::Error(std::string(op) + ": data is null");
  }
  return Status::Ok();
}

}  // namespace

std::size_t DTypeSize(DType dt) {
  return mccl::mcclDataSize(ToMccl(dt));
}

Status AllReduce(Comm& comm, void* data, std::size_t count, DType dt,
                 ReduceOp op) {
  Status s = CheckArgs("AllReduce", comm, data, count);
  if (!s) return s;
  // In place: sendbuff == recvbuff == data.
  return FromResult("AllReduce",
                    mccl::mcclAllReduce(comm.handle(), data, data, count,
                                        ToMccl(dt), ToMccl(op)));
}

Status AllGather(Comm& comm, void* data, std::size_t count, DType dt) {
  Status s = CheckArgs("AllGather", comm, data, count);
  if (!s) return s;
  return FromResult("AllGather",
                    mccl::mcclAllGather(comm.handle(), data, data, count,
                                        ToMccl(dt)));
}

Status ReduceScatter(Comm& comm, void* data, std::size_t count, DType dt,
                     ReduceOp op) {
  Status s = CheckArgs("ReduceScatter", comm, data, count);
  if (!s) return s;
  return FromResult("ReduceScatter",
                    mccl::mcclReduceScatter(comm.handle(), data, data, count,
                                            ToMccl(dt), ToMccl(op)));
}

Status Broadcast(Comm& comm, void* data, std::size_t count, DType dt,
                 int root) {
  Status s = CheckArgs("Broadcast", comm, data, count);
  if (!s) return s;
  return FromResult("Broadcast",
                    mccl::mcclBroadcast(comm.handle(), data, data, count,
                                        ToMccl(dt), root));
}

Status AllToAll(Comm& comm, void* data, std::size_t count, DType dt) {
  Status s = CheckArgs("AllToAll", comm, data, count);
  if (!s) return s;
  return FromResult("AllToAll",
                    mccl::mcclAllToAll(comm.handle(), data, data, count,
                                       ToMccl(dt)));
}

// ---- out-of-place forms (send != recv) ----
namespace {
Status CheckArgs2(const char* op, Comm& comm, const void* send, const void* recv) {
  if (comm.handle() == nullptr) return Status::Error(std::string(op) + ": comm has no handle");
  if (send == nullptr || recv == nullptr) return Status::Error(std::string(op) + ": null buffer");
  return Status::Ok();
}
}  // namespace

Status AllReduce(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, ReduceOp op) {
  Status s = CheckArgs2("AllReduce", comm, send, recv);
  if (!s) return s;
  return FromResult("AllReduce",
                    mccl::mcclAllReduce(comm.handle(), send, recv, count, ToMccl(dt), ToMccl(op)));
}

Status AllGather(Comm& comm, const void* send, void* recv, std::size_t sendcount, DType dt) {
  Status s = CheckArgs2("AllGather", comm, send, recv);
  if (!s) return s;
  return FromResult("AllGather",
                    mccl::mcclAllGather(comm.handle(), send, recv, sendcount, ToMccl(dt)));
}

Status ReduceScatter(Comm& comm, const void* send, void* recv, std::size_t recvcount, DType dt, ReduceOp op) {
  Status s = CheckArgs2("ReduceScatter", comm, send, recv);
  if (!s) return s;
  return FromResult("ReduceScatter",
                    mccl::mcclReduceScatter(comm.handle(), send, recv, recvcount, ToMccl(dt), ToMccl(op)));
}

Status Broadcast(Comm& comm, const void* send, void* recv, std::size_t count, DType dt, int root) {
  Status s = CheckArgs2("Broadcast", comm, send, recv);
  if (!s) return s;
  return FromResult("Broadcast",
                    mccl::mcclBroadcast(comm.handle(), send, recv, count, ToMccl(dt), root));
}

Status AllToAll(Comm& comm, const void* send, void* recv, std::size_t count, DType dt) {
  Status s = CheckArgs2("AllToAll", comm, send, recv);
  if (!s) return s;
  return FromResult("AllToAll",
                    mccl::mcclAllToAll(comm.handle(), send, recv, count, ToMccl(dt)));
}

Status Permute(Comm& comm, const void* send, void* recv, std::size_t count, DType dt,
               int target, int source) {
  if (comm.handle() == nullptr) return Status::Error("Permute: comm has no handle");
  // Group the send + recv so peer ranks' matching calls rendezvous instead of deadlocking.
  mccl::mcclGroupStart();
  if (target >= 0) mccl::mcclSend(comm.handle(), send, count, ToMccl(dt), target);
  if (source >= 0) mccl::mcclRecv(comm.handle(), recv, count, ToMccl(dt), source);
  return FromResult("Permute", mccl::mcclGroupEnd());
}

}  // namespace mccl_collective
