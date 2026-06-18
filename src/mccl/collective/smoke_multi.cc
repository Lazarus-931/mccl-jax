#include <cstdio>
#include <memory>
#include <string>

#include "collectives.h"
#include "comm.h"
#include "mccl.h"

using mccl_collective::AllReduce;
using mccl_collective::Comm;
using mccl_collective::DType;
using mccl_collective::ReduceOp;

int main() {
  std::string err;
  std::unique_ptr<Comm> comm = Comm::FromEnv(&err);
  if (comm == nullptr) {
    std::printf("FAIL: Comm::FromEnv: %s\n", err.c_str());
    return 1;
  }
  const int rank = comm->rank(), n = comm->n_ranks();
  std::printf("[rank %d/%d] communicator up\n", rank, n);
  std::fflush(stdout);

  const int N = 1024;
  void* ptr = nullptr;
  if (mccl::mcclPageAlloc(N * sizeof(float), &ptr) != mccl::mcclSuccess || ptr == nullptr) {
    std::printf("[rank %d] FAIL: mcclPageAlloc\n", rank);
    return 1;
  }
  float* buf = static_cast<float*>(ptr);
  for (int i = 0; i < N; ++i) buf[i] = static_cast<float>(rank + 1);

  auto s = AllReduce(*comm, buf, N, DType::kFloat32, ReduceOp::kSum);
  if (!s) {
    std::printf("[rank %d] FAIL: AllReduce: %s\n", rank, s.message.c_str());
    mccl::mcclPageFree(ptr);
    return 1;
  }

  const float expected = static_cast<float>(n * (n + 1) / 2);
  for (int i = 0; i < N; ++i) {
    if (buf[i] != expected) {
      std::printf("[rank %d] FAIL: buf[%d] = %.1f, expected %.1f\n", rank, i, buf[i], expected);
      mccl::mcclPageFree(ptr);
      return 1;
    }
  }
  std::printf("[rank %d/%d] PASS: AllReduce sum = %.0f  (N(N+1)/2 over %d ranks)\n",
              rank, n, expected, n);
  mccl::mcclPageFree(ptr);
  return 0;
}
