#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "collectives.h"
#include "comm.h"

using mccl_collective::AllReduce;
using mccl_collective::Comm;
using mccl_collective::DType;
using mccl_collective::ReduceOp;

int main() {
  std::string err;
  std::unique_ptr<Comm> comm = Comm::Create(1, 0, &err);
  if (comm == nullptr) {
    std::printf("FAIL: Comm::Create(1, 0): %s\n", err.c_str());
    return 1;
  }
  std::printf("Comm created: rank=%d n_ranks=%d\n", comm->rank(),
              comm->n_ranks());

  std::vector<float> buf = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expected = buf;

  auto s = AllReduce(*comm, buf.data(), buf.size(), DType::kFloat32,
                     ReduceOp::kSum);
  if (!s) {
    std::printf("FAIL: AllReduce: %s\n", s.message.c_str());
    return 1;
  }

  for (std::size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] != expected[i]) {
      std::printf("FAIL: buf[%zu] = %f, expected %f\n", i, buf[i],
                  expected[i]);
      return 1;
    }
  }

  std::printf("PASS: single-rank in-place AllReduce is identity\n");
  return 0;
}
