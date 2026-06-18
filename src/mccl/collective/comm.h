#pragma once

#include <memory>
#include <string>

#include "status.h"

namespace mccl {
struct mcclComm;
}

namespace mccl_collective {

class Comm {
 public:

  static std::unique_ptr<Comm> Create(int n_ranks, int rank, std::string* err);

  static std::unique_ptr<Comm> FromEnv(std::string* err);

  ~Comm();

  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;
  Comm(Comm&& other) noexcept;
  Comm& operator=(Comm&& other) noexcept;

  mccl::mcclComm* handle() const { return comm_; }

  int rank() const { return rank_; }
  int n_ranks() const { return n_ranks_; }

 private:
  Comm(mccl::mcclComm* comm, int n_ranks, int rank)
      : comm_(comm), n_ranks_(n_ranks), rank_(rank) {}

  mccl::mcclComm* comm_ = nullptr;
  int n_ranks_ = 0;
  int rank_ = 0;
};

}
