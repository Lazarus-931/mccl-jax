#pragma once

#include <memory>
#include <string>

#include "status.h"

// Forward-declare the mccl comm so this header doesn't pull in libmccl's headers.
namespace mccl {
struct mcclComm;
}

namespace mccl_collective {

// RAII wrapper around an mccl communicator. Construction does the mccl rendezvous; move-only.
class Comm {
 public:
  // Build a communicator for `rank` of `n_ranks`. Returns nullptr + writes `err` on failure.
  static std::unique_ptr<Comm> Create(int n_ranks, int rank, std::string* err);

  // Read rank/world_size from the environment (via BootstrapFromEnv) and Create.
  static std::unique_ptr<Comm> FromEnv(std::string* err);

  ~Comm();

  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;
  Comm(Comm&& other) noexcept;
  Comm& operator=(Comm&& other) noexcept;

  // Underlying mccl handle; consumed by the collectives in collectives.h.
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

}  // namespace mccl_collective
