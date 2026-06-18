#include "comm.h"

#include <string>
#include <utility>

#include "bootstrap.h"

// libmccl public API.
#include "mccl.h"

namespace mccl_collective {

namespace {

// Append the mccl error string for `r` to `*err` (when err is non-null).
void SetErr(std::string* err, const std::string& context, mccl::mcclResult r) {
  if (err == nullptr) return;
  *err = context + ": " + mccl::mcclResultStr(r) + " (code " +
         std::to_string(static_cast<int>(r)) + ")";
}

}  // namespace

std::unique_ptr<Comm> Comm::Create(int n_ranks, int rank, std::string* err) {
  if (n_ranks < 1 || rank < 0 || rank >= n_ranks) {
    if (err != nullptr) {
      *err = "Comm::Create: invalid (n_ranks=" + std::to_string(n_ranks) +
             ", rank=" + std::to_string(rank) + ")";
    }
    return nullptr;
  }

  // mcclGetUniqueId packs MCCL_BOOTSTRAP_IP / MCCL_BOOTSTRAP_PORT from the env.
  mccl::mcclUniqueId id{};
  mccl::mcclResult rc = mccl::mcclGetUniqueId(&id);
  if (rc != mccl::mcclSuccess) {
    SetErr(err, "Comm::Create: mcclGetUniqueId failed", rc);
    return nullptr;
  }

  mccl::mcclComm* comm = nullptr;
  rc = mccl::mcclCommInitRank(&comm, n_ranks, id, rank);
  if (rc != mccl::mcclSuccess || comm == nullptr) {
    SetErr(err, "Comm::Create: mcclCommInitRank failed", rc);
    if (comm != nullptr) mccl::mcclCommDestroy(comm);
    return nullptr;
  }

  // std::make_unique can't reach the private ctor; construct directly.
  return std::unique_ptr<Comm>(new Comm(comm, n_ranks, rank));
}

std::unique_ptr<Comm> Comm::FromEnv(std::string* err) {
  BootstrapInfo info;
  Status s = BootstrapFromEnv(&info);
  if (!s) {
    if (err != nullptr) *err = s.message;
    return nullptr;
  }
  return Create(info.world_size, info.rank, err);
}

Comm::~Comm() {
  if (comm_ != nullptr) {
    mccl::mcclCommDestroy(comm_);
    comm_ = nullptr;
  }
}

Comm::Comm(Comm&& other) noexcept
    : comm_(other.comm_), n_ranks_(other.n_ranks_), rank_(other.rank_) {
  other.comm_ = nullptr;
}

Comm& Comm::operator=(Comm&& other) noexcept {
  if (this != &other) {
    if (comm_ != nullptr) mccl::mcclCommDestroy(comm_);
    comm_ = other.comm_;
    n_ranks_ = other.n_ranks_;
    rank_ = other.rank_;
    other.comm_ = nullptr;
  }
  return *this;
}

}  // namespace mccl_collective
