#include "bootstrap.h"

#include <cstdlib>
#include <string>

namespace mccl_collective {

namespace {

// Parse an integer env var. On absence -> default. On malformed value -> error.
Status ParseIntEnv(const char* key, int default_value, int* out) {
  const char* v = std::getenv(key);
  if (v == nullptr || v[0] == '\0') {
    *out = default_value;
    return Status::Ok();
  }
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') {
    return Status::Error(std::string("bootstrap: env var ") + key +
                         " is not an integer: \"" + v + "\"");
  }
  *out = static_cast<int>(parsed);
  return Status::Ok();
}

}  // namespace

Status BootstrapFromEnv(BootstrapInfo* out) {
  if (out == nullptr) return Status::Error("bootstrap: out is null");

  Status s = ParseIntEnv("MCCL_RANK", 0, &out->rank);
  if (!s) return s;

  s = ParseIntEnv("MCCL_WORLD_SIZE", 1, &out->world_size);
  if (!s) return s;

  if (out->world_size < 1) {
    return Status::Error("bootstrap: MCCL_WORLD_SIZE must be >= 1, got " +
                         std::to_string(out->world_size));
  }
  if (out->rank < 0 || out->rank >= out->world_size) {
    return Status::Error("bootstrap: MCCL_RANK " + std::to_string(out->rank) +
                         " out of range [0, " +
                         std::to_string(out->world_size) + ")");
  }

  if (const char* ip = std::getenv("MCCL_BOOTSTRAP_IP")) {
    if (ip[0] != '\0') out->coordinator_ip = ip;
  }

  int port = 0;
  s = ParseIntEnv("MCCL_BOOTSTRAP_PORT", 53700, &port);
  if (!s) return s;
  out->coordinator_port = static_cast<uint16_t>(port);

  return Status::Ok();
}

}  // namespace mccl_collective
