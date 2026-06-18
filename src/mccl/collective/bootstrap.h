#pragma once

#include <cstdint>
#include <string>

#include "status.h"

namespace mccl_collective {

// Cluster membership + coordinator endpoint for a single rank (filled by BootstrapFromEnv).
struct BootstrapInfo {
  int rank = 0;             // MCCL_RANK
  int world_size = 1;       // MCCL_WORLD_SIZE
  std::string coordinator_ip = "127.0.0.1";  // MCCL_BOOTSTRAP_IP
  uint16_t coordinator_port = 53700;         // MCCL_BOOTSTRAP_PORT
};

// Populate `out` from MCCL_RANK / MCCL_WORLD_SIZE / MCCL_BOOTSTRAP_IP / MCCL_BOOTSTRAP_PORT.
// Returns an error only if a present env var is malformed; missing vars use the defaults above.
Status BootstrapFromEnv(BootstrapInfo* out);

}  // namespace mccl_collective
