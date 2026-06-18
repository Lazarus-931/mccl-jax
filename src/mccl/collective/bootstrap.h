#pragma once

#include <cstdint>
#include <string>

#include "status.h"

namespace mccl_collective {

struct BootstrapInfo {
  int rank = 0;
  int world_size = 1;
  std::string coordinator_ip = "127.0.0.1";
  uint16_t coordinator_port = 53700;
};

Status BootstrapFromEnv(BootstrapInfo* out);

}
