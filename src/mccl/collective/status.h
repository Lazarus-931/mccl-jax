#pragma once

#include <string>
#include <utility>

namespace mccl_collective {

struct Status {
  bool ok = true;
  std::string message;

  Status() = default;
  explicit Status(std::string msg) : ok(false), message(std::move(msg)) {}

  static Status Ok() { return Status(); }
  static Status Error(std::string msg) { return Status(std::move(msg)); }

  explicit operator bool() const { return ok; }
};

}
