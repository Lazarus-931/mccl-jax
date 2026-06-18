#ifndef MCCL_JAX_SRC_JAM_JAM_H_
#define MCCL_JAX_SRC_JAM_JAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mccl_jax::jam {

enum class DType {
  kInvalid, kPred,
  kI8, kI16, kI32, kI64,
  kU8, kU16, kU32, kU64,
  kF16, kBF16, kF32, kF64,
};

struct IoSpec {
  std::vector<int64_t> dims;
  DType dtype = DType::kInvalid;
};

enum class CollectiveOp {
  kAllReduce, kAllGather, kReduceScatter, kBroadcast, kAllToAll, kCollectivePermute
};
enum class ReduceKind { kSum, kProd, kMax, kMin, kAvg };

class CompiledProgram {
 public:
  struct Impl;
  explicit CompiledProgram(std::unique_ptr<Impl> impl);
  ~CompiledProgram();
  CompiledProgram(CompiledProgram&&) noexcept;
  CompiledProgram& operator=(CompiledProgram&&) noexcept;

  const std::vector<IoSpec>& inputs() const;
  const std::vector<IoSpec>& outputs() const;
  Impl* impl() const { return impl_.get(); }

 private:
  std::unique_ptr<Impl> impl_;
};

struct CompileResult {
  std::unique_ptr<CompiledProgram> program;
  std::string error;
};

CompileResult Compile(const char* stablehlo_bytecode, std::size_t size, int num_processes = 1);

}

#endif
