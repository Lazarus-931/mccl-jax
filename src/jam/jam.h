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

// Cross-host collectives. jam splits the program at these and routes them (via a callback) to
// the collective layer (mccl); the plugin maps these to mcclAllReduce/etc.
enum class CollectiveOp {
  kAllReduce, kAllGather, kReduceScatter, kBroadcast, kAllToAll, kCollectivePermute
};
enum class ReduceKind { kSum, kProd, kMax, kMin, kAvg };

// Compiled artifact (pimpl): owns the MPSGraph + ordered I/O tensors. Internals in program_impl.h.
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
  std::unique_ptr<CompiledProgram> program;  // null on failure
  std::string error;                          // empty on success
};

// num_processes = cluster size (1 ⇒ single device: collectives lower as identity, no extra segments).
CompileResult Compile(const char* stablehlo_bytecode, std::size_t size, int num_processes = 1);

}  // namespace mccl_jax::jam

#endif  // MCCL_JAX_SRC_JAM_JAM_H_
