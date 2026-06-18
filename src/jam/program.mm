#include "src/jam/jam.h"
#include "src/jam/program_impl.h"

namespace mccl_jax::jam {

CompiledProgram::CompiledProgram(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CompiledProgram::~CompiledProgram() = default;
CompiledProgram::CompiledProgram(CompiledProgram&&) noexcept = default;
CompiledProgram& CompiledProgram::operator=(CompiledProgram&&) noexcept = default;

const std::vector<IoSpec>& CompiledProgram::inputs() const { return impl_->input_specs; }
const std::vector<IoSpec>& CompiledProgram::outputs() const { return impl_->output_specs; }

}
