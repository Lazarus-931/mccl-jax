#include "src/jam/parse.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mccl_jax::jam {

std::unique_ptr<mlir::MLIRContext> MakeContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
                  mlir::func::FuncDialect>();
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);
  ctx->loadAllAvailableDialects();
  // Pass names must be registered before PassManager::parse() can resolve them.
  static bool passesRegistered = false;
  if (!passesRegistered) {
    mlir::stablehlo::registerPasses();
    mlir::stablehlo::registerOptimizationPasses();
    passesRegistered = true;
  }
  return ctx;
}

// Best-effort normalize a StableHLO module into jam's supported core; lowering re-validates.
static void Normalize(mlir::ModuleOp module, mlir::MLIRContext& ctx) {
  // Ordered so each pass feeds the next; func.func-nested passes wrapped per their anchor op.
  static const char* kPipeline =
      "builtin.module("
      "func.func(stablehlo-legalize-composite-to-call),"
      "func.func(stablehlo-legalize-deprecated-ops{fail-on-unused=false}),"
      "func.func(chlo-legalize-to-stablehlo),"
      "stablehlo-refine-shapes,"
      "func.func(stablehlo-canonicalize-dynamism),"
      "func.func(stablehlo-aggressive-simplification)"
      ")";

  mlir::PassManager pm(&ctx);
  if (mlir::failed(mlir::parsePassPipeline(kPipeline, pm))) return;
  // Swallow pass-internal diagnostics; a pass declining to apply is not a jam error.
  mlir::ScopedDiagnosticHandler swallow(
      &ctx, [](mlir::Diagnostic&) { return mlir::success(); });
  (void)pm.run(module);  // best-effort
}

mlir::OwningOpRef<mlir::ModuleOp> Parse(const char* bytecode, std::size_t size,
                                        mlir::MLIRContext& ctx, std::string& error) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::stablehlo::deserializePortableArtifact(llvm::StringRef(bytecode, size), &ctx);
  if (!module) {
    error = "jam: failed to deserialize StableHLO portable artifact";
    return module;
  }
  Normalize(module.get(), ctx);
  return module;
}

}  // namespace mccl_jax::jam
