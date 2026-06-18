# jam — StableHLO → MPSGraph compiler. Needs LLVM/MLIR + openxla/stablehlo + Metal/MPSGraph.
#   cmake ... -DMLIR_DIR=<llvm>/lib/cmake/mlir -DLLVM_DIR=<llvm>/lib/cmake/llvm \
#             -DSTABLEHLO_SOURCE_DIR=<stablehlo> -DSTABLEHLO_BUILD_DIR=<stablehlo>/build

find_package(MLIR REQUIRED CONFIG)

# StableHLO source + build trees (defaults match the build host layout: ~/stablehlo).
set(STABLEHLO_SOURCE_DIR "$ENV{HOME}/stablehlo" CACHE PATH "openxla/stablehlo source tree")
set(STABLEHLO_BUILD_DIR "${STABLEHLO_SOURCE_DIR}/build" CACHE PATH "openxla/stablehlo build tree")

add_library(jam STATIC
  jam.cc            # Compile(): parse + lower
  parse.cc          # MLIRContext + deserializePortableArtifact
  lower.mm          # registry + Lowering + module walk → MPSGraph
  program.mm        # CompiledProgram methods
  ops/elementwise.mm
  ops/unary.mm
  ops/binary.mm
  ops/reduce.mm
  ops/shape.mm
  ops/linalg.mm
  ops/gather_scatter.mm
  ops/structural.mm
  ops/control_flow.mm
  ops/rng.mm
  ops/norm.mm
  ops/dynamic.mm
  ops/collectives.mm
  ops/hard_limit.mm
)

target_include_directories(jam PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${STABLEHLO_SOURCE_DIR}      # "stablehlo/dialect/..." headers
  ${STABLEHLO_BUILD_DIR}       # generated *.h.inc / *.cpp.inc
)

target_compile_options(jam PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)

# StableHLO static archives, ordered high-level → low-level for the static linker.
set(STABLEHLO_LIBS
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloSerialization.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloPasses.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloLinalgTransforms.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloOptimizationPasses.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloPassUtils.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloRegister.a
  ${STABLEHLO_BUILD_DIR}/lib/libChloOps.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloOps.a
  ${STABLEHLO_BUILD_DIR}/lib/libVhloOps.a
  ${STABLEHLO_BUILD_DIR}/lib/libVhloTypes.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloTypeInference.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloTypeConversion.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloAssemblyFormat.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloBroadcastUtils.a
  ${STABLEHLO_BUILD_DIR}/lib/libStablehloBase.a
  ${STABLEHLO_BUILD_DIR}/lib/libVersion.a
)

target_link_libraries(jam PRIVATE
  ${STABLEHLO_LIBS}
  MLIR    # aggregate libMLIR.dylib
  LLVM    # aggregate libLLVM.dylib
  "-framework Metal" "-framework MetalPerformanceShadersGraph" "-framework Foundation"
)

# jam_check — standalone end-to-end validation harness. Build-host only; off by default.
option(JAM_BUILD_CHECK "Build the jam_check end-to-end validation harness" OFF)
if(JAM_BUILD_CHECK)
  add_executable(jam_check jam_check.mm)
  target_include_directories(jam_check PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(jam_check PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  # -force_load so statically-registered op handlers aren't dead-stripped.
  target_link_libraries(jam_check PRIVATE
    "-Wl,-force_load,$<TARGET_FILE:jam>" jam
    "-framework Metal" "-framework MetalPerformanceShadersGraph" "-framework Foundation"
  )
endif()
