cmake_minimum_required(VERSION 3.20)
project(mccl_jax LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export only GetPjrtApi; hide everything else.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(pjrt_metal SHARED
  src/pjrt/api/pjrt_c_api.cc
  src/execution/metal_buffer.mm
)

# Include roots: repo root and third_party.
target_include_directories(pjrt_metal PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
)

target_compile_options(pjrt_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(pjrt_metal PRIVATE "-framework Metal" "-framework Foundation")

# jam needs LLVM/MLIR + stablehlo; only built when MLIR_DIR is passed, else compile/execute stub out.
if(MLIR_DIR)
  add_subdirectory(src/jam)
  target_sources(pjrt_metal PRIVATE src/jam/jam_run.mm)
  target_link_libraries(pjrt_metal PRIVATE jam "-framework MetalPerformanceShadersGraph")
  target_compile_definitions(pjrt_metal PRIVATE MCCL_JAX_WITH_JAM)

  # mccl collective layer (the multi-host all_reduce backend) compiled directly into the plugin.
  if(NOT DEFINED MCCL_HOME)
    if(DEFINED ENV{MCCL_HOME})
      set(MCCL_HOME "$ENV{MCCL_HOME}")
    else()
      set(MCCL_HOME "$ENV{HOME}/mccl")
    endif()
  endif()
  message(STATUS "pjrt_metal: MCCL_HOME = ${MCCL_HOME}")
  target_sources(pjrt_metal PRIVATE
    src/mccl/collective/bootstrap.cc
    src/mccl/collective/comm.cc
    src/mccl/collective/collectives.cc
  )
  # The collective *.cc include libmccl headers privately; add the mccl include dir on those files.
  set_source_files_properties(
    src/mccl/collective/bootstrap.cc
    src/mccl/collective/comm.cc
    src/mccl/collective/collectives.cc
    PROPERTIES INCLUDE_DIRECTORIES "${MCCL_HOME}/src/include"
  )
  target_link_libraries(pjrt_metal PRIVATE
    "${MCCL_HOME}/build/libmccl.a"
    "-framework IOKit" "-framework CoreFoundation"
  )

  # C-API end-to-end harness. -DPJRT_BUILD_E2E=ON.
  if(PJRT_BUILD_E2E)
    add_executable(pjrt_e2e src/pjrt/pjrt_e2e.cc)
    target_include_directories(pjrt_e2e PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/third_party)
    target_link_libraries(pjrt_e2e PRIVATE pjrt_metal)
  endif()
endif()

set_target_properties(pjrt_metal PROPERTIES
  OUTPUT_NAME pjrt_metal
  PREFIX "lib"
  SUFFIX ".dylib"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)
