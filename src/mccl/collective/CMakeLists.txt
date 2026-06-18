# mccl_collective — collective-ops layer wrapping libmccl. Standalone; links libmccl + Metal only.
#   cmake -S src/mccl/collective -B build/collective && cmake --build build/collective
#   Override the mccl checkout with -DMCCL_HOME=/path/to/mccl.

cmake_minimum_required(VERSION 3.16)
project(mccl_collective CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Where the user's mccl library lives (headers + prebuilt static lib).
if(NOT DEFINED MCCL_HOME)
  if(DEFINED ENV{MCCL_HOME})
    set(MCCL_HOME "$ENV{MCCL_HOME}")
  else()
    set(MCCL_HOME "$ENV{HOME}/mccl")
  endif()
endif()
message(STATUS "mccl_collective: MCCL_HOME = ${MCCL_HOME}")

set(MCCL_INCLUDE_DIR "${MCCL_HOME}/src/include")
set(MCCL_STATIC_LIB  "${MCCL_HOME}/build/libmccl.a")

if(NOT EXISTS "${MCCL_INCLUDE_DIR}/mccl.h")
  message(WARNING "mccl_collective: ${MCCL_INCLUDE_DIR}/mccl.h not found; "
                  "set -DMCCL_HOME to your mccl checkout.")
endif()

# --- the collective wrapper -------------------------------------------------
add_library(mccl_collective STATIC
  bootstrap.cc
  comm.cc
  collectives.cc
)

target_include_directories(mccl_collective
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}     # public headers
  PRIVATE
    ${MCCL_INCLUDE_DIR}             # libmccl headers (impl-only)
)

# Frameworks libmccl pulls in.
set(MCCL_FRAMEWORKS
  "-framework Metal"
  "-framework Foundation"
  "-framework IOKit"
  "-framework CoreFoundation"
)

target_link_libraries(mccl_collective
  PUBLIC
    ${MCCL_STATIC_LIB}
    ${MCCL_FRAMEWORKS}
)

# --- single-rank smoke test -------------------------------------------------
add_executable(mccl_collective_smoke smoke.cc)
target_link_libraries(mccl_collective_smoke PRIVATE mccl_collective)
# Run with MCCL_METAL_DIR=${MCCL_HOME}/src/device pointing at the kernels.
