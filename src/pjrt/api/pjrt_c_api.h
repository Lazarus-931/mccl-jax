#ifndef MCCL_JAX_SRC_PJRT_API_PJRT_C_API_H_
#define MCCL_JAX_SRC_PJRT_API_PJRT_C_API_H_

#include "xla/pjrt/c/pjrt_c_api.h"

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi();

#endif
