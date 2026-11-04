#pragma once
// src/gpu/vulkan/mbvk.h
//
// What mbvk.dll exports beyond the shared GPU interface (src/gpu/mbgpu.h, which it
// implements in full). The solver looks this up with GetProcAddress, so the CUDA module
// does not need it.

#include "src/gpu/mbgpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Every Vulkan device, one line each: index, name, type, driver, memory, whether it can run
// the kernels (and why not), and "auto" on the ones mbgpu_open("auto") would take. Only asks
// the driver: no device is opened and no memory is allocated. Returns 0 on success.
MBGPU_API int mbvk_list_devices(char* out, int outLen);

#ifdef __cplusplus
}
#endif
