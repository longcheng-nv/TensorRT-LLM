// Standalone-bench stub for tvm-ffi's env API. Only referenced by the unused
// DLDevice constructor path of host::LaunchKernel (we always pass a raw
// cudaStream_t); never called at runtime.
#pragma once
#include <cstdint>
static inline void* TVMFFIEnvGetStream(int32_t, int32_t) { return nullptr; }
