// Minimal stand-ins for the TRT-LLM common headers the GVR Top-K kernel
// references — just enough symbols to build heuristicTopKDecode.cu
// standalone with torch.utils.cpp_extension. NOT a full reimplementation
// of `tensorrt_llm/common/*.h`.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// `tensorrt_llm/common/config.h` defines this pair of macros. We open the
// kernel into a local namespace so multiple copies of this extension can
// coexist without ODR conflicts with a real libtensorrt_llm.so in the
// same process.
#define TRTLLM_NAMESPACE_BEGIN namespace tensorrt_llm_gvr_local {
#define TRTLLM_NAMESPACE_END   } // namespace tensorrt_llm_gvr_local

// `tensorrt_llm/common/assert.h` macro surface (subset used by GVR kernel).
#define TLLM_CHECK_WITH_INFO(cond, fmt, ...)                                              \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            char _msg[512];                                                               \
            std::snprintf(_msg, sizeof(_msg), fmt, ##__VA_ARGS__);                        \
            throw std::runtime_error(std::string("[gvr_kernel] check failed: ") + _msg); \
        }                                                                                 \
    } while (0)

#define TLLM_CHECK(cond) TLLM_CHECK_WITH_INFO((cond), "%s", #cond)

#define TLLM_THROW(msg) throw std::runtime_error(std::string("[gvr_kernel] ") + (msg))

// `tensorrt_llm/common/envUtils.h::getEnvEnablePDL()` — read TRTLLM_ENABLE_PDL.
// PDL (Programmatic Stream Serialization) is a kernel launch attribute; this
// stub keeps behavior parity with the in-tree default (on unless explicitly
// disabled).
namespace tensorrt_llm_gvr_local {
namespace common {
inline bool getEnvEnablePDL() {
    char const* env = std::getenv("TRTLLM_ENABLE_PDL");
    if (env == nullptr) return true;
    return !(env[0] == '0' && env[1] == '\0');
}
} // namespace common
} // namespace tensorrt_llm_gvr_local

// Compatibility alias — heuristicTopKDecode.cu writes
// `tensorrt_llm::common::getEnvEnablePDL()`. Provide that symbol path.
namespace tensorrt_llm {
namespace common {
inline bool getEnvEnablePDL() { return tensorrt_llm_gvr_local::common::getEnvEnablePDL(); }
} // namespace common
} // namespace tensorrt_llm
