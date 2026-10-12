#pragma once

#include "src/utils/processor.hpp"   // __host__ / __device__ outside CUDA

#include <cstdint>

// For MSVC, declare the intrinsic outside the function
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif

// Leading zeros of a 32-bit value without intrinsics (32 for 0): halves the
// search range for the highest set bit five times. Used when no compiler
// intrinsic is available; testable on any compiler.
__host__ __device__
        inline int my_clz_portable(int x) {
    uint32_t v = static_cast<uint32_t>(x);
    if (v == 0) return 32;
    int n = 0;
    if ((v & 0xFFFF0000u) == 0) { n += 16; v <<= 16; }
    if ((v & 0xFF000000u) == 0) { n += 8; v <<= 8; }
    if ((v & 0xF0000000u) == 0) { n += 4; v <<= 4; }
    if ((v & 0xC0000000u) == 0) { n += 2; v <<= 2; }
    if ((v & 0x80000000u) == 0) { n += 1; }
    return n;
}

__host__ __device__
        inline int my_clz(int x) {
#ifdef __CUDA_ARCH__
    // Device code: Use CUDA intrinsic
    return __clz(x);
#else
// Host code: Use compiler built-in or intrinsic
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang
    return __builtin_clz(x);
#elif defined(_MSC_VER)
    // MSVC
    unsigned long leading_zero = 0;
    if (_BitScanReverse(&leading_zero, x)) {
        return 31 - leading_zero;
    } else {
        // x is zero, leading zeros is 32
        return 32;
    }
#else
    // Fallback: no intrinsic available
    return my_clz_portable(x);
#endif
#endif
}
