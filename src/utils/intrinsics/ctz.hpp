#pragma once

#include "src/utils/processor.hpp"   // __host__ / __device__ outside CUDA

#include <cstdint>

// For MSVC, declare the intrinsic outside the function
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
#endif

// Trailing zeros of a 64-bit value without intrinsics (64 for 0): halves the
// search range for the lowest set bit six times. Used when no compiler
// intrinsic is available; testable on any compiler.
__host__ __device__
        inline int my_ctz64_portable(uint64_t v) {
    if (v == 0) return 64;
    int n = 0;
    if ((v & 0xFFFFFFFFULL) == 0) { n += 32; v >>= 32; }
    if ((v & 0xFFFFULL) == 0) { n += 16; v >>= 16; }
    if ((v & 0xFFULL) == 0) { n += 8; v >>= 8; }
    if ((v & 0xFULL) == 0) { n += 4; v >>= 4; }
    if ((v & 0x3ULL) == 0) { n += 2; v >>= 2; }
    if ((v & 0x1ULL) == 0) { n += 1; }
    return n;
}

__host__ __device__
        inline int my_ctz64(uint64_t x) {
#ifdef __CUDA_ARCH__
    // Device code: __ffsll is 1-based, 0 for x == 0
    return x == 0 ? 64 : __ffsll(static_cast<long long>(x)) - 1;
#else
// Host code: Use compiler built-in or intrinsic
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang (undefined for 0, like __builtin_clz in my_clz)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    // MSVC
    unsigned long index = 0;
    return _BitScanForward64(&index, x) ? static_cast<int>(index) : 64;
#else
    // Fallback: no intrinsic available
    return my_ctz64_portable(x);
#endif
#endif
}
