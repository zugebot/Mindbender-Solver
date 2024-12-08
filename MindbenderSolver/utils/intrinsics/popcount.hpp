#pragma once

#include <cstdint>

// For MSVC, declare the intrinsic outside the function
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(__popcnt64)
#endif

__host__ __device__ inline int my_popcount(uint64_t x) {
#ifdef __CUDA_ARCH__
    // Device code: Use CUDA intrinsic
    return __popcll(x);
#else
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    // MSVC
    return __popcnt64(x);
#else
    // Fallback implementation
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
#endif
}
