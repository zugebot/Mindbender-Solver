#pragma once

#include <cstdint>

// For MSVC, declare the intrinsic outside the function
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif

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
    // Fallback implementation using bitset
    if (x == 0) return 32;
    return 32 - std::bitset<32>(x).count();
#endif
#endif
}
