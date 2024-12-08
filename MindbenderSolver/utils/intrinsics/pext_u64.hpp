#pragma once

#include "MindbenderSolver/utils/processor.hpp"


#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif


MU HD inline u64 my_pext_u64(u64 src, u64 mask) {
#ifdef __CUDA_ARCH__
    u64 result = 0;
    u64 bit_position = 0;

    while (mask != 0) {
        u64 lowest_bit = mask & -mask; // Isolate the lowest set bit
        if (src & lowest_bit) {
            result |= (u64(1) << bit_position);
        }
        mask &= mask - 1; // Clear the lowest set bit in mask
        ++bit_position;
    }

    return result;

#else
    // Host implementation using BMI2 intrinsic
    return _pext_u64(src, mask);
#endif
}