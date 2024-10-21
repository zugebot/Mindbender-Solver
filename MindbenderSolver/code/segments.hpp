#pragma once

#include "MindbenderSolver/utils/processor.hpp"
#include <immintrin.h>


static u64 prime_func1(c_u64 b1, c_u64 b2) {
    return ((b1 << 4) + b1) ^ b2;
}


// check commits before 10/16/24 for previous impl.
static u64 getSegment2bits(c_u64 segment) {
    static constexpr u64 MASK_X0 = 0'111111'111111'111111;
    return _pext_u64(segment, MASK_X0);
}


// check commits before 10/16/24 for previous impl.
static u64 getSegment3bits(c_u64 segment) {
    static constexpr u64 MASK_CS = 0'003003'003003'003003;
    c_u64 o1 = ((segment >> 6) & MASK_CS) * 9 /* 0b1001 */
               |
               ((segment >> 3) & MASK_CS) * 3 /* 0b1001 */
               |
               segment & MASK_CS
            ;
    static constexpr u64 MASK_X23 = 0'037037'037037'037037;
    c_u64 x23 = _pext_u64(o1, MASK_X23);
    return x23;
}


// check commits before 10/16/24 for previous impl.
static u64 getSegment4bits(c_u64 segment) {
    static constexpr u64 MASK_X0 = 0'333333'333333'333333;
    return _pext_u64(segment, MASK_X0);
}