#pragma once

#include "utils/intrinsics/pext_u64.hpp"
#include "utils/processor.hpp"


MU HD static u64 prime_func1(C u64 b1, C u64 b2) {
    return ((b1 << 4) + b1) ^ b2;
}


// check commits before 10/16/24 for previous impl.
MU HD static u64 getSegment2bits(C u64 segment) {
#ifndef __CUDA_ARCH__
    static constexpr u64 MASK_X0 = 0'111111'111111'111111;
    return my_pext_u64(segment, MASK_X0);
#else
    static constexpr u64 MASK_A1 = 0'101010'101010'101010;
    static constexpr u64 MASK_B1 = MASK_A1 >> 3;
    static constexpr u64 MASK_A2 = 0'030000'030000'030000;
    static constexpr u64 MASK_B2 = MASK_A2 >> 6;
    static constexpr u64 MASK_C2 = MASK_A2 >> 12;
    static constexpr u64 MASK_A3 = 0'000077'000000'000000;
    static constexpr u64 MASK_B3 = MASK_A3 >> 18;
    static constexpr u64 MASK_C3 = MASK_A3 >> 36;
    C u64 o1 = (segment & MASK_A1) >> 2 | segment & MASK_B1;
    C u64 o2 = (o1 & MASK_A2) >> 8 | (o1 & MASK_B2) >> 4 | o1 & MASK_C2;
    C u64 o3 = (o2 & MASK_A3) >> 24 | (o2 & MASK_B3) >> 12 | o2 & MASK_C3;
    return o3;
#endif
}


// check commits before 10/16/24 for previous impl.
MU HD static u64 getSegment3bits(C u64 segment) {
    static constexpr u64 MASK_CS = 0'003003'003003'003003;
    C u64 o1 = (segment >> 6 & MASK_CS) * 9 /* 0b1001 */
               |
               (segment >> 3 & MASK_CS) * 3 /* 0b1001 */
               |
               segment & MASK_CS
            ;
#ifndef __CUDA_ARCH__
    static constexpr u64 MASK_X23 = 0'037037'037037'037037;
    C u64 x23 = my_pext_u64(o1, MASK_X23);
    return x23;
#else
    static constexpr u64 MASK_A1 = 0'037000'037000'037000;
    static constexpr u64 MASK_B1 = MASK_A1 >> 9;
    static constexpr u64 MASK_A2 = 0'001777'000000'000000;
    static constexpr u64 MASK_B2 = MASK_A2 >> 18;
    static constexpr u64 MASK_C2 = MASK_A2 >> 36;
    C u64 o2 = (o1 & MASK_A1) >> 4 | o1 & MASK_B1;
    C u64 o3 = (o2 & MASK_A2) >> 16 | (o2 & MASK_B2) >> 8 | o2 & MASK_C2;
    return o3;
#endif
}


// check commits before 10/16/24 for previous impl.
MU HD static u64 getSegment4bits(C u64 segment) {
#ifndef __CUDA_ARCH__
    static constexpr u64 MASK_X0 = 0'333333'333333'333333;
    return my_pext_u64(segment, MASK_X0);
#else
    static constexpr u64 MASK_A1 = 0'303030'303030'303030;
    static constexpr u64 MASK_B1 = MASK_A1 >> 3;
    static constexpr u64 MASK_A2 = 0'170000'170000'170000;
    static constexpr u64 MASK_B2 = MASK_A2 >> 6;
    static constexpr u64 MASK_C2 = MASK_A2 >> 12;
    static constexpr u64 MASK_A3 = 0'007777'000000'000000;
    static constexpr u64 MASK_B3 = MASK_A3 >> 18;
    static constexpr u64 MASK_C3 = MASK_A3 >> 36;
    C u64 o1 = (segment & MASK_A1) >> 1 | segment & MASK_B1;
    C u64 o2 = (o1 & MASK_A2) >> 4 | (o1 & MASK_B2) >> 2 | o1 & MASK_C2;
    C u64 o3 = (o2 & MASK_A3) >> 12 | (o2 & MASK_B3) >> 6 | o2 & MASK_C3;
    return o3;
#endif
}