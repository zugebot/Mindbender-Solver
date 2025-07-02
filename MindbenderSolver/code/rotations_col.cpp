#include "rotations.hpp"

#include <map>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


static constexpr u64 C_MASK_0 = 0'700000'700000'700000;
static constexpr u64 C_MASK_1 = 0'070000'070000'070000;
static constexpr u64 C_MASK_2 = 0'007000'007000'007000;
static constexpr u64 C_MASK_3 = 0'000700'000700'000700;
static constexpr u64 C_MASK_4 = 0'000070'000070'000070;
static constexpr u64 C_MASK_5 = 0'000007'000007'000007;

#define var1var2(mask)              \
    C u64 var1 = board.b1 & (mask); \
    C u64 var2 = board.b2 & (mask)

PERM_MACRO(C01) {
    var1var2(C_MASK_0);
    board.b1 = board.b1 & ~C_MASK_0 | (var2 << 36 | var1 >> 18) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var1 << 36 | var2 >> 18) & C_MASK_0;
}
PERM_MACRO(C02) {
    var1var2(C_MASK_0);
    board.b1 = board.b1 & ~C_MASK_0 | (var2 << 18 | var1 >> 36) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var1 << 18 | var2 >> 36) & C_MASK_0;
}
PERM_MACRO(C03) {
    var1var2(C_MASK_0);
    board.b1 = board.b1 & ~C_MASK_0 | var2;
    board.b2 = board.b2 & ~C_MASK_0 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_0 | (var2 << 0 | var1 >> 0) & C_MASK_0;
    // board.b2 = board.b2 & ~C_MASK_0 | (var1 << 0 | var2 >> 0) & C_MASK_0;
    // or
    // board.b1 = board.b1 & ~C_MASK_0 | (var1 << 0 | var2 >> 0) & C_MASK_0;
    // board.b2 = board.b2 & ~C_MASK_0 | (var2 << 0 | var1 >> 0) & C_MASK_0;
}
PERM_MACRO(C04) {
    var1var2(C_MASK_0);
    board.b1 = board.b1 & ~C_MASK_0 | (var1 << 36 | var2 >> 18) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 << 36 | var1 >> 18) & C_MASK_0;
}
PERM_MACRO(C05) {
    var1var2(C_MASK_0);
    board.b1 = board.b1 & ~C_MASK_0 | (var1 << 18 | var2 >> 36) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 << 18 | var1 >> 36) & C_MASK_0;
}


PERM_MACRO(C11) {
    var1var2(C_MASK_1);
    board.b1 = board.b1 & ~C_MASK_1 | (var2 << 36 | var1 >> 18) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var1 << 36 | var2 >> 18) & C_MASK_1;
}
PERM_MACRO(C12) {
    var1var2(C_MASK_1);
    board.b1 = board.b1 & ~C_MASK_1 | (var2 << 18 | var1 >> 36) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var1 << 18 | var2 >> 36) & C_MASK_1;
}
PERM_MACRO(C13) {
    var1var2(C_MASK_1);
    board.b1 = board.b1 & ~C_MASK_1 | var2;
    board.b2 = board.b2 & ~C_MASK_1 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_1 | (var2 << 0 | var1 >> 0) & C_MASK_1;
    // board.b2 = board.b2 & ~C_MASK_1 | (var1 << 0 | var2 >> 0) & C_MASK_1;
    // or
    // board.b1 = board.b1 & ~C_MASK_1 | (var1 << 0 | var2 >> 0) & C_MASK_1;
    // board.b2 = board.b2 & ~C_MASK_1 | (var2 << 0 | var1 >> 0) & C_MASK_1;
}
PERM_MACRO(C14) {
    var1var2(C_MASK_1);
    board.b1 = board.b1 & ~C_MASK_1 | (var1 << 36 | var2 >> 18) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 << 36 | var1 >> 18) & C_MASK_1;
}
PERM_MACRO(C15) {
    var1var2(C_MASK_1);
    board.b1 = board.b1 & ~C_MASK_1 | (var1 << 18 | var2 >> 36) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 << 18 | var1 >> 36) & C_MASK_1;
}


PERM_MACRO(C21) {
    var1var2(C_MASK_2);
    board.b1 = board.b1 & ~C_MASK_2 | (var2 << 36 | var1 >> 18) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var1 << 36 | var2 >> 18) & C_MASK_2;
}
PERM_MACRO(C22) {
    var1var2(C_MASK_2);
    board.b1 = board.b1 & ~C_MASK_2 | (var2 << 18 | var1 >> 36) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var1 << 18 | var2 >> 36) & C_MASK_2;
}
PERM_MACRO(C23) {
    var1var2(C_MASK_2);
    board.b1 = board.b1 & ~C_MASK_2 | var2;
    board.b2 = board.b2 & ~C_MASK_2 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_2 | (var2 << 0 | var1 >> 0) & C_MASK_2;
    // board.b2 = board.b2 & ~C_MASK_2 | (var1 << 0 | var2 >> 0) & C_MASK_2;
    // or
    // board.b1 = board.b1 & ~C_MASK_2 | (var1 << 0 | var2 >> 0) & C_MASK_2;
    // board.b2 = board.b2 & ~C_MASK_2 | (var2 << 0 | var1 >> 0) & C_MASK_2;
}
PERM_MACRO(C24) {
    var1var2(C_MASK_2);
    board.b1 = board.b1 & ~C_MASK_2 | (var1 << 36 | var2 >> 18) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 << 36 | var1 >> 18) & C_MASK_2;
}
PERM_MACRO(C25) {
    var1var2(C_MASK_2);
    board.b1 = board.b1 & ~C_MASK_2 | (var1 << 18 | var2 >> 36) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 << 18 | var1 >> 36) & C_MASK_2;
}


PERM_MACRO(C31) {
    var1var2(C_MASK_3);
    board.b1 = board.b1 & ~C_MASK_3 | (var2 << 36 | var1 >> 18) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var1 << 36 | var2 >> 18) & C_MASK_3;
}
PERM_MACRO(C32) {
    var1var2(C_MASK_3);
    board.b1 = board.b1 & ~C_MASK_3 | (var2 << 18 | var1 >> 36) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var1 << 18 | var2 >> 36) & C_MASK_3;
}
PERM_MACRO(C33) {
    var1var2(C_MASK_3);
    board.b1 = board.b1 & ~C_MASK_3 | var2;
    board.b2 = board.b2 & ~C_MASK_3 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_3 | (var2 << 0 | var1 >> 0) & C_MASK_3;
    // board.b2 = board.b2 & ~C_MASK_3 | (var1 << 0 | var2 >> 0) & C_MASK_3;
    // or
    // board.b1 = board.b1 & ~C_MASK_3 | (var1 << 0 | var2 >> 0) & C_MASK_3;
    // board.b2 = board.b2 & ~C_MASK_3 | (var2 << 0 | var1 >> 0) & C_MASK_3;
}
PERM_MACRO(C34) {
    var1var2(C_MASK_3);
    board.b1 = board.b1 & ~C_MASK_3 | (var1 << 36 | var2 >> 18) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 << 36 | var1 >> 18) & C_MASK_3;
}
PERM_MACRO(C35) {
    var1var2(C_MASK_3);
    board.b1 = board.b1 & ~C_MASK_3 | (var1 << 18 | var2 >> 36) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 << 18 | var1 >> 36) & C_MASK_3;
}


PERM_MACRO(C41) {
    var1var2(C_MASK_4);
    board.b1 = board.b1 & ~C_MASK_4 | (var2 << 36 | var1 >> 18) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var1 << 36 | var2 >> 18) & C_MASK_4;
}
PERM_MACRO(C42) {
    var1var2(C_MASK_4);
    board.b1 = board.b1 & ~C_MASK_4 | (var2 << 18 | var1 >> 36) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var1 << 18 | var2 >> 36) & C_MASK_4;
}
PERM_MACRO(C43) {
    var1var2(C_MASK_4);
    board.b1 = board.b1 & ~C_MASK_4 | var2;
    board.b2 = board.b2 & ~C_MASK_4 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_4 | (var2 << 0 | var1 >> 0) & C_MASK_4;
    // board.b2 = board.b2 & ~C_MASK_4 | (var1 << 0 | var2 >> 0) & C_MASK_4;
    // or
    // board.b1 = board.b1 & ~C_MASK_4 | (var1 << 0 | var2 >> 0) & C_MASK_4;
    // board.b2 = board.b2 & ~C_MASK_4 | (var2 << 0 | var1 >> 0) & C_MASK_4;
}
PERM_MACRO(C44) {
    var1var2(C_MASK_4);
    board.b1 = board.b1 & ~C_MASK_4 | (var1 << 36 | var2 >> 18) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 << 36 | var1 >> 18) & C_MASK_4;
}
PERM_MACRO(C45) {
    var1var2(C_MASK_4);
    board.b1 = board.b1 & ~C_MASK_4 | (var1 << 18 | var2 >> 36) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 << 18 | var1 >> 36) & C_MASK_4;
}


PERM_MACRO(C51) {
    var1var2(C_MASK_5);
    board.b1 = board.b1 & ~C_MASK_5 | (var1 >> 18 | var2 << 36) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 >> 18 | var1 << 36) & C_MASK_5;
}
PERM_MACRO(C52) {
    var1var2(C_MASK_5);
    board.b1 = board.b1 & ~C_MASK_5 | (var1 >> 36 | var2 << 18) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 >> 36 | var1 << 18) & C_MASK_5;
}
PERM_MACRO(C53) {
    var1var2(C_MASK_5);
    board.b1 = board.b1 & ~C_MASK_5 | var2;
    board.b2 = board.b2 & ~C_MASK_5 | var1;
    // or
    // board.b1 = board.b1 & ~C_MASK_5 | (var2 << 0 | var1 >> 0) & C_MASK_5;
    // board.b2 = board.b2 & ~C_MASK_5 | (var1 << 0 | var2 >> 0) & C_MASK_5;
    // or
    // board.b1 = board.b1 & ~C_MASK_5 | (var1 << 0 | var2 >> 0) & C_MASK_5;
    // board.b2 = board.b2 & ~C_MASK_5 | (var2 << 0 | var1 >> 0) & C_MASK_5;
}
PERM_MACRO(C54) {
    var1var2(C_MASK_5);
    board.b1 = board.b1 & ~C_MASK_5 | (var1 << 36 | var2 >> 18) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 << 36 | var1 >> 18) & C_MASK_5;
}
PERM_MACRO(C55) {
    var1var2(C_MASK_5);
    board.b1 = board.b1 & ~C_MASK_5 | (var1 << 18 | var2 >> 36) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 << 18 | var1 >> 36) & C_MASK_5;
}


/*
static constexpr u64 C_MASK_01 = 0'770000'770000'770000;
static constexpr u64 C_MASK_12 = 0'077000'077000'077000;
static constexpr u64 C_MASK_23 = 0'007700'007700'007700;
static constexpr u64 C_MASK_34 = 0'000770'000770'000770;
static constexpr u64 C_MASK_45 = 0'000077'000077'000077;





PERM_MACRO(C_01_1) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(1);
}

PERM_MACRO(C_01_2) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(2);
}

PERM_MACRO(C_01_3) {
    C u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_01;
    board.addFatY(3);
}

PERM_MACRO(C_01_4) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(4);
}

PERM_MACRO(C_01_5) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(5);
}



PERM_MACRO(C_12_1) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(1);
}

PERM_MACRO(C_12_2) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(2);
}

PERM_MACRO(C_12_3) {
    C u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_12;
    board.addFatY(3);
}

PERM_MACRO(C_12_4) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(4);
}

PERM_MACRO(C_12_5) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(5);
}



PERM_MACRO(C_23_1) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(1);
}

PERM_MACRO(C_23_2) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(2);
}

PERM_MACRO(C_23_3) {
    C u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_23;
    board.addFatY(3);
}

PERM_MACRO(C_23_4) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(4);
}

PERM_MACRO(C_23_5) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(5);
}



PERM_MACRO(C_34_1) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(1);
}

PERM_MACRO(C_34_2) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(2);
}

PERM_MACRO(C_34_3) {
    C u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_34;
    board.addFatY(3);
}

PERM_MACRO(C_34_4) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(4);
}

PERM_MACRO(C_34_5) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(5);
}



PERM_MACRO(C_45_1) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(1);
}

PERM_MACRO(C_45_2) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(2);
}

PERM_MACRO(C_45_3) {
    C u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_45;
    board.addFatY(3);
}

PERM_MACRO(C_45_4) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(4);
}

PERM_MACRO(C_45_5) {
    C u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(5);
}
*/


PERM_MACRO(C011) {
    C01(board);
    C11(board);
    board.addFatY(1);
}
PERM_MACRO(C012) {
    C02(board);
    C12(board);
    board.addFatY(2);
}
PERM_MACRO(C013) {
    C03(board);
    C13(board);
    board.addFatY(3);
}
PERM_MACRO(C014) {
    C04(board);
    C14(board);
    board.addFatY(4);
}
PERM_MACRO(C015) {
    C05(board);
    C15(board);
    board.addFatY(5);
}


PERM_MACRO(C121) {
    C11(board);
    C21(board);
    board.addFatY(1);
}
PERM_MACRO(C122) {
    C12(board);
    C22(board);
    board.addFatY(2);
}
PERM_MACRO(C123) {
    C13(board);
    C23(board);
    board.addFatY(3);
}
PERM_MACRO(C124) {
    C14(board);
    C24(board);
    board.addFatY(4);
}
PERM_MACRO(C125) {
    C15(board);
    C25(board);
    board.addFatY(5);
}


PERM_MACRO(C231) {
    C21(board);
    C31(board);
    board.addFatY(1);
}
PERM_MACRO(C232) {
    C22(board);
    C32(board);
    board.addFatY(2);
}
PERM_MACRO(C233) {
    C23(board);
    C33(board);
    board.addFatY(3);
}
PERM_MACRO(C234) {
    C24(board);
    C34(board);
    board.addFatY(4);
}
PERM_MACRO(C235) {
    C25(board);
    C35(board);
    board.addFatY(5);
}


PERM_MACRO(C341) {
    C31(board);
    C41(board);
    board.addFatY(1);
}
PERM_MACRO(C342) {
    C32(board);
    C42(board);
    board.addFatY(2);
}
PERM_MACRO(C343) {
    C33(board);
    C43(board);
    board.addFatY(3);
}
PERM_MACRO(C344) {
    C34(board);
    C44(board);
    board.addFatY(4);
}
PERM_MACRO(C345) {
    C35(board);
    C45(board);
    board.addFatY(5);
}


PERM_MACRO(C451) {
    C41(board);
    C51(board);
    board.addFatY(1);
}
PERM_MACRO(C452) {
    C42(board);
    C52(board);
    board.addFatY(2);
}
PERM_MACRO(C453) {
    C43(board);
    C53(board);
    board.addFatY(3);
}
PERM_MACRO(C454) {
    C44(board);
    C54(board);
    board.addFatY(4);
}
PERM_MACRO(C455) {
    C45(board);
    C55(board);
    board.addFatY(5);
}
