#include "rotations.hpp"
#include <map>


static constexpr u64 MASK_R0_NT = 0'1777'000000'777777'777777;
static constexpr u64 MASK_R0_B1 = 0'0000'777770'000000'000000, MASK_R0_S1 = 0'0000'000007'000000'000000;
static constexpr u64 MASK_R0_B2 = 0'0000'777700'000000'000000, MASK_R0_S2 = 0'0000'000077'000000'000000;
static constexpr u64 MASK_R0_B3 = 0'0000'777000'000000'000000, MASK_R0_S3 = 0'0000'000777'000000'000000;
static constexpr u64 MASK_R0_B4 = 0'0000'770000'000000'000000, MASK_R0_S4 = 0'0000'007777'000000'000000;
static constexpr u64 MASK_R0_B5 = 0'0000'700000'000000'000000, MASK_R0_S5 = 0'0000'077777'000000'000000;
static constexpr u64 MASK_R1_NT = 0'1777'777777'000000'777777;
static constexpr u64 MASK_R1_B1 = 0'0000'000000'777770'000000, MASK_R1_S1 = 0'0000'000000'000007'000000;
static constexpr u64 MASK_R1_B2 = 0'0000'000000'777700'000000, MASK_R1_S2 = 0'0000'000000'000077'000000;
static constexpr u64 MASK_R1_B3 = 0'0000'000000'777000'000000, MASK_R1_S3 = 0'0000'000000'000777'000000;
static constexpr u64 MASK_R1_B4 = 0'0000'000000'770000'000000, MASK_R1_S4 = 0'0000'000000'007777'000000;
static constexpr u64 MASK_R1_B5 = 0'0000'000000'700000'000000, MASK_R1_S5 = 0'0000'000000'077777'000000;
static constexpr u64 MASK_R2_NT = 0'1777'777777'777777'000000;
static constexpr u64 MASK_R2_B1 = 0'0000'000000'000000'777770, MASK_R2_S1 = 0'0000'000000'000000'000007;
static constexpr u64 MASK_R2_B2 = 0'0000'000000'000000'777700, MASK_R2_S2 = 0'0000'000000'000000'000077;
static constexpr u64 MASK_R2_B3 = 0'0000'000000'000000'777000, MASK_R2_S3 = 0'0000'000000'000000'000777;
static constexpr u64 MASK_R2_B4 = 0'0000'000000'000000'770000, MASK_R2_S4 = 0'0000'000000'000000'007777;
static constexpr u64 MASK_R2_B5 = 0'0000'000000'000000'700000, MASK_R2_S5 = 0'0000'000000'000000'077777;
PERM_MACRO(R_0_1) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B1) >> 3 | (board.b1 & MASK_R0_S1) << 15; }
PERM_MACRO(R_0_2) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B2) >> 6 | (board.b1 & MASK_R0_S2) << 12; }
PERM_MACRO(R_0_3) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B3) >> 9 | (board.b1 & MASK_R0_S3) << 9; }
PERM_MACRO(R_0_4) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B4) >> 12 | (board.b1 & MASK_R0_S4) << 6; }
PERM_MACRO(R_0_5) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B5) >> 15 | (board.b1 & MASK_R0_S5) << 3; }
PERM_MACRO(R_1_1) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B1) >> 3 | (board.b1 & MASK_R1_S1) << 15; }
PERM_MACRO(R_1_2) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B2) >> 6 | (board.b1 & MASK_R1_S2) << 12; }
PERM_MACRO(R_1_3) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B3) >> 9 | (board.b1 & MASK_R1_S3) << 9; }
PERM_MACRO(R_1_4) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B4) >> 12 | (board.b1 & MASK_R1_S4) << 6; }
PERM_MACRO(R_1_5) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B5) >> 15 | (board.b1 & MASK_R1_S5) << 3; }
PERM_MACRO(R_2_1) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B1) >> 3 | (board.b1 & MASK_R2_S1) << 15; }
PERM_MACRO(R_2_2) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B2) >> 6 | (board.b1 & MASK_R2_S2) << 12; }
PERM_MACRO(R_2_3) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B3) >> 9 | (board.b1 & MASK_R2_S3) << 9; }
PERM_MACRO(R_2_4) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B4) >> 12 | (board.b1 & MASK_R2_S4) << 6; }
PERM_MACRO(R_2_5) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B5) >> 15 | (board.b1 & MASK_R2_S5) << 3; }
PERM_MACRO(R_3_1) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B1) >> 3 | (board.b2 & MASK_R0_S1) << 15; }
PERM_MACRO(R_3_2) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B2) >> 6 | (board.b2 & MASK_R0_S2) << 12; }
PERM_MACRO(R_3_3) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B3) >> 9 | (board.b2 & MASK_R0_S3) << 9; }
PERM_MACRO(R_3_4) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B4) >> 12 | (board.b2 & MASK_R0_S4) << 6; }
PERM_MACRO(R_3_5) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B5) >> 15 | (board.b2 & MASK_R0_S5) << 3; }
PERM_MACRO(R_4_1) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B1) >> 3 | (board.b2 & MASK_R1_S1) << 15; }
PERM_MACRO(R_4_2) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B2) >> 6 | (board.b2 & MASK_R1_S2) << 12; }
PERM_MACRO(R_4_3) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B3) >> 9 | (board.b2 & MASK_R1_S3) << 9; }
PERM_MACRO(R_4_4) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B4) >> 12 | (board.b2 & MASK_R1_S4) << 6; }
PERM_MACRO(R_4_5) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B5) >> 15 | (board.b2 & MASK_R1_S5) << 3; }
PERM_MACRO(R_5_1) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B1) >> 3 | (board.b2 & MASK_R2_S1) << 15; }
PERM_MACRO(R_5_2) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B2) >> 6 | (board.b2 & MASK_R2_S2) << 12; }
PERM_MACRO(R_5_3) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B3) >> 9 | (board.b2 & MASK_R2_S3) << 9; }
PERM_MACRO(R_5_4) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B4) >> 12 | (board.b2 & MASK_R2_S4) << 6; }
PERM_MACRO(R_5_5) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B5) >> 15 | (board.b2 & MASK_R2_S5) << 3; }


static constexpr u64 C_MASK_0 = 0'700000'700000'700000;
static constexpr u64 C_MASK_1 = 0'070000'070000'070000;
static constexpr u64 C_MASK_2 = 0'007000'007000'007000;
static constexpr u64 C_MASK_3 = 0'000700'000700'000700;
static constexpr u64 C_MASK_4 = 0'000070'000070'000070;
static constexpr u64 C_MASK_5 = 0'000007'000007'000007;


PERM_MACRO(C_0_1) {
    c_u64 var1 = board.b1 & C_MASK_0;
    c_u64 var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | (var1 >> 18 | var2 << 36) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 >> 18 | var1 << 36) & C_MASK_0;
}
PERM_MACRO(C_0_2) {
    c_u64 var1 = board.b1 & C_MASK_0;
    c_u64 var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | (var1 >> 36 | var2 << 18) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 >> 36 | var1 << 18) & C_MASK_0;
}
PERM_MACRO(C_0_3) {
    c_u64 var1 = board.b1 & C_MASK_0;
    c_u64 var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var2;
    board.b2 = board.b2 & ~C_MASK_0 | var1;
}
PERM_MACRO(C_0_4) {
    c_u64 var1 = board.b1 & C_MASK_0;
    c_u64 var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | (var1 << 36 | var2 >> 18) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 << 36 | var1 >> 18) & C_MASK_0;
}
PERM_MACRO(C_0_5) {
    c_u64 var1 = board.b1 & C_MASK_0;
    c_u64 var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | (var1 << 18 | var2 >> 36) & C_MASK_0;
    board.b2 = board.b2 & ~C_MASK_0 | (var2 << 18 | var1 >> 36) & C_MASK_0;
}


PERM_MACRO(C_1_1) {
    c_u64 var1 = board.b1 & C_MASK_1;
    c_u64 var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | (var1 >> 18 | var2 << 36) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 >> 18 | var1 << 36) & C_MASK_1;
}
PERM_MACRO(C_1_2) {
    c_u64 var1 = board.b1 & C_MASK_1;
    c_u64 var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | (var1 >> 36 | var2 << 18) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 >> 36 | var1 << 18) & C_MASK_1;
}
PERM_MACRO(C_1_3) {
    c_u64 var1 = board.b1 & C_MASK_1;
    c_u64 var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var2;
    board.b2 = board.b2 & ~C_MASK_1 | var1;
}
PERM_MACRO(C_1_4) {
    c_u64 var1 = board.b1 & C_MASK_1;
    c_u64 var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | (var1 << 36 | var2 >> 18) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 << 36 | var1 >> 18) & C_MASK_1;
}
PERM_MACRO(C_1_5) {
    c_u64 var1 = board.b1 & C_MASK_1;
    c_u64 var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | (var1 << 18 | var2 >> 36) & C_MASK_1;
    board.b2 = board.b2 & ~C_MASK_1 | (var2 << 18 | var1 >> 36) & C_MASK_1;
}


PERM_MACRO(C_2_1) {
    c_u64 var1 = board.b1 & C_MASK_2;
    c_u64 var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | (var1 >> 18 | var2 << 36) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 >> 18 | var1 << 36) & C_MASK_2;
}
PERM_MACRO(C_2_2) {
    c_u64 var1 = board.b1 & C_MASK_2;
    c_u64 var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | (var1 >> 36 | var2 << 18) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 >> 36 | var1 << 18) & C_MASK_2;
}
PERM_MACRO(C_2_3) {
    c_u64 var1 = board.b1 & C_MASK_2;
    c_u64 var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var2;
    board.b2 = board.b2 & ~C_MASK_2 | var1;
}
PERM_MACRO(C_2_4) {
    c_u64 var1 = board.b1 & C_MASK_2;
    c_u64 var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | (var1 << 36 | var2 >> 18) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 << 36 | var1 >> 18) & C_MASK_2;
}
PERM_MACRO(C_2_5) {
    c_u64 var1 = board.b1 & C_MASK_2;
    c_u64 var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | (var1 << 18 | var2 >> 36) & C_MASK_2;
    board.b2 = board.b2 & ~C_MASK_2 | (var2 << 18 | var1 >> 36) & C_MASK_2;
}


PERM_MACRO(C_3_1) {
    c_u64 var1 = board.b1 & C_MASK_3;
    c_u64 var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 >> 18 | var2 << 36) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 >> 18 | var1 << 36) & C_MASK_3;
}
PERM_MACRO(C_3_2) {
    c_u64 var1 = board.b1 & C_MASK_3;
    c_u64 var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 >> 36 | var2 << 18) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 >> 36 | var1 << 18) & C_MASK_3;
}
PERM_MACRO(C_3_3) {
    c_u64 var1 = board.b1 & C_MASK_3;
    c_u64 var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var2;
    board.b2 = board.b2 & ~C_MASK_3 | var1;
}
PERM_MACRO(C_3_4) {
    c_u64 var1 = board.b1 & C_MASK_3;
    c_u64 var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 << 36 | var2 >> 18) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 << 36 | var1 >> 18) & C_MASK_3;
}
PERM_MACRO(C_3_5) {
    c_u64 var1 = board.b1 & C_MASK_3;
    c_u64 var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 << 18 | var2 >> 36) & C_MASK_3;
    board.b2 = board.b2 & ~C_MASK_3 | (var2 << 18 | var1 >> 36) & C_MASK_3;
}


PERM_MACRO(C_4_1) {
    c_u64 var1 = board.b1 & C_MASK_4;
    c_u64 var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | (var1 >> 18 | var2 << 36) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 >> 18 | var1 << 36) & C_MASK_4;
}
PERM_MACRO(C_4_2) {
    c_u64 var1 = board.b1 & C_MASK_4;
    c_u64 var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | (var1 >> 36 | var2 << 18) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 >> 36 | var1 << 18) & C_MASK_4;
}
PERM_MACRO(C_4_3) {
    c_u64 var1 = board.b1 & C_MASK_4;
    c_u64 var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var2;
    board.b2 = board.b2 & ~C_MASK_4 | var1;
}
PERM_MACRO(C_4_4) {
    c_u64 var1 = board.b1 & C_MASK_4;
    c_u64 var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | (var1 << 36 | var2 >> 18) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 << 36 | var1 >> 18) & C_MASK_4;
}
PERM_MACRO(C_4_5) {
    c_u64 var1 = board.b1 & C_MASK_4;
    c_u64 var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | (var1 << 18 | var2 >> 36) & C_MASK_4;
    board.b2 = board.b2 & ~C_MASK_4 | (var2 << 18 | var1 >> 36) & C_MASK_4;
}


PERM_MACRO(C_5_1) {
    c_u64 var1 = board.b1 & C_MASK_5;
    c_u64 var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | (var1 >> 18 | var2 << 36) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 >> 18 | var1 << 36) & C_MASK_5;
}
PERM_MACRO(C_5_2) {
    c_u64 var1 = board.b1 & C_MASK_5;
    c_u64 var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | (var1 >> 36 | var2 << 18) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 >> 36 | var1 << 18) & C_MASK_5;
}
PERM_MACRO(C_5_3) {
    c_u64 var1 = board.b1 & C_MASK_5;
    c_u64 var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var2;
    board.b2 = board.b2 & ~C_MASK_5 | var1;
}
PERM_MACRO(C_5_4) {
    c_u64 var1 = board.b1 & C_MASK_5;
    c_u64 var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | (var1 << 36 | var2 >> 18) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 << 36 | var1 >> 18) & C_MASK_5;
}
PERM_MACRO(C_5_5) {
    c_u64 var1 = board.b1 & C_MASK_5;
    c_u64 var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | (var1 << 18 | var2 >> 36) & C_MASK_5;
    board.b2 = board.b2 & ~C_MASK_5 | (var2 << 18 | var1 >> 36) & C_MASK_5;
}


/*
static constexpr u64 MASK_R01_NT = 0'1777'000000'000000'777777;
static constexpr u64 MASK_R01_B1 = 0'0000'777770'777770'000000, MASK_R01_S1 = 0'0000'000007'000007'000000;
static constexpr u64 MASK_R01_B2 = 0'0000'777700'777700'000000, MASK_R01_S2 = 0'0000'000077'000077'000000;
static constexpr u64 MASK_R01_B3 = 0'0000'777000'777000'000000, MASK_R01_S3 = 0'0000'000777'000777'000000;
static constexpr u64 MASK_R01_B4 = 0'0000'770000'770000'000000, MASK_R01_S4 = 0'0000'007777'007777'000000;
static constexpr u64 MASK_R01_B5 = 0'0000'700000'700000'000000, MASK_R01_S5 = 0'0000'077777'077777'000000;


static constexpr u64 MASK_R12_NT = 0'1777'777777'000000'000000;
static constexpr u64 MASK_R12_B1 = 0'0000'000000'777770'777770, MASK_R12_S1 = 0'0000'000000'000007'000007;
static constexpr u64 MASK_R12_B2 = 0'0000'000000'777700'777700, MASK_R12_S2 = 0'0000'000000'000077'000077;
static constexpr u64 MASK_R12_B3 = 0'0000'000000'777000'777000, MASK_R12_S3 = 0'0000'000000'000777'000777;
static constexpr u64 MASK_R12_B4 = 0'0000'000000'770000'770000, MASK_R12_S4 = 0'0000'000000'007777'007777;
static constexpr u64 MASK_R12_B5 = 0'0000'000000'700000'700000, MASK_R12_S5 = 0'0000'000000'077777'077777;



PERM_MACRO(R_01_1) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B1) >>  3 | (board.b1 & MASK_R01_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R_01_2) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B2) >>  6 | (board.b1 & MASK_R01_S2) << 12;
    board.addFatX(2);
}

PERM_MACRO(R_01_3) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B3) >>  9 | (board.b1 & MASK_R01_S3) <<  9;
    board.addFatX(3);
}

PERM_MACRO(R_01_4) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B4) >> 12 | (board.b1 & MASK_R01_S4) <<  6;
    board.addFatX(4);
}

PERM_MACRO(R_01_5) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B5) >> 15 | (board.b1 & MASK_R01_S5) <<  3;
    board.addFatX(5);
}

PERM_MACRO(R_12_1) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B1) >>  3 | (board.b1 & MASK_R12_S1) << 15;
    board.addFatX(1);
}

PERM_MACRO(R_12_2) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B2) >>  6 | (board.b1 & MASK_R12_S2) << 12;
    board.addFatX(2);
}

PERM_MACRO(R_12_3) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B3) >>  9 | (board.b1 & MASK_R12_S3) <<  9;
    board.addFatX(3);
}

PERM_MACRO(R_12_4) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B4) >> 12 | (board.b1 & MASK_R12_S4) <<  6;
    board.addFatX(4);
}

PERM_MACRO(R_12_5) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B5) >> 15 | (board.b1 & MASK_R12_S5) <<  3;
    board.addFatX(5);
}

PERM_MACRO(R_23_1) {
    R_2_1(board);
    R_3_1(board);
    board.addFatX(1);
}

PERM_MACRO(R_23_2) {
    R_2_2(board);
    R_3_2(board);
    board.addFatX(2);
}

PERM_MACRO(R_23_3) {
    R_2_3(board);
    R_3_3(board);
    board.addFatX(3);
}

PERM_MACRO(R_23_4) {
    R_2_4(board);
    R_3_4(board);
    board.addFatX(4);
}

PERM_MACRO(R_23_5) {
    R_2_5(board);
    R_3_5(board);
    board.addFatX(5);
}



PERM_MACRO(R_34_1) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B1) >>  3 | (board.b2 & MASK_R01_S1) << 15;
    board.addFatX(1);
}

PERM_MACRO(R_34_2) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B2) >>  6 | (board.b2 & MASK_R01_S2) << 12;
    board.addFatX(2);
}

PERM_MACRO(R_34_3) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B3) >>  9 | (board.b2 & MASK_R01_S3) <<  9;
    board.addFatX(3);
}

PERM_MACRO(R_34_4) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B4) >> 12 | (board.b2 & MASK_R01_S4) <<  6;
    board.addFatX(4);
}

PERM_MACRO(R_34_5) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B5) >> 15 | (board.b2 & MASK_R01_S5) <<  3;
    board.addFatX(5);
}

PERM_MACRO(R_45_1) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B1) >>  3 | (board.b2 & MASK_R12_S1) << 15;
    board.addFatX(1);
}

PERM_MACRO(R_45_2) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B2) >>  6 | (board.b2 & MASK_R12_S2) << 12;
    board.addFatX(2);
}

PERM_MACRO(R_45_3) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B3) >>  9 | (board.b2 & MASK_R12_S3) <<  9;
    board.addFatX(3);
}

PERM_MACRO(R_45_4) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B4) >> 12 | (board.b2 & MASK_R12_S4) <<  6;
    board.addFatX(4);
}

PERM_MACRO(R_45_5) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B5) >> 15 | (board.b2 & MASK_R12_S5) <<  3;
    board.addFatX(5);
}





static constexpr u64 C_MASK_01 = 0'770000'770000'770000;
static constexpr u64 C_MASK_12 = 0'077000'077000'077000;
static constexpr u64 C_MASK_23 = 0'007700'007700'007700;
static constexpr u64 C_MASK_34 = 0'000770'000770'000770;
static constexpr u64 C_MASK_45 = 0'000077'000077'000077;





PERM_MACRO(C_01_1) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(1);
}

PERM_MACRO(C_01_2) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(2);
}

PERM_MACRO(C_01_3) {
    c_u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_01;
    board.addFatY(3);
}

PERM_MACRO(C_01_4) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(4);
}

PERM_MACRO(C_01_5) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(5);
}



PERM_MACRO(C_12_1) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(1);
}

PERM_MACRO(C_12_2) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(2);
}

PERM_MACRO(C_12_3) {
    c_u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_12;
    board.addFatY(3);
}

PERM_MACRO(C_12_4) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(4);
}

PERM_MACRO(C_12_5) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(5);
}



PERM_MACRO(C_23_1) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(1);
}

PERM_MACRO(C_23_2) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(2);
}

PERM_MACRO(C_23_3) {
    c_u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_23;
    board.addFatY(3);
}

PERM_MACRO(C_23_4) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(4);
}

PERM_MACRO(C_23_5) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(5);
}



PERM_MACRO(C_34_1) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(1);
}

PERM_MACRO(C_34_2) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(2);
}

PERM_MACRO(C_34_3) {
    c_u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_34;
    board.addFatY(3);
}

PERM_MACRO(C_34_4) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(4);
}

PERM_MACRO(C_34_5) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(5);
}



PERM_MACRO(C_45_1) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(1);
}

PERM_MACRO(C_45_2) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(2);
}

PERM_MACRO(C_45_3) {
    c_u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_45;
    board.addFatY(3);
}

PERM_MACRO(C_45_4) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(4);
}

PERM_MACRO(C_45_5) {
    c_u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(5);
}
*/


static constexpr u64 MASK_R01_NT = MASK_R0_NT & MASK_R1_NT;
static constexpr u64 MASK_R01_B1 = MASK_R0_B1 | MASK_R1_B1, MASK_R01_S1 = MASK_R0_S1 | MASK_R1_S1;
static constexpr u64 MASK_R01_B2 = MASK_R0_B2 | MASK_R1_B2, MASK_R01_S2 = MASK_R0_S2 | MASK_R1_S2;
static constexpr u64 MASK_R01_B3 = MASK_R0_B3 | MASK_R1_B3, MASK_R01_S3 = MASK_R0_S3 | MASK_R1_S3;
static constexpr u64 MASK_R01_B4 = MASK_R0_B4 | MASK_R1_B4, MASK_R01_S4 = MASK_R0_S4 | MASK_R1_S4;
static constexpr u64 MASK_R01_B5 = MASK_R0_B5 | MASK_R1_B5, MASK_R01_S5 = MASK_R0_S5 | MASK_R1_S5;


PERM_MACRO(R_01_1) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B1) >> 3 | (board.b1 & MASK_R01_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R_01_2) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B2) >> 6 | (board.b1 & MASK_R01_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R_01_3) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B3) >> 9 | (board.b1 & MASK_R01_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R_01_4) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B4) >> 12 | (board.b1 & MASK_R01_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R_01_5) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B5) >> 15 | (board.b1 & MASK_R01_S5) << 3;
    board.addFatX(5);
}


static constexpr u64 MASK_R12_NT = MASK_R1_NT & MASK_R2_NT;
static constexpr u64 MASK_R12_B1 = MASK_R1_B1 | MASK_R2_B1, MASK_R12_S1 = MASK_R1_S1 | MASK_R2_S1;
static constexpr u64 MASK_R12_B2 = MASK_R1_B2 | MASK_R2_B2, MASK_R12_S2 = MASK_R1_S2 | MASK_R2_S2;
static constexpr u64 MASK_R12_B3 = MASK_R1_B3 | MASK_R2_B3, MASK_R12_S3 = MASK_R1_S3 | MASK_R2_S3;
static constexpr u64 MASK_R12_B4 = MASK_R1_B4 | MASK_R2_B4, MASK_R12_S4 = MASK_R1_S4 | MASK_R2_S4;
static constexpr u64 MASK_R12_B5 = MASK_R1_B5 | MASK_R2_B5, MASK_R12_S5 = MASK_R1_S5 | MASK_R2_S5;


PERM_MACRO(R_12_1) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B1) >> 3 | (board.b1 & MASK_R12_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R_12_2) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B2) >> 6 | (board.b1 & MASK_R12_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R_12_3) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B3) >> 9 | (board.b1 & MASK_R12_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R_12_4) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B4) >> 12 | (board.b1 & MASK_R12_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R_12_5) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B5) >> 15 | (board.b1 & MASK_R12_S5) << 3;
    board.addFatX(5);
}


PERM_MACRO(R_23_1) {
    R_2_1(board);
    R_3_1(board);
    board.addFatX(1);
}
PERM_MACRO(R_23_2) {
    R_2_2(board);
    R_3_2(board);
    board.addFatX(2);
}
PERM_MACRO(R_23_3) {
    R_2_3(board);
    R_3_3(board);
    board.addFatX(3);
}
PERM_MACRO(R_23_4) {
    R_2_4(board);
    R_3_4(board);
    board.addFatX(4);
}
PERM_MACRO(R_23_5) {
    R_2_5(board);
    R_3_5(board);
    board.addFatX(5);
}



PERM_MACRO(R_34_1) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B1) >> 3 | (board.b2 & MASK_R01_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R_34_2) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B2) >> 6 | (board.b2 & MASK_R01_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R_34_3) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B3) >> 9 | (board.b2 & MASK_R01_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R_34_4) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B4) >> 12 | (board.b2 & MASK_R01_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R_34_5) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B5) >> 15 | (board.b2 & MASK_R01_S5) << 3;
    board.addFatX(5);
}




PERM_MACRO(R_45_1) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B1) >> 3 | (board.b2 & MASK_R12_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R_45_2) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B2) >> 6 | (board.b2 & MASK_R12_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R_45_3) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B3) >> 9 | (board.b2 & MASK_R12_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R_45_4) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B4) >> 12 | (board.b2 & MASK_R12_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R_45_5) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B5) >> 15 | (board.b2 & MASK_R12_S5) << 3;
    board.addFatX(5);
}


PERM_MACRO(C_01_1) {
    C_0_1(board);
    C_1_1(board);
    board.addFatY(1);
}
PERM_MACRO(C_01_2) {
    C_0_2(board);
    C_1_2(board);
    board.addFatY(2);
}
PERM_MACRO(C_01_3) {
    C_0_3(board);
    C_1_3(board);
    board.addFatY(3);
}
PERM_MACRO(C_01_4) {
    C_0_4(board);
    C_1_4(board);
    board.addFatY(4);
}
PERM_MACRO(C_01_5) {
    C_0_5(board);
    C_1_5(board);
    board.addFatY(5);
}


PERM_MACRO(C_12_1) {
    C_1_1(board);
    C_2_1(board);
    board.addFatY(1);
}
PERM_MACRO(C_12_2) {
    C_1_2(board);
    C_2_2(board);
    board.addFatY(2);
}
PERM_MACRO(C_12_3) {
    C_1_3(board);
    C_2_3(board);
    board.addFatY(3);
}
PERM_MACRO(C_12_4) {
    C_1_4(board);
    C_2_4(board);
    board.addFatY(4);
}
PERM_MACRO(C_12_5) {
    C_1_5(board);
    C_2_5(board);
    board.addFatY(5);
}


PERM_MACRO(C_23_1) {
    C_2_1(board);
    C_3_1(board);
    board.addFatY(1);
}
PERM_MACRO(C_23_2) {
    C_2_2(board);
    C_3_2(board);
    board.addFatY(2);
}
PERM_MACRO(C_23_3) {
    C_2_3(board);
    C_3_3(board);
    board.addFatY(3);
}
PERM_MACRO(C_23_4) {
    C_2_4(board);
    C_3_4(board);
    board.addFatY(4);
}
PERM_MACRO(C_23_5) {
    C_2_5(board);
    C_3_5(board);
    board.addFatY(5);
}


PERM_MACRO(C_34_1) {
    C_3_1(board);
    C_4_1(board);
    board.addFatY(1);
}
PERM_MACRO(C_34_2) {
    C_3_2(board);
    C_4_2(board);
    board.addFatY(2);
}
PERM_MACRO(C_34_3) {
    C_3_3(board);
    C_4_3(board);
    board.addFatY(3);
}
PERM_MACRO(C_34_4) {
    C_3_4(board);
    C_4_4(board);
    board.addFatY(4);
}
PERM_MACRO(C_34_5) {
    C_3_5(board);
    C_4_5(board);
    board.addFatY(5);
}


PERM_MACRO(C_45_1) {
    C_4_1(board);
    C_5_1(board);
    board.addFatY(1);
}
PERM_MACRO(C_45_2) {
    C_4_2(board);
    C_5_2(board);
    board.addFatY(2);
}
PERM_MACRO(C_45_3) {
    C_4_3(board);
    C_5_3(board);
    board.addFatY(3);
}
PERM_MACRO(C_45_4) {
    C_4_4(board);
    C_5_4(board);
    board.addFatY(4);
}
PERM_MACRO(C_45_5) {
    C_4_5(board);
    C_5_5(board);
    board.addFatY(5);
}


Action allActionsList[110] = {
        R_0_1,  R_0_2,  R_0_3,  R_0_4,  R_0_5,
        R_1_1,  R_1_2,  R_1_3,  R_1_4,  R_1_5,
        R_2_1,  R_2_2,  R_2_3,  R_2_4,  R_2_5,
        R_3_1,  R_3_2,  R_3_3,  R_3_4,  R_3_5,
        R_4_1,  R_4_2,  R_4_3,  R_4_4,  R_4_5,
        R_5_1,  R_5_2,  R_5_3,  R_5_4,  R_5_5,
        C_0_1,  C_0_2,  C_0_3,  C_0_4,  C_0_5,
        C_1_1,  C_1_2,  C_1_3,  C_1_4,  C_1_5,
        C_2_1,  C_2_2,  C_2_3,  C_2_4,  C_2_5,
        C_3_1,  C_3_2,  C_3_3,  C_3_4,  C_3_5,
        C_4_1,  C_4_2,  C_4_3,  C_4_4,  C_4_5,
        C_5_1,  C_5_2,  C_5_3,  C_5_4,  C_5_5,
        R_01_1, R_01_2, R_01_3, R_01_4, R_01_5,
        R_12_1, R_12_2, R_12_3, R_12_4, R_12_5,
        R_23_1, R_23_2, R_23_3, R_23_4, R_23_5,
        R_34_1, R_34_2, R_34_3, R_34_4, R_34_5,
        R_45_1, R_45_2, R_45_3, R_45_4, R_45_5,
        C_01_1, C_01_2, C_01_3, C_01_4, C_01_5,
        C_12_1, C_12_2, C_12_3, C_12_4, C_12_5,
        C_23_1, C_23_2, C_23_3, C_23_4, C_23_5,
        C_34_1, C_34_2, C_34_3, C_34_4, C_34_5,
        C_45_1, C_45_2, C_45_3, C_45_4, C_45_5
};


ActStruct allActStructList[110] = {
        {R_0_1,    0, 2, 5, 0}, {R_0_2,    1, 2, 4, 1}, {R_0_3,    2, 2, 3, 2}, {R_0_4,    3, 2, 2, 3}, {R_0_5,    4, 2, 1, 4},
        {R_1_1,    5, 2, 5, 0}, {R_1_2,    6, 2, 4, 1}, {R_1_3,    7, 2, 3, 2}, {R_1_4,    8, 2, 2, 3}, {R_1_5,    9, 2, 1, 4},
        {R_2_1,   10, 2, 5, 0}, {R_2_2,   11, 2, 4, 1}, {R_2_3,   12, 2, 3, 2}, {R_2_4,   13, 2, 2, 3}, {R_2_5,   14, 2, 1, 4},
        {R_3_1,   15, 2, 5, 0}, {R_3_2,   16, 2, 4, 1}, {R_3_3,   17, 2, 3, 2}, {R_3_4,   18, 2, 2, 3}, {R_3_5,   19, 2, 1, 4},
        {R_4_1,   20, 2, 5, 0}, {R_4_2,   21, 2, 4, 1}, {R_4_3,   22, 2, 3, 2}, {R_4_4,   23, 2, 2, 3}, {R_4_5,   24, 2, 1, 4},
        {R_5_1,   25, 2, 5, 0}, {R_5_2,   26, 2, 4, 1}, {R_5_3,   27, 2, 3, 2}, {R_5_4,   28, 2, 2, 3}, {R_5_5,   29, 2, 1, 4},

        {C_0_1,   30, 1, 5, 0}, {C_0_2,   31, 1, 4, 1}, {C_0_3,   32, 1, 3, 2}, {C_0_4,   33, 1, 2, 3}, {C_0_5,   34, 1, 1, 4},
        {C_1_1,   35, 1, 5, 0}, {C_1_2,   36, 1, 4, 1}, {C_1_3,   37, 1, 3, 2}, {C_1_4,   38, 1, 2, 3}, {C_1_5,   39, 1, 1, 4},
        {C_2_1,   40, 1, 5, 0}, {C_2_2,   41, 1, 4, 1}, {C_2_3,   42, 1, 3, 2}, {C_2_4,   43, 1, 2, 3}, {C_2_5,   44, 1, 1, 4},
        {C_3_1,   45, 1, 5, 0}, {C_3_2,   46, 1, 4, 1}, {C_3_3,   47, 1, 3, 2}, {C_3_4,   48, 1, 2, 3}, {C_3_5,   49, 1, 1, 4},
        {C_4_1,   50, 1, 5, 0}, {C_4_2,   51, 1, 4, 1}, {C_4_3,   52, 1, 3, 2}, {C_4_4,   53, 1, 2, 3}, {C_4_5,   54, 1, 1, 4},
        {C_5_1,   55, 1, 5, 0}, {C_5_2,   56, 1, 4, 1}, {C_5_3,   57, 1, 3, 2}, {C_5_4,   58, 1, 2, 3}, {C_5_5,   59, 1, 1, 4},

        {R_01_1,  60, 0, 4, 0}, {R_01_2,  61, 0, 3, 1}, {R_01_3,  62, 0, 2, 2}, {R_01_4,  63, 0, 1, 3}, {R_01_5,  64, 0, 0, 4},
        {R_12_1,  65, 0, 4, 0}, {R_12_2,  66, 0, 3, 1}, {R_12_3,  67, 0, 2, 2}, {R_12_4,  68, 0, 1, 3}, {R_12_5,  69, 0, 0, 4},
        {R_23_1,  70, 0, 4, 0}, {R_23_2,  71, 0, 3, 1}, {R_23_3,  72, 0, 2, 2}, {R_23_4,  73, 0, 1, 3}, {R_23_5,  74, 0, 0, 4},
        {R_34_1,  75, 0, 4, 0}, {R_34_2,  76, 0, 3, 1}, {R_34_3,  77, 0, 2, 2}, {R_34_4,  78, 0, 1, 3}, {R_34_5,  79, 0, 0, 4},
        {R_45_1,  80, 0, 4, 0}, {R_45_2,  81, 0, 3, 1}, {R_45_3,  82, 0, 2, 2}, {R_45_4,  83, 0, 1, 3}, {R_45_5,  84, 0, 0, 4},

        {C_01_1,  85, 0, 4, 0}, {C_01_2,  86, 0, 3, 1}, {C_01_3,  87, 0, 2, 2}, {C_01_4,  88, 0, 1, 3}, {C_01_5,  89, 0, 0, 4},
        {C_12_1,  90, 0, 4, 0}, {C_12_2,  91, 0, 3, 1}, {C_12_3,  92, 0, 2, 2}, {C_12_4,  93, 0, 1, 3}, {C_12_5,  94, 0, 0, 4},
        {C_23_1,  95, 0, 4, 0}, {C_23_2,  96, 0, 3, 1}, {C_23_3,  97, 0, 2, 2}, {C_23_4,  98, 0, 1, 3}, {C_23_5,  99, 0, 0, 4},
        {C_34_1, 100, 0, 4, 0}, {C_34_2, 101, 0, 3, 1}, {C_34_3, 102, 0, 2, 2}, {C_34_4, 103, 0, 1, 3}, {C_34_5, 104, 0, 0, 4},
        {C_45_1, 105, 0, 4, 0}, {C_45_2, 106, 0, 3, 1}, {C_45_3, 107, 0, 2, 2}, {C_45_4, 108, 0, 1, 3}, {C_45_5, 109, 0, 0, 4},
};



/*
R starts at  0, ends at 23
C starts at 24, ends at 47


 */
u8 fatActionsIndexes[25][48] = {
/*  X  Y          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23     24  25  26  27  28  29  30  31  32  33  34  35  36  37  38   39   40   41   42   43   44   45   46   47 */
/* (0, 0)  0 */ {60, 61, 62, 63, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    85, 86, 87, 88, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  4 | x<30: 23 |  x>=30: 28 | x<60: 47 */
/* (0, 1)  1 */ { 0,  1,  2,  3,  4, 65, 66, 67, 68, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    85, 86, 87, 89, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 28 | x<60: 47 */
/* (0, 2)  2 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 70, 71, 72, 73, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    85, 86, 88, 89, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 28 | x<60: 47 */
/* (0, 3)  3 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 75, 76, 77, 78, 25, 26, 27, 28, 29,    85, 87, 88, 89, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 28 | x<60: 47 */
/* (0, 4)  4 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83,    86, 87, 88, 89, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 19 |  x>=30: 28 | x<60: 47 */
/* (1, 0)  5 */ {60, 61, 62, 64, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 90, 91, 92, 93, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  4 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (1, 1)  6 */ { 0,  1,  2,  3,  4, 65, 66, 67, 69, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 90, 91, 92, 94, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (1, 2)  7 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 70, 71, 72, 74, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 90, 91, 93, 94, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (1, 3)  8 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 75, 76, 77, 79, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 90, 92, 93, 94, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (1, 4)  9 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 84,    30, 31, 32, 33, 34, 91, 92, 93, 94, 45, 46, 47, 48, 49, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 19 |  x>=30: 24 | x<60: 47 */
/* (2, 0) 10 */ {60, 61, 63, 64, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 95, 96, 97, 98, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  4 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (2, 1) 11 */ { 0,  1,  2,  3,  4, 65, 66, 68, 69, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 95, 96, 97, 99, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (2, 2) 12 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 70, 71, 73, 74, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 95, 96, 98, 99, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (2, 3) 13 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 75, 76, 78, 79, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 95, 97, 98, 99, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (2, 4) 14 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 83, 84,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 96, 97, 98, 99, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 19 |  x>=30: 24 | x<60: 47 */
/* (3, 0) 15 */ {60, 62, 63, 64, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 100, 101, 102, 103,  55,  56,  57,  58,  59}, /* x>0:  4 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (3, 1) 16 */ { 0,  1,  2,  3,  4, 65, 67, 68, 69, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 100, 101, 102, 104,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (3, 2) 17 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 70, 72, 73, 74, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 100, 101, 103, 104,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (3, 3) 18 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 75, 77, 78, 79, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 100, 102, 103, 104,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (3, 4) 19 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 82, 83, 84,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 101, 102, 103, 104,  55,  56,  57,  58,  59}, /* x>0:  0 | x<30: 19 |  x>=30: 24 | x<60: 47 */
/* (4, 0) 20 */ {61, 62, 63, 64, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  45,  46,  47,  48,  49, 105, 106, 107, 108}, /* x>0:  4 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (4, 1) 21 */ { 0,  1,  2,  3,  4, 66, 67, 68, 69, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  45,  46,  47,  48,  49, 105, 106, 107, 109}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (4, 2) 22 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 71, 72, 73, 74, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  45,  46,  47,  48,  49, 105, 106, 108, 109}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (4, 3) 23 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 76, 77, 78, 79, 25, 26, 27, 28, 29,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  45,  46,  47,  48,  49, 105, 107, 108, 109}, /* x>0:  0 | x<30: 23 |  x>=30: 24 | x<60: 47 */
/* (4, 4) 24 */ { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 81, 82, 83, 84,    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,  45,  46,  47,  48,  49, 106, 107, 108, 109}, /* x>0:  0 | x<30: 19 |  x>=30: 24 | x<60: 47 */
};


static std::map<Action, std::string> actionToNameLookup = {
        {R_0_1,  "R01" }, {R_0_2,  "R02" }, {R_0_3,  "R03" }, {R_0_4,  "R04" }, {R_0_5,  "R05" },
        {R_1_1,  "R11" }, {R_1_2,  "R12" }, {R_1_3,  "R13" }, {R_1_4,  "R14" }, {R_1_5,  "R15" },
        {R_2_1,  "R21" }, {R_2_2,  "R22" }, {R_2_3,  "R23" }, {R_2_4,  "R24" }, {R_2_5,  "R25" },
        {R_3_1,  "R31" }, {R_3_2,  "R32" }, {R_3_3,  "R33" }, {R_3_4,  "R34" }, {R_3_5,  "R35" },
        {R_4_1,  "R41" }, {R_4_2,  "R42" }, {R_4_3,  "R43" }, {R_4_4,  "R44" }, {R_4_5,  "R45" },
        {R_5_1,  "R51" }, {R_5_2,  "R52" }, {R_5_3,  "R53" }, {R_5_4,  "R54" }, {R_5_5,  "R55" },
        {C_0_1,  "C01" }, {C_0_2,  "C02" }, {C_0_3,  "C03" }, {C_0_4,  "C04" }, {C_0_5,  "C05" },
        {C_1_1,  "C11" }, {C_1_2,  "C12" }, {C_1_3,  "C13" }, {C_1_4,  "C14" }, {C_1_5,  "C15" },
        {C_2_1,  "C21" }, {C_2_2,  "C22" }, {C_2_3,  "C23" }, {C_2_4,  "C24" }, {C_2_5,  "C25" },
        {C_3_1,  "C31" }, {C_3_2,  "C32" }, {C_3_3,  "C33" }, {C_3_4,  "C34" }, {C_3_5,  "C35" },
        {C_4_1,  "C41" }, {C_4_2,  "C42" }, {C_4_3,  "C43" }, {C_4_4,  "C44" }, {C_4_5,  "C45" },
        {C_5_1,  "C51" }, {C_5_2,  "C52" }, {C_5_3,  "C53" }, {C_5_4,  "C54" }, {C_5_5,  "C55" },
        {R_01_1, "R011"}, {R_01_2, "R012"}, {R_01_3, "R013"}, {R_01_4, "R014"}, {R_01_5, "R015"},
        {R_12_1, "R121"}, {R_12_2, "R122"}, {R_12_3, "R123"}, {R_12_4, "R124"}, {R_12_5, "R125"},
        {R_23_1, "R231"}, {R_23_2, "R232"}, {R_23_3, "R233"}, {R_23_4, "R234"}, {R_23_5, "R235"},
        {R_34_1, "R341"}, {R_34_2, "R342"}, {R_34_3, "R343"}, {R_34_4, "R344"}, {R_34_5, "R345"},
        {R_45_1, "R451"}, {R_45_2, "R452"}, {R_45_3, "R453"}, {R_45_4, "R454"}, {R_45_5, "R455"},
        {C_01_1, "C011"}, {C_01_2, "C012"}, {C_01_3, "C013"}, {C_01_4, "C014"}, {C_01_5, "C015"},
        {C_12_1, "C121"}, {C_12_2, "C122"}, {C_12_3, "C123"}, {C_12_4, "C124"}, {C_12_5, "C125"},
        {C_23_1, "C231"}, {C_23_2, "C232"}, {C_23_3, "C233"}, {C_23_4, "C234"}, {C_23_5, "C235"},
        {C_34_1, "C341"}, {C_34_2, "C342"}, {C_34_3, "C343"}, {C_34_4, "C344"}, {C_34_5, "C345"},
        {C_45_1, "C451"}, {C_45_2, "C452"}, {C_45_3, "C453"}, {C_45_4, "C454"}, {C_45_5, "C455"},
};


static std::map<std::string, Action> nameToActionLookup = {
        {"R01",  R_0_1 }, {"R02",  R_0_2 }, {"R03",  R_0_3 }, {"R04",  R_0_4 }, {"R05",  R_0_5 },
        {"R11",  R_1_1 }, {"R12",  R_1_2 }, {"R13",  R_1_3 }, {"R14",  R_1_4 }, {"R15",  R_1_5 },
        {"R21",  R_2_1 }, {"R22",  R_2_2 }, {"R23",  R_2_3 }, {"R24",  R_2_4 }, {"R25",  R_2_5 },
        {"R31",  R_3_1 }, {"R32",  R_3_2 }, {"R33",  R_3_3 }, {"R34",  R_3_4 }, {"R35",  R_3_5 },
        {"R41",  R_4_1 }, {"R42",  R_4_2 }, {"R43",  R_4_3 }, {"R44",  R_4_4 }, {"R45",  R_4_5 },
        {"R51",  R_5_1 }, {"R52",  R_5_2 }, {"R53",  R_5_3 }, {"R54",  R_5_4 }, {"R55",  R_5_5 },
        {"C01",  C_0_1 }, {"C02",  C_0_2 }, {"C03",  C_0_3 }, {"C04",  C_0_4 }, {"C05",  C_0_5 },
        {"C11",  C_1_1 }, {"C12",  C_1_2 }, {"C13",  C_1_3 }, {"C14",  C_1_4 }, {"C15",  C_1_5 },
        {"C21",  C_2_1 }, {"C22",  C_2_2 }, {"C23",  C_2_3 }, {"C24",  C_2_4 }, {"C25",  C_2_5 },
        {"C31",  C_3_1 }, {"C32",  C_3_2 }, {"C33",  C_3_3 }, {"C34",  C_3_4 }, {"C35",  C_3_5 },
        {"C41",  C_4_1 }, {"C42",  C_4_2 }, {"C43",  C_4_3 }, {"C44",  C_4_4 }, {"C45",  C_4_5 },
        {"C51",  C_5_1 }, {"C52",  C_5_2 }, {"C53",  C_5_3 }, {"C54",  C_5_4 }, {"C55",  C_5_5 },
        {"R011", R_01_1}, {"R012", R_01_2}, {"R013", R_01_3}, {"R014", R_01_4}, {"R015", R_01_5},
        {"R121", R_12_1}, {"R122", R_12_2}, {"R123", R_12_3}, {"R124", R_12_4}, {"R125", R_12_5},
        {"R231", R_23_1}, {"R232", R_23_2}, {"R233", R_23_3}, {"R234", R_23_4}, {"R235", R_23_5},
        {"R341", R_34_1}, {"R342", R_34_2}, {"R343", R_34_3}, {"R344", R_34_4}, {"R345", R_34_5},
        {"R451", R_45_1}, {"R452", R_45_2}, {"R453", R_45_3}, {"R454", R_45_4}, {"R455", R_45_5},
        {"C011", C_01_1}, {"C012", C_01_2}, {"C013", C_01_3}, {"C014", C_01_4}, {"C015", C_01_5},
        {"C121", C_12_1}, {"C122", C_12_2}, {"C123", C_12_3}, {"C124", C_12_4}, {"C125", C_12_5},
        {"C231", C_23_1}, {"C232", C_23_2}, {"C233", C_23_3}, {"C234", C_23_4}, {"C235", C_23_5},
        {"C341", C_34_1}, {"C342", C_34_2}, {"C343", C_34_3}, {"C344", C_34_4}, {"C345", C_34_5},
        {"C451", C_45_1}, {"C452", C_45_2}, {"C453", C_45_3}, {"C454", C_45_4}, {"C455", C_45_5},
};


std::map<Action, u8> actionToIndex = {
        {R_0_1,    0}, {R_0_2,    1}, {R_0_3,    2}, {R_0_4,    3}, {R_0_5,    4},
        {R_1_1,    5}, {R_1_2,    6}, {R_1_3,    7}, {R_1_4,    8}, {R_1_5,    9},
        {R_2_1,   10}, {R_2_2,   11}, {R_2_3,   12}, {R_2_4,   13}, {R_2_5,   14},
        {R_3_1,   15}, {R_3_2,   16}, {R_3_3,   17}, {R_3_4,   18}, {R_3_5,   19},
        {R_4_1,   20}, {R_4_2,   21}, {R_4_3,   22}, {R_4_4,   23}, {R_4_5,   24},
        {R_5_1,   25}, {R_5_2,   26}, {R_5_3,   27}, {R_5_4,   28}, {R_5_5,   29},
        {C_0_1,   30}, {C_0_2,   31}, {C_0_3,   32}, {C_0_4,   33}, {C_0_5,   34},
        {C_1_1,   35}, {C_1_2,   36}, {C_1_3,   37}, {C_1_4,   38}, {C_1_5,   39},
        {C_2_1,   40}, {C_2_2,   41}, {C_2_3,   42}, {C_2_4,   43}, {C_2_5,   44},
        {C_3_1,   45}, {C_3_2,   46}, {C_3_3,   47}, {C_3_4,   48}, {C_3_5,   49},
        {C_4_1,   50}, {C_4_2,   51}, {C_4_3,   52}, {C_4_4,   53}, {C_4_5,   54},
        {C_5_1,   55}, {C_5_2,   56}, {C_5_3,   57}, {C_5_4,   58}, {C_5_5,   59},
        {R_01_1,  60}, {R_01_2,  61}, {R_01_3,  62}, {R_01_4,  63}, {R_01_5,  64},
        {R_12_1,  65}, {R_12_2,  66}, {R_12_3,  67}, {R_12_4,  68}, {R_12_5,  69},
        {R_23_1,  70}, {R_23_2,  71}, {R_23_3,  72}, {R_23_4,  73}, {R_23_5,  74},
        {R_34_1,  75}, {R_34_2,  76}, {R_34_3,  77}, {R_34_4,  78}, {R_34_5,  79},
        {R_45_1,  80}, {R_45_2,  81}, {R_45_3,  82}, {R_45_4,  83}, {R_45_5,  84},
        {C_01_1,  85}, {C_01_2,  86}, {C_01_3,  87}, {C_01_4,  88}, {C_01_5,  89},
        {C_12_1,  90}, {C_12_2,  91}, {C_12_3,  92}, {C_12_4,  93}, {C_12_5,  94},
        {C_23_1,  95}, {C_23_2,  96}, {C_23_3,  97}, {C_23_4,  98}, {C_23_5,  99},
        {C_34_1, 100}, {C_34_2, 101}, {C_34_3, 102}, {C_34_4, 103}, {C_34_5, 104},
        {C_45_1, 105}, {C_45_2, 106}, {C_45_3, 107}, {C_45_4, 108}, {C_45_5, 109}
};


std::string getNameFromAction(const Action action) {
    return actionToNameLookup[action];
}


Action getActionFromName(const std::string &name) {
    return nameToActionLookup[name];
}


u8 getIndexFromAction(Action action) {
    return actionToIndex[action];
}


void applyMoves(Board &board, const HashMem &hashMem) {
    for (int i = 0; i < hashMem.getMemoryConst().getMoveCount(); i++)
        allActionsList[hashMem.getMemoryConst().getMove(i)](board);
}


void applyFatMoves(Board &board, const HashMem &hashMem) {
    for (int index = 0; index < hashMem.getMemoryConst().getMoveCount(); index++) {
        u8 move = hashMem.getMemoryConst().getMove(index);

        u8* funcIndexes = fatActionsIndexes[board.getFatXY()];
        allActionsList[funcIndexes[move]](board);
    }
}


Board makeBoardWithMoves(const Board& board, const HashMem& hashMem) {
    Board temp = board;
    applyMoves(temp, hashMem);
    return temp;
}


Board makeBoardWithFatMoves(const Board& board, const HashMem& hashMem) {
    Board temp = board;
    applyFatMoves(temp, hashMem);
    return temp;
}