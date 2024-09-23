#include "rotations.hpp"


static constexpr u64 R_0_MASK = 0xFFC0'000F'FFFF'FFFF;
PERM_MACRO(R_0_1) {
    static constexpr u64 MASK_B1 = 0x003F'FF80'0000'0000, MASK_S1 = 0x70'0000'0000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}

PERM_MACRO(R_0_2) {
    static constexpr u64 MASK_B2 = 0x003F'FC00'0000'0000, MASK_S2 = 0x3F0'0000'0000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
PERM_MACRO(R_0_3) {
    static constexpr u64 MASK_B3 = 0x003F'E000'0000'0000, MASK_S3 = 0x1FF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
PERM_MACRO(R_0_4) {
    static constexpr u64 MASK_B4 = 0x003F'0000'0000'0000, MASK_S4 = 0xFFF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
PERM_MACRO(R_0_5) {
    static constexpr u64 MASK_B5 = 0x0038'0000'0000'0000, MASK_S5 = 0x7FFF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}



static constexpr u64 R_1_MASK = 0xFFFF'FFF0'0003'FFFF;
PERM_MACRO(R_1_1) {
    static constexpr u64 MASK_B1 = 0xFFFE00000, MASK_S1 = 0x1C0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}
PERM_MACRO(R_1_2) {
    static constexpr u64 MASK_B2 = 0xFFF000000, MASK_S2 = 0xFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
PERM_MACRO(R_1_3) {
    static constexpr u64 MASK_B3 = 0xFF8000000, MASK_S3 = 0x7FC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
PERM_MACRO(R_1_4) {
    static constexpr u64 MASK_B4 = 0xFC0000000, MASK_S4 = 0x3FFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
PERM_MACRO(R_1_5) {
    static constexpr u64 MASK_B5 = 0xE00000000, MASK_S5 = 0x1FFFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}



static constexpr u64 R_2_MASK = 0xFFFF'FFFF'FFFC'0000;
PERM_MACRO(R_2_1) {
    static constexpr u64 MASK_B1 = 0x3FFF8, MASK_S1 = 0x7;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}
PERM_MACRO(R_2_2) {
    static constexpr u64 MASK_B2 = 0x3FFC0, MASK_S2 = 0x3F;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
PERM_MACRO(R_2_3) {
    static constexpr u64 MASK_B3 = 0x3FE00, MASK_S3 = 0x1FF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
PERM_MACRO(R_2_4) {
    static constexpr u64 MASK_B4 = 0x3F000, MASK_S4 = 0xFFF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
PERM_MACRO(R_2_5) {
    static constexpr u64 MASK_B5 = 0x38000, MASK_S5 = 0x7FFF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}


PERM_MACRO(R_3_1) {
    static constexpr u64 MASK_B1 = 0x3FFF8000000000, MASK_S1 = 0x7000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
PERM_MACRO(R_3_2) {
    static constexpr u64 MASK_B2 = 0x3FFC0000000000, MASK_S2 = 0x3F000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
PERM_MACRO(R_3_3) {
    static constexpr u64 MASK_B3 = 0x3FE00000000000, MASK_S3 = 0x1FF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
PERM_MACRO(R_3_4) {
    static constexpr u64 MASK_B4 = 0x3F000000000000, MASK_S4 = 0xFFF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
PERM_MACRO(R_3_5) {
    static constexpr u64 MASK_B5 = 0x38000000000000, MASK_S5 = 0x7FFF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}




PERM_MACRO(R_4_1) {
    static constexpr u64 MASK_B1 = 0xFFFE00000, MASK_S1 = 0x1C0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
PERM_MACRO(R_4_2) {
    static constexpr u64 MASK_B2 = 0xFFF000000, MASK_S2 = 0xFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
PERM_MACRO(R_4_3) {
    static constexpr u64 MASK_B3 = 0xFF8000000, MASK_S3 = 0x7FC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
PERM_MACRO(R_4_4) {
    static constexpr u64 MASK_B4 = 0xFC0000000, MASK_S4 = 0x3FFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
PERM_MACRO(R_4_5) {
    static constexpr u64 MASK_B5 = 0xE00000000, MASK_S5 = 0x1FFFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}





PERM_MACRO(R_5_1) {
    static constexpr u64 MASK_B1 = 0x3FFF8, MASK_S1 = 0x7;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
PERM_MACRO(R_5_2) {
    static constexpr u64 MASK_B2 = 0x3FFC0, MASK_S2 = 0x3F;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
PERM_MACRO(R_5_3) {
    static constexpr u64 MASK_B3 = 0x3FE00, MASK_S3 = 0x1FF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
PERM_MACRO(R_5_4) {
    static constexpr u64 MASK_B4 = 0x3F000, MASK_S4 = 0xFFF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
PERM_MACRO(R_5_5) {
    static constexpr u64 MASK_B5 = 0x38000, MASK_S5 = 0x7FFF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}



static constexpr u64 C_MASK_0 = 0x0038'000E'0003'8000;
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


static constexpr u64 C_MASK_1 = 0x0007'0001'C000'7000;
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



static constexpr u64 C_MASK_2 = 0x0000'E000'3800'0E00;
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



static constexpr u64 C_MASK_3 = 0x0000'1C00'0700'01C0;
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



static constexpr u64 C_MASK_4 = 0x0000'0380'00E0'0038;
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



static constexpr u64 C_MASK_5 = 0x0000'0070'001C'0007;
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



/*                                                                       */
PERM_MACRO(R_01_1) { R_0_1(board); R_1_1(board); board.addFatX(1); }
PERM_MACRO(R_01_2) { R_0_2(board); R_1_2(board); board.addFatX(2); }
PERM_MACRO(R_01_3) { R_0_3(board); R_1_3(board); board.addFatX(3); }
PERM_MACRO(R_01_4) { R_0_4(board); R_1_4(board); board.addFatX(4); }
PERM_MACRO(R_01_5) { R_0_5(board); R_1_5(board); board.addFatX(5); }
PERM_MACRO(R_12_1) { R_1_1(board); R_2_1(board); board.addFatX(1); }
PERM_MACRO(R_12_2) { R_1_2(board); R_2_2(board); board.addFatX(2); }
PERM_MACRO(R_12_3) { R_1_3(board); R_2_3(board); board.addFatX(3); }
PERM_MACRO(R_12_4) { R_1_4(board); R_2_4(board); board.addFatX(4); }
PERM_MACRO(R_12_5) { R_1_5(board); R_2_5(board); board.addFatX(5); }
PERM_MACRO(R_23_1) { R_2_1(board); R_3_1(board); board.addFatX(1); }
PERM_MACRO(R_23_2) { R_2_2(board); R_3_2(board); board.addFatX(2); }
PERM_MACRO(R_23_3) { R_2_3(board); R_3_3(board); board.addFatX(3); }
PERM_MACRO(R_23_4) { R_2_4(board); R_3_4(board); board.addFatX(4); }
PERM_MACRO(R_23_5) { R_2_5(board); R_3_5(board); board.addFatX(5); }
PERM_MACRO(R_34_1) { R_3_1(board); R_4_1(board); board.addFatX(1); }
PERM_MACRO(R_34_2) { R_3_2(board); R_4_2(board); board.addFatX(2); }
PERM_MACRO(R_34_3) { R_3_3(board); R_4_3(board); board.addFatX(3); }
PERM_MACRO(R_34_4) { R_3_4(board); R_4_4(board); board.addFatX(4); }
PERM_MACRO(R_34_5) { R_3_5(board); R_4_5(board); board.addFatX(5); }
PERM_MACRO(R_45_1) { R_4_1(board); R_5_1(board); board.addFatX(1); }
PERM_MACRO(R_45_2) { R_4_2(board); R_5_2(board); board.addFatX(2); }
PERM_MACRO(R_45_3) { R_4_3(board); R_5_3(board); board.addFatX(3); }
PERM_MACRO(R_45_4) { R_4_4(board); R_5_4(board); board.addFatX(4); }
PERM_MACRO(R_45_5) { R_4_5(board); R_5_5(board); board.addFatX(5); }
PERM_MACRO(C_01_1) { C_0_1(board); C_1_1(board); board.addFatY(1); }
PERM_MACRO(C_01_2) { C_0_2(board); C_1_2(board); board.addFatY(2); }
PERM_MACRO(C_01_3) { C_0_3(board); C_1_3(board); board.addFatY(3); }
PERM_MACRO(C_01_4) { C_0_4(board); C_1_4(board); board.addFatY(4); }
PERM_MACRO(C_01_5) { C_0_5(board); C_1_5(board); board.addFatY(5); }
PERM_MACRO(C_12_1) { C_1_1(board); C_2_1(board); board.addFatY(1); }
PERM_MACRO(C_12_2) { C_1_2(board); C_2_2(board); board.addFatY(2); }
PERM_MACRO(C_12_3) { C_1_3(board); C_2_3(board); board.addFatY(3); }
PERM_MACRO(C_12_4) { C_1_4(board); C_2_4(board); board.addFatY(4); }
PERM_MACRO(C_12_5) { C_1_5(board); C_2_5(board); board.addFatY(5); }
PERM_MACRO(C_23_1) { C_2_1(board); C_3_1(board); board.addFatY(1); }
PERM_MACRO(C_23_2) { C_2_2(board); C_3_2(board); board.addFatY(2); }
PERM_MACRO(C_23_3) { C_2_3(board); C_3_3(board); board.addFatY(3); }
PERM_MACRO(C_23_4) { C_2_4(board); C_3_4(board); board.addFatY(4); }
PERM_MACRO(C_23_5) { C_2_5(board); C_3_5(board); board.addFatY(5); }
PERM_MACRO(C_34_1) { C_3_1(board); C_4_1(board); board.addFatY(1); }
PERM_MACRO(C_34_2) { C_3_2(board); C_4_2(board); board.addFatY(2); }
PERM_MACRO(C_34_3) { C_3_3(board); C_4_3(board); board.addFatY(3); }
PERM_MACRO(C_34_4) { C_3_4(board); C_4_4(board); board.addFatY(4); }
PERM_MACRO(C_34_5) { C_3_5(board); C_4_5(board); board.addFatY(5); }
PERM_MACRO(C_45_1) { C_4_1(board); C_5_1(board); board.addFatY(1); }
PERM_MACRO(C_45_2) { C_4_2(board); C_5_2(board); board.addFatY(2); }
PERM_MACRO(C_45_3) { C_4_3(board); C_5_3(board); board.addFatY(3); }
PERM_MACRO(C_45_4) { C_4_4(board); C_5_4(board); board.addFatY(4); }
PERM_MACRO(C_45_5) { C_4_5(board); C_5_5(board); board.addFatY(5); }







std::map<Action, std::string> RCNameForwardLookup = {
        {R_0_1,  "R_0_1"}, {R_0_2,  "R_0_2"}, {R_0_3,  "R_0_3"}, {R_0_4,  "R_0_4"}, {R_0_5,  "R_0_5"},
        {R_1_1,  "R_1_1"}, {R_1_2,  "R_1_2"}, {R_1_3,  "R_1_3"}, {R_1_4,  "R_1_4"}, {R_1_5,  "R_1_5"},
        {R_2_1,  "R_2_1"}, {R_2_2,  "R_2_2"}, {R_2_3,  "R_2_3"}, {R_2_4,  "R_2_4"}, {R_2_5,  "R_2_5"},
        {R_3_1,  "R_3_1"}, {R_3_2,  "R_3_2"}, {R_3_3,  "R_3_3"}, {R_3_4,  "R_3_4"}, {R_3_5,  "R_3_5"},
        {R_4_1,  "R_4_1"}, {R_4_2,  "R_4_2"}, {R_4_3,  "R_4_3"}, {R_4_4,  "R_4_4"}, {R_4_5,  "R_4_5"},
        {R_5_1,  "R_5_1"}, {R_5_2,  "R_5_2"}, {R_5_3,  "R_5_3"}, {R_5_4,  "R_5_4"}, {R_5_5,  "R_5_5"},
        {C_0_1,  "C_0_1"}, {C_0_2,  "C_0_2"}, {C_0_3,  "C_0_3"}, {C_0_4,  "C_0_4"}, {C_0_5,  "C_0_5"},
        {C_1_1,  "C_1_1"}, {C_1_2,  "C_1_2"}, {C_1_3,  "C_1_3"}, {C_1_4,  "C_1_4"}, {C_1_5,  "C_1_5"},
        {C_2_1,  "C_2_1"}, {C_2_2,  "C_2_2"}, {C_2_3,  "C_2_3"}, {C_2_4,  "C_2_4"}, {C_2_5,  "C_2_5"},
        {C_3_1,  "C_3_1"}, {C_3_2,  "C_3_2"}, {C_3_3,  "C_3_3"}, {C_3_4,  "C_3_4"}, {C_3_5,  "C_3_5"},
        {C_4_1,  "C_4_1"}, {C_4_2,  "C_4_2"}, {C_4_3,  "C_4_3"}, {C_4_4,  "C_4_4"}, {C_4_5,  "C_4_5"},
        {C_5_1,  "C_5_1"}, {C_5_2,  "C_5_2"}, {C_5_3,  "C_5_3"}, {C_5_4,  "C_5_4"}, {C_5_5,  "C_5_5"},
        {R_01_1, "R_01_1"}, {R_01_2, "R_01_2"}, {R_01_3, "R_01_3"}, {R_01_4, "R_01_4"}, {R_01_5, "R_01_5"},
        {R_12_1, "R_12_1"}, {R_12_2, "R_12_2"}, {R_12_3, "R_12_3"}, {R_12_4, "R_12_4"}, {R_12_5, "R_12_5"},
        {R_23_1, "R_23_1"}, {R_23_2, "R_23_2"}, {R_23_3, "R_23_3"}, {R_23_4, "R_23_4"}, {R_23_5, "R_23_5"},
        {R_34_1, "R_34_1"}, {R_34_2, "R_34_2"}, {R_34_3, "R_34_3"}, {R_34_4, "R_34_4"}, {R_34_5, "R_34_5"},
        {R_45_1, "R_45_1"}, {R_45_2, "R_45_2"}, {R_45_3, "R_45_3"}, {R_45_4, "R_45_4"}, {R_45_5, "R_45_5"},
        {C_01_1, "C_01_1"}, {C_01_2, "C_01_2"}, {C_01_3, "C_01_3"}, {C_01_4, "C_01_4"}, {C_01_5, "C_01_5"},
        {C_12_1, "C_12_1"}, {C_12_2, "C_12_2"}, {C_12_3, "C_12_3"}, {C_12_4, "C_12_4"}, {C_12_5, "C_12_5"},
        {C_23_1, "C_23_1"}, {C_23_2, "C_23_2"}, {C_23_3, "C_23_3"}, {C_23_4, "C_23_4"}, {C_23_5, "C_23_5"},
        {C_34_1, "C_34_1"}, {C_34_2, "C_34_2"}, {C_34_3, "C_34_3"}, {C_34_4, "C_34_4"}, {C_34_5, "C_34_5"},
        {C_45_1, "C_45_1"}, {C_45_2, "C_45_2"}, {C_45_3, "C_45_3"}, {C_45_4, "C_45_4"}, {C_45_5, "C_45_5"},
};


Action actions[60] = {
        R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,
        R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,
        R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,
        R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,
        R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,
        R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,
        C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,
        C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,
        C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,
        C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,
        C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,
        C_5_1, C_5_2, C_5_3, C_5_4, C_5_5,
};


Action allActionsList[110] = {
        R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,
        R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,
        R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,
        R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,
        R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,
        R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,
        C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,
        C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,
        C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,
        C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,
        C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,
        C_5_1, C_5_2, C_5_3, C_5_4, C_5_5,
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


Action fatActions[25][48] = {
/* (0, 0) */ { R_01_1,  R_01_2,  R_01_3,  R_01_4 , R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_01_1,  C_01_2,  C_01_3,  C_01_4 , C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (0, 1) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_12_1,  R_12_2,  R_12_3,  R_12_4,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_01_1,  C_01_2,  C_01_3,  C_01_5 , C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (0, 2) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_23_1,  R_23_2,  R_23_3,  R_23_4,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_01_1,  C_01_2,  C_01_4,  C_01_5 , C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (0, 3) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_34_1,  R_34_2,  R_34_3,  R_34_4,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_01_1,  C_01_3,  C_01_4,  C_01_5 , C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (0, 4) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_45_1,  R_45_2,  R_45_3,  R_45_4,   C_01_2,  C_01_3,  C_01_4,  C_01_5 , C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (1, 0) */ { R_01_1,  R_01_2,  R_01_3,  R_01_5 , R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_12_1,  C_12_2,  C_12_3,  C_12_4,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (1, 1) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_12_1,  R_12_2,  R_12_3,  R_12_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_12_1,  C_12_2,  C_12_3,  C_12_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (1, 2) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_23_1,  R_23_2,  R_23_3,  R_23_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_12_1,  C_12_2,  C_12_4,  C_12_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (1, 3) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_34_1,  R_34_2,  R_34_3,  R_34_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_12_1,  C_12_3,  C_12_4,  C_12_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (1, 4) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_45_1,  R_45_2,  R_45_3,  R_45_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_12_2,  C_12_3,  C_12_4,  C_12_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (2, 0) */ { R_01_1,  R_01_2,  R_01_4,  R_01_5 , R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_23_1,  C_23_2,  C_23_3,  C_23_4,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (2, 1) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_12_1,  R_12_2,  R_12_4,  R_12_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_23_1,  C_23_2,  C_23_3,  C_23_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (2, 2) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_23_1,  R_23_2,  R_23_4,  R_23_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_23_1,  C_23_2,  C_23_4,  C_23_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (2, 3) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_34_1,  R_34_2,  R_34_4,  R_34_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_23_1,  C_23_3,  C_23_4,  C_23_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (2, 4) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_45_1,  R_45_2,  R_45_4,  R_45_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_23_2,  C_23_3,  C_23_4,  C_23_5,   C_4_1, C_4_2, C_4_3, C_4_4, C_4_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (3, 0) */ { R_01_1,  R_01_3,  R_01_4,  R_01_5 , R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_34_1,  C_34_2,  C_34_3,  C_34_4,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (3, 1) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_12_1,  R_12_3,  R_12_4,  R_12_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_34_1,  C_34_2,  C_34_3,  C_34_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (3, 2) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_23_1,  R_23_3,  R_23_4,  R_23_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_34_1,  C_34_2,  C_34_4,  C_34_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (3, 3) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_34_1,  R_34_3,  R_34_4,  R_34_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_34_1,  C_34_3,  C_34_4,  C_34_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (3, 4) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_45_1,  R_45_3,  R_45_4,  R_45_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_34_2,  C_34_3,  C_34_4,  C_34_5,   C_5_1, C_5_2, C_5_3, C_5_4, C_5_5 },
/* (4, 0) */ { R_01_2,  R_01_3,  R_01_4,  R_01_5 , R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_45_1,  C_45_2,  C_45_3,  C_45_4 },
/* (4, 1) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_12_2,  R_12_3,  R_12_4,  R_12_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_45_1,  C_45_2,  C_45_3,  C_45_5 },
/* (4, 2) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_23_2,  R_23_3,  R_23_4,  R_23_5,   R_4_1, R_4_2, R_4_3, R_4_4, R_4_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_45_1,  C_45_2,  C_45_4,  C_45_5 },
/* (4, 3) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_34_2,  R_34_3,  R_34_4,  R_34_5,   R_5_1, R_5_2, R_5_3, R_5_4, R_5_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_45_1,  C_45_3,  C_45_4,  C_45_5 },
/* (4, 4) */ { R_0_1, R_0_2, R_0_3, R_0_4, R_0_5,  R_1_1, R_1_2, R_1_3, R_1_4, R_1_5,   R_2_1, R_2_2, R_2_3, R_2_4, R_2_5,   R_3_1, R_3_2, R_3_3, R_3_4, R_3_5,   R_45_2,  R_45_3,  R_45_4,  R_45_5,   C_0_1, C_0_2, C_0_3, C_0_4, C_0_5,  C_1_1, C_1_2, C_1_3, C_1_4, C_1_5,   C_2_1, C_2_2, C_2_3, C_2_4, C_2_5,   C_3_1, C_3_2, C_3_3, C_3_4, C_3_5,   C_45_2,  C_45_3,  C_45_4,  C_45_5 },
};



const uintptr_t ActionHelper::smallestPtr = reinterpret_cast<uintptr_t>(R_0_1);
const std::array<ActionHelper::ptrType, 110> ActionHelper::myAllActions = ActionHelper::init_allActions();
const std::array<ActionHelper::ptrType, 60> ActionHelper::myNormalActions = ActionHelper::init_normalActions();
const std::array<std::array<ActionHelper::ptrType, 48>, 25> ActionHelper::myFatActions = ActionHelper::init_fatActions();
