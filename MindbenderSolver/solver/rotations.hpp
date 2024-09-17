#pragma once

#include "board.hpp"



constexpr uint64_t R_0_MASK = 0x000FFFFFFFFF;
inline void R_0_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0x3FFF8000000000, MASK_S1 = 0x7000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}
inline void R_0_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0x3FFC0000000000, MASK_S2 = 0x3F000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
inline void R_0_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0x3FE00000000000, MASK_S3 = 0x1FF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
inline void R_0_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0x3F000000000000, MASK_S4 = 0xFFF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
inline void R_0_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0x38000000000000, MASK_S5 = 0x7FFF000000000;
    board.b1 = board.b1 & R_0_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}



constexpr uint64_t R_1_MASK = 0x3FFFF00003FFFF;
inline void R_1_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0xFFFE00000, MASK_S1 = 0x1C0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}
inline void R_1_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0xFFF000000, MASK_S2 = 0xFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
inline void R_1_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0xFF8000000, MASK_S3 = 0x7FC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
inline void R_1_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0xFC0000000, MASK_S4 = 0x3FFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
inline void R_1_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0xE00000000, MASK_S5 = 0x1FFFC0000;
    board.b1 = board.b1 & R_1_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}



constexpr uint64_t R_2_MASK = 0x7FFFFFFFFC0000;
inline void R_2_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0x3FFF8, MASK_S1 = 0x7;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B1) >> 3 | (board.b1 & MASK_S1) << 15;
}
inline void R_2_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0x3FFC0, MASK_S2 = 0x3F;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B2) >> 6 | (board.b1 & MASK_S2) << 12;
}
inline void R_2_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0x3FE00, MASK_S3 = 0x1FF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B3) >> 9 | (board.b1 & MASK_S3) << 9;
}
inline void R_2_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0x3F000, MASK_S4 = 0xFFF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B4) >> 12 | (board.b1 & MASK_S4) << 6;
}
inline void R_2_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0x38000, MASK_S5 = 0x7FFF;
    board.b1 = board.b1 & R_2_MASK | (board.b1 & MASK_B5) >> 15 | (board.b1 & MASK_S5) << 3;
}


inline void R_3_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0x3FFF8000000000, MASK_S1 = 0x7000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
inline void R_3_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0x3FFC0000000000, MASK_S2 = 0x3F000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
inline void R_3_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0x3FE00000000000, MASK_S3 = 0x1FF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
inline void R_3_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0x3F000000000000, MASK_S4 = 0xFFF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
inline void R_3_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0x38000000000000, MASK_S5 = 0x7FFF000000000;
    board.b2 = board.b2 & R_0_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}




inline void R_4_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0xFFFE00000, MASK_S1 = 0x1C0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
inline void R_4_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0xFFF000000, MASK_S2 = 0xFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
inline void R_4_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0xFF8000000, MASK_S3 = 0x7FC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
inline void R_4_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0xFC0000000, MASK_S4 = 0x3FFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
inline void R_4_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0xE00000000, MASK_S5 = 0x1FFFC0000;
    board.b2 = board.b2 & R_1_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}




inline void R_5_1(Board &board) {
    static constexpr uint64_t MASK_B1 = 0x3FFF8, MASK_S1 = 0x7;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B1) >> 3 | (board.b2 & MASK_S1) << 15;
}
inline void R_5_2(Board &board) {
    static constexpr uint64_t MASK_B2 = 0x3FFC0, MASK_S2 = 0x3F;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B2) >> 6 | (board.b2 & MASK_S2) << 12;
}
inline void R_5_3(Board &board) {
    static constexpr uint64_t MASK_B3 = 0x3FE00, MASK_S3 = 0x1FF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B3) >> 9 | (board.b2 & MASK_S3) << 9;
}
inline void R_5_4(Board &board) {
    static constexpr uint64_t MASK_B4 = 0x3F000, MASK_S4 = 0xFFF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B4) >> 12 | (board.b2 & MASK_S4) << 6;
}
inline void R_5_5(Board &board) {
    static constexpr uint64_t MASK_B5 = 0x38000, MASK_S5 = 0x7FFF;
    board.b2 = board.b2 & R_2_MASK | (board.b2 & MASK_B5) >> 15 | (board.b2 & MASK_S5) << 3;
}



static constexpr uint64_t C_MASK_0 = 0x38000E00038000;
inline void C_0_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_0 | var2 >> 18 | var1 << 36;
}
inline void C_0_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_0 | var2 >> 36 | var1 << 18;
}
inline void C_0_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var2;
    board.b2 = board.b2 & ~C_MASK_0 | var1;
}
inline void C_0_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_0 | var2 << 36 | var1 >> 18;
}
inline void C_0_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_0 | var2 << 18 | var1 >> 36;
}


static constexpr uint64_t C_MASK_1 = 0x70001C0007000;
inline void C_1_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_1 | var2 >> 18 | var1 << 36;
}
inline void C_1_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_1 | var2 >> 36 | var1 << 18;
}
inline void C_1_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var2;
    board.b2 = board.b2 & ~C_MASK_1 | var1;
}
inline void C_1_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_1 | var2 << 36 | var1 >> 18;
}
inline void C_1_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_1 | var2 << 18 | var1 >> 36;
}



static constexpr uint64_t C_MASK_2 = 0xE00038000E00;
inline void C_2_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_2 | var2 >> 18 | var1 << 36;
}
inline void C_2_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_2 | var2 >> 36 | var1 << 18;
}
inline void C_2_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var2;
    board.b2 = board.b2 & ~C_MASK_2 | var1;
}
inline void C_2_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_2 | var2 << 36 | var1 >> 18;
}
inline void C_2_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_2 | var2 << 18 | var1 >> 36;
}



static constexpr uint64_t C_MASK_3 = 0x1C00070001C0;
inline void C_3_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 >> 18 | var2 << 36);
    board.b2 = board.b2 & ~C_MASK_3 | (var2 >> 18 | var1 << 36);
}
inline void C_3_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_3 | var2 >> 36 | var1 << 18;
}
inline void C_3_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var2;
    board.b2 = board.b2 & ~C_MASK_3 | var1;
}
inline void C_3_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_3 | var2 << 36 | var1 >> 18;
}
inline void C_3_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_3 | var2 << 18 | var1 >> 36;
}



static constexpr uint64_t C_MASK_4 = 0x38000E00038;
inline void C_4_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_4 | var2 >> 18 | var1 << 36;
}
inline void C_4_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_4 | var2 >> 36 | var1 << 18;
}
inline void C_4_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var2;
    board.b2 = board.b2 & ~C_MASK_4 | var1;
}
inline void C_4_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_4 | var2 << 36 | var1 >> 18;
}
inline void C_4_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_4 | var2 << 18 | var1 >> 36;
}



static constexpr uint64_t C_MASK_5 = 0x70001C0007;
inline void C_5_1(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_5 | var2 >> 18 | var1 << 36;
}
inline void C_5_2(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_5 | var2 >> 36 | var1 << 18;
}
inline void C_5_3(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var2;
    board.b2 = board.b2 & ~C_MASK_5 | var1;
}
inline void C_5_4(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_5 | var2 << 36 | var1 >> 18;
}
inline void C_5_5(Board &board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_5 | var2 << 18 | var1 >> 36;
}

typedef void (*Action)(Board &);
inline Action actions[60] = {
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


