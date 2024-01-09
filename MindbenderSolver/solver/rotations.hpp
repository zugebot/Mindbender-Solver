#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include "board.hpp"


constexpr uint64_t R_03_MASK = 0x0000000FFFFFFFFF;
constexpr uint64_t R_14_MASK = 0x003FFFF00003FFFF;
constexpr uint64_t R_25_MASK = 0x007FFFFFFFFC0000;

/*
// faster version, but need to find out why
inline void R_0_1(Board& board) {
    uint64_t buffer = board.b1 >> 36) & 0x3FFFF;
    buffer = (buffer >> 3) | ((buffer & 0x7) << 15);
    board.b1 = board.b1 & R_03_MASK) | (buffer << 36);
}
 */

inline void R_0_1(Board& board) {
    uint64_t buffer = board.b1 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b1 &= R_03_MASK;
    board.b1 |= buffer << 36;
}

inline void R_0_2(Board& board) {
    uint64_t buffer = board.b1 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b1 &= R_03_MASK;
    board.b1 |= buffer << 36;
}

inline void R_0_3(Board& board) {
    uint64_t buffer = board.b1 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b1 &= R_03_MASK;
    board.b1 |= buffer << 36;
}

inline void R_0_4(Board& board) {
    uint64_t buffer = board.b1 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b1 &= R_03_MASK;
    board.b1 |= buffer << 36;
}

inline void R_0_5(Board& board) {
    uint64_t buffer = board.b1 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b1 &= R_03_MASK;
    board.b1 |= buffer << 36;
}

inline void R_1_1(Board& board) {
    uint64_t buffer = board.b1 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b1 &= R_14_MASK;
    board.b1 |= buffer << 18;
}

inline void R_1_2(Board& board) {
    uint64_t buffer = board.b1 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b1 &= R_14_MASK;
    board.b1 |= buffer << 18;
}

inline void R_1_3(Board& board) {
    uint64_t buffer = board.b1 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b1 &= R_14_MASK;
    board.b1 |= buffer << 18;
}

inline void R_1_4(Board& board) {
    uint64_t buffer = board.b1 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b1 &= R_14_MASK;
    board.b1 |= buffer << 18;
}

inline void R_1_5(Board& board) {
    uint64_t buffer = board.b1 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b1 &= R_14_MASK;
    board.b1 |= buffer << 18;
}

inline void R_2_1(Board& board) {
    uint64_t buffer = board.b1 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b1 &= R_25_MASK;
    board.b1 |= buffer;
}

inline void R_2_2(Board& board) {
    uint64_t buffer = board.b1 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b1 &= R_25_MASK;
    board.b1 |= buffer;
}

inline void R_2_3(Board& board) {
    uint64_t buffer = board.b1 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b1 &= R_25_MASK;
    board.b1 |= buffer;
}

inline void R_2_4(Board& board) {
    uint64_t buffer = board.b1 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b1 &= R_25_MASK;
    board.b1 |= buffer;
}

inline void R_2_5(Board& board) {
    uint64_t buffer = board.b1 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b1 &= R_25_MASK;
    board.b1 |= buffer;
}

inline void R_3_1(Board& board) {
    uint64_t buffer = board.b2 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b2 &= R_03_MASK;
    board.b2 |= buffer << 36;
}

inline void R_3_2(Board& board) {
    uint64_t buffer = board.b2 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b2 &= R_03_MASK;
    board.b2 |= buffer << 36;
}

inline void R_3_3(Board& board) {
    uint64_t buffer = board.b2 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b2 &= R_03_MASK;
    board.b2 |= buffer << 36;
}

inline void R_3_4(Board& board) {
    uint64_t buffer = board.b2 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b2 &= R_03_MASK;
    board.b2 |= buffer << 36;
}

inline void R_3_5(Board& board) {
    uint64_t buffer = board.b2 >> 36 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b2 &= R_03_MASK;
    board.b2 |= buffer << 36;
}


inline void R_4_1(Board& board) {
    uint64_t buffer = board.b2 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b2 &= R_14_MASK;
    board.b2 |= buffer << 18;
}

inline void R_4_2(Board& board) {
    uint64_t buffer = board.b2 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b2 &= R_14_MASK;
    board.b2 |= buffer << 18;
}

inline void R_4_3(Board& board) {
    uint64_t buffer = board.b2 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b2 &= R_14_MASK;
    board.b2 |= buffer << 18;
}

inline void R_4_4(Board& board) {
    uint64_t buffer = board.b2 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b2 &= R_14_MASK;
    board.b2 |= buffer << 18;
}

inline void R_4_5(Board& board) {
    uint64_t buffer = board.b2 >> 18 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b2 &= R_14_MASK;
    board.b2 |= buffer << 18;
}


inline void R_5_1(Board& board) {
    uint64_t buffer = board.b2 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7;
    buffer >>= 3;
    buffer |= shift << 15;
    board.b2 &= R_25_MASK;
    board.b2 |= buffer;
}

inline void R_5_2(Board& board) {
    uint64_t buffer = board.b2 & 0x3FFFF;
    const uint32_t shift = buffer & 0x3F;
    buffer >>= 6;
    buffer |= shift << 12;
    board.b2 &= R_25_MASK;
    board.b2 |= buffer;
}

inline void R_5_3(Board& board) {
    uint64_t buffer = board.b2 & 0x3FFFF;
    const uint32_t shift = buffer & 0x1FF;
    buffer >>= 9;
    buffer |= shift << 9;
    board.b2 &= R_25_MASK;
    board.b2 |= buffer;
}

inline void R_5_4(Board& board) {
    uint64_t buffer = board.b2 & 0x3FFFF;
    const uint32_t shift = buffer & 0xFFF;
    buffer >>= 12;
    buffer |= shift << 6;
    board.b2 &= R_25_MASK;
    board.b2 |= buffer;
}

inline void R_5_5(Board& board) {
    uint64_t buffer = board.b2 & 0x3FFFF;
    const uint32_t shift = buffer & 0x7FFF;
    buffer >>= 15;
    buffer |= shift << 3;
    board.b2 &= R_25_MASK;
    board.b2 |= buffer;
}


static constexpr uint64_t C_MASK_0 = 0x0038000E00038000;
static constexpr uint64_t C_MASK_1 = 0x00070001C0007000;
static constexpr uint64_t C_MASK_2 = 0x0000E00038000E00;
static constexpr uint64_t C_MASK_3 = 0x00001C00070001C0;
static constexpr uint64_t C_MASK_4 = 0x0000038000E00038;
static constexpr uint64_t C_MASK_5 = 0x00000070001C0007;

inline void C_0_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & C_MASK_0 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & C_MASK_0 | var2 >> 18 | var1 << 36;
}

inline void C_0_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_0 | var2 >> 36 | var1 << 18;
}

inline void C_0_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var2;
    board.b2 = board.b2 & ~C_MASK_0 | var1;
}

inline void C_0_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_0 | var2 << 36 | var1 >> 18;
}

inline void C_0_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_0;
    const uint64_t var2 = board.b2 & C_MASK_0;
    board.b1 = board.b1 & ~C_MASK_0 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_0 | var2 << 18 | var1 >> 36;
}

inline void C_1_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_1 | var2 >> 18 | var1 << 36;
}

inline void C_1_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_1 | var2 >> 36 | var1 << 18;
}

inline void C_1_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var2;
    board.b2 = board.b2 & ~C_MASK_1 | var1;
}

inline void C_1_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_1 | var2 << 36 | var1 >> 18;
}

inline void C_1_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_1;
    const uint64_t var2 = board.b2 & C_MASK_1;
    board.b1 = board.b1 & ~C_MASK_1 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_1 | var2 << 18 | var1 >> 36;
}

inline void C_2_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_2 | var2 >> 18 | var1 << 36;
}

inline void C_2_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_2 | var2 >> 36 | var1 << 18;
}

inline void C_2_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var2;
    board.b2 = board.b2 & ~C_MASK_2 | var1;
}

inline void C_2_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_2 | var2 << 36 | var1 >> 18;
}

inline void C_2_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_2;
    const uint64_t var2 = board.b2 & C_MASK_2;
    board.b1 = board.b1 & ~C_MASK_2 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_2 | var2 << 18 | var1 >> 36;
}

inline void C_3_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | (var1 >> 18 | var2 << 36);
    board.b2 = board.b2 & ~C_MASK_3 | (var2 >> 18 | var1 << 36);
}

inline void C_3_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_3 | var2 >> 36 | var1 << 18;
}

inline void C_3_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var2;
    board.b2 = board.b2 & ~C_MASK_3 | var1;
}

inline void C_3_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_3 | var2 << 36 | var1 >> 18;
}

inline void C_3_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_3;
    const uint64_t var2 = board.b2 & C_MASK_3;
    board.b1 = board.b1 & ~C_MASK_3 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_3 | var2 << 18 | var1 >> 36;
}

inline void C_4_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_4 | var2 >> 18 | var1 << 36;
}

inline void C_4_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_4 | var2 >> 36 | var1 << 18;
}

inline void C_4_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var2;
    board.b2 = board.b2 & ~C_MASK_4 | var1;
}

inline void C_4_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_4 | var2 << 36 | var1 >> 18;
}

inline void C_4_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_4;
    const uint64_t var2 = board.b2 & C_MASK_4;
    board.b1 = board.b1 & ~C_MASK_4 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_4 | var2 << 18 | var1 >> 36;
}

inline void C_5_1(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 >> 18 | var2 << 36;
    board.b2 = board.b2 & ~C_MASK_5 | var2 >> 18 | var1 << 36;
}

inline void C_5_2(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 >> 36 | var2 << 18;
    board.b2 = board.b2 & ~C_MASK_5 | var2 >> 36 | var1 << 18;
}

inline void C_5_3(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var2;
    board.b2 = board.b2 & ~C_MASK_5 | var1;
}

inline void C_5_4(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 << 36 | var2 >> 18;
    board.b2 = board.b2 & ~C_MASK_5 | var2 << 36 | var1 >> 18;
}

inline void C_5_5(Board& board) {
    const uint64_t var1 = board.b1 & C_MASK_5;
    const uint64_t var2 = board.b2 & C_MASK_5;
    board.b1 = board.b1 & ~C_MASK_5 | var1 << 18 | var2 >> 36;
    board.b2 = board.b2 & ~C_MASK_5 | var2 << 18 | var1 >> 36;
}

typedef void (*Action)(Board &);

inline Action actions[60] = {
        R_0_1,
        R_0_2,
        R_0_3,
        R_0_4,
        R_0_5,
        R_1_1,
        R_1_2,
        R_1_3,
        R_1_4,
        R_1_5,
        R_2_1,
        R_2_2,
        R_2_3,
        R_2_4,
        R_2_5,
        R_3_1,
        R_3_2,
        R_3_3,
        R_3_4,
        R_3_5,
        R_4_1,
        R_4_2,
        R_4_3,
        R_4_4,
        R_4_5,
        R_5_1,
        R_5_2,
        R_5_3,
        R_5_4,
        R_5_5,
        C_0_1,
        C_0_2,
        C_0_3,
        C_0_4,
        C_0_5,
        C_1_1,
        C_1_2,
        C_1_3,
        C_1_4,
        C_1_5,
        C_2_1,
        C_2_2,
        C_2_3,
        C_2_4,
        C_2_5,
        C_3_1,
        C_3_2,
        C_3_3,
        C_3_4,
        C_3_5,
        C_4_1,
        C_4_2,
        C_4_3,
        C_4_4,
        C_4_5,
        C_5_1,
        C_5_2,
        C_5_3,
        C_5_4,
        C_5_5,
};
