// code/rotations_col.cpp
#include "rotations.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


static constexpr u64 C_MASK_0 = 0'700000'700000'700000;
static constexpr u64 C_MASK_1 = 0'070000'070000'070000;
static constexpr u64 C_MASK_2 = 0'007000'007000'007000;
static constexpr u64 C_MASK_3 = 0'000700'000700'000700;
static constexpr u64 C_MASK_4 = 0'000070'000070'000070;
static constexpr u64 C_MASK_5 = 0'000007'000007'000007;
static constexpr u64 C_MASK_01 = C_MASK_0 | C_MASK_1;
static constexpr u64 C_MASK_12 = C_MASK_1 | C_MASK_2;
static constexpr u64 C_MASK_23 = C_MASK_2 | C_MASK_3;
static constexpr u64 C_MASK_34 = C_MASK_3 | C_MASK_4;
static constexpr u64 C_MASK_45 = C_MASK_4 | C_MASK_5;

enum class ColRotation : u8 {
    Forwards18,
    Forwards36,
    SwapHalves,
    Backwards18,
    Backwards36,
};

template<u64 MASK, ColRotation ROT>
FORCEINLINE HD void applyColumnMoveInlined(B1B2& board) {
    const u64 oldB1 = board.b1;
    const u64 oldB2 = board.b2;
    const u64 upper = oldB1 & MASK;
    const u64 lower = oldB2 & MASK;

    u64 rotatedUpper = 0;
    u64 rotatedLower = 0;

    if constexpr (ROT == ColRotation::Forwards18) {
        rotatedUpper = ((lower << 36) | (upper >> 18)) & MASK;
        rotatedLower = ((upper << 36) | (lower >> 18)) & MASK;
    } else if constexpr (ROT == ColRotation::Forwards36) {
        rotatedUpper = ((lower << 18) | (upper >> 36)) & MASK;
        rotatedLower = ((upper << 18) | (lower >> 36)) & MASK;
    } else if constexpr (ROT == ColRotation::SwapHalves) {
        rotatedUpper = lower;
        rotatedLower = upper;
    } else if constexpr (ROT == ColRotation::Backwards18) {
        rotatedUpper = ((upper << 36) | (lower >> 18)) & MASK;
        rotatedLower = ((lower << 36) | (upper >> 18)) & MASK;
    } else {
        rotatedUpper = ((upper << 18) | (lower >> 36)) & MASK;
        rotatedLower = ((lower << 18) | (upper >> 36)) & MASK;
    }

    board.b1 = (oldB1 & ~MASK) | rotatedUpper;
    board.b2 = (oldB2 & ~MASK) | rotatedLower;
}

template<u64 MASK, ColRotation ROT, u64 FAT_Y_DELTA>
FORCEINLINE HD void applyFatColumnMoveInlined(B1B2& board) {
    static constexpr u64 ADD_FAT_MAGIC = 0x8D116344;

    const u64 oldB1 = board.b1;
    const u64 oldB2 = board.b2;
    const u64 upper = oldB1 & MASK;
    const u64 lower = oldB2 & MASK;

    u64 rotatedUpper = 0;
    u64 rotatedLower = 0;

    if constexpr (ROT == ColRotation::Forwards18) {
        rotatedUpper = ((lower << 36) | (upper >> 18)) & MASK;
        rotatedLower = ((upper << 36) | (lower >> 18)) & MASK;
    } else if constexpr (ROT == ColRotation::Forwards36) {
        rotatedUpper = ((lower << 18) | (upper >> 36)) & MASK;
        rotatedLower = ((upper << 18) | (lower >> 36)) & MASK;
    } else if constexpr (ROT == ColRotation::SwapHalves) {
        rotatedUpper = lower;
        rotatedLower = upper;
    } else if constexpr (ROT == ColRotation::Backwards18) {
        rotatedUpper = ((upper << 36) | (lower >> 18)) & MASK;
        rotatedLower = ((lower << 36) | (upper >> 18)) & MASK;
    } else {
        rotatedUpper = ((upper << 18) | (lower >> 36)) & MASK;
        rotatedLower = ((lower << 18) | (upper >> 36)) & MASK;
    }

    const u64 fatY = (oldB1 >> b1b2::FAT_Y_OFFSET) & 0b111;
    const u64 newFatYBits =
            ((ADD_FAT_MAGIC >> (3 * (fatY + FAT_Y_DELTA) - 1)) & 0b111) << b1b2::FAT_Y_OFFSET;

    const u64 preservedB1 = oldB1 & (~MASK) & b1b2::FAT_Y_MASK;
    board.b1 = preservedB1 | rotatedUpper | newFatYBits;
    board.b2 = (oldB2 & ~MASK) | rotatedLower;
}

HD void C01(B1B2& board) { applyColumnMoveInlined<C_MASK_0, ColRotation::Forwards18>(board); }
HD void C02(B1B2& board) { applyColumnMoveInlined<C_MASK_0, ColRotation::Forwards36>(board); }
HD void C03(B1B2& board) { applyColumnMoveInlined<C_MASK_0, ColRotation::SwapHalves>(board); }
HD void C04(B1B2& board) { applyColumnMoveInlined<C_MASK_0, ColRotation::Backwards18>(board); }
HD void C05(B1B2& board) { applyColumnMoveInlined<C_MASK_0, ColRotation::Backwards36>(board); }

HD void C11(B1B2& board) { applyColumnMoveInlined<C_MASK_1, ColRotation::Forwards18>(board); }
HD void C12(B1B2& board) { applyColumnMoveInlined<C_MASK_1, ColRotation::Forwards36>(board); }
HD void C13(B1B2& board) { applyColumnMoveInlined<C_MASK_1, ColRotation::SwapHalves>(board); }
HD void C14(B1B2& board) { applyColumnMoveInlined<C_MASK_1, ColRotation::Backwards18>(board); }
HD void C15(B1B2& board) { applyColumnMoveInlined<C_MASK_1, ColRotation::Backwards36>(board); }

HD void C21(B1B2& board) { applyColumnMoveInlined<C_MASK_2, ColRotation::Forwards18>(board); }
HD void C22(B1B2& board) { applyColumnMoveInlined<C_MASK_2, ColRotation::Forwards36>(board); }
HD void C23(B1B2& board) { applyColumnMoveInlined<C_MASK_2, ColRotation::SwapHalves>(board); }
HD void C24(B1B2& board) { applyColumnMoveInlined<C_MASK_2, ColRotation::Backwards18>(board); }
HD void C25(B1B2& board) { applyColumnMoveInlined<C_MASK_2, ColRotation::Backwards36>(board); }

HD void C31(B1B2& board) { applyColumnMoveInlined<C_MASK_3, ColRotation::Forwards18>(board); }
HD void C32(B1B2& board) { applyColumnMoveInlined<C_MASK_3, ColRotation::Forwards36>(board); }
HD void C33(B1B2& board) { applyColumnMoveInlined<C_MASK_3, ColRotation::SwapHalves>(board); }
HD void C34(B1B2& board) { applyColumnMoveInlined<C_MASK_3, ColRotation::Backwards18>(board); }
HD void C35(B1B2& board) { applyColumnMoveInlined<C_MASK_3, ColRotation::Backwards36>(board); }

HD void C41(B1B2& board) { applyColumnMoveInlined<C_MASK_4, ColRotation::Forwards18>(board); }
HD void C42(B1B2& board) { applyColumnMoveInlined<C_MASK_4, ColRotation::Forwards36>(board); }
HD void C43(B1B2& board) { applyColumnMoveInlined<C_MASK_4, ColRotation::SwapHalves>(board); }
HD void C44(B1B2& board) { applyColumnMoveInlined<C_MASK_4, ColRotation::Backwards18>(board); }
HD void C45(B1B2& board) { applyColumnMoveInlined<C_MASK_4, ColRotation::Backwards36>(board); }

HD void C51(B1B2& board) { applyColumnMoveInlined<C_MASK_5, ColRotation::Forwards18>(board); }
HD void C52(B1B2& board) { applyColumnMoveInlined<C_MASK_5, ColRotation::Forwards36>(board); }
HD void C53(B1B2& board) { applyColumnMoveInlined<C_MASK_5, ColRotation::SwapHalves>(board); }
HD void C54(B1B2& board) { applyColumnMoveInlined<C_MASK_5, ColRotation::Backwards18>(board); }
HD void C55(B1B2& board) { applyColumnMoveInlined<C_MASK_5, ColRotation::Backwards36>(board); }


template<u64 MASK>
FORCEINLINE HD void applyFatColumnMove1(B1B2& board) {
    applyFatColumnMoveInlined<MASK, ColRotation::Forwards18, 1>(board);
}

template<u64 MASK>
FORCEINLINE HD void applyFatColumnMove2(B1B2& board) {
    applyFatColumnMoveInlined<MASK, ColRotation::Forwards36, 2>(board);
}

template<u64 MASK>
FORCEINLINE HD void applyFatColumnMove3(B1B2& board) {
    applyFatColumnMoveInlined<MASK, ColRotation::SwapHalves, 3>(board);
}

template<u64 MASK>
FORCEINLINE HD void applyFatColumnMove4(B1B2& board) {
    applyFatColumnMoveInlined<MASK, ColRotation::Backwards18, 4>(board);
}

template<u64 MASK>
FORCEINLINE HD void applyFatColumnMove5(B1B2& board) {
    applyFatColumnMoveInlined<MASK, ColRotation::Backwards36, 5>(board);
}




HD void C011(B1B2& board) { applyFatColumnMove1<C_MASK_01>(board); }
HD void C012(B1B2& board) { applyFatColumnMove2<C_MASK_01>(board); }
HD void C013(B1B2& board) { applyFatColumnMove3<C_MASK_01>(board); }
HD void C014(B1B2& board) { applyFatColumnMove4<C_MASK_01>(board); }
HD void C015(B1B2& board) { applyFatColumnMove5<C_MASK_01>(board); }

HD void C121(B1B2& board) { applyFatColumnMove1<C_MASK_12>(board); }
HD void C122(B1B2& board) { applyFatColumnMove2<C_MASK_12>(board); }
HD void C123(B1B2& board) { applyFatColumnMove3<C_MASK_12>(board); }
HD void C124(B1B2& board) { applyFatColumnMove4<C_MASK_12>(board); }
HD void C125(B1B2& board) { applyFatColumnMove5<C_MASK_12>(board); }

HD void C231(B1B2& board) { applyFatColumnMove1<C_MASK_23>(board); }
HD void C232(B1B2& board) { applyFatColumnMove2<C_MASK_23>(board); }
HD void C233(B1B2& board) { applyFatColumnMove3<C_MASK_23>(board); }
HD void C234(B1B2& board) { applyFatColumnMove4<C_MASK_23>(board); }
HD void C235(B1B2& board) { applyFatColumnMove5<C_MASK_23>(board); }

HD void C341(B1B2& board) { applyFatColumnMove1<C_MASK_34>(board); }
HD void C342(B1B2& board) { applyFatColumnMove2<C_MASK_34>(board); }
HD void C343(B1B2& board) { applyFatColumnMove3<C_MASK_34>(board); }
HD void C344(B1B2& board) { applyFatColumnMove4<C_MASK_34>(board); }
HD void C345(B1B2& board) { applyFatColumnMove5<C_MASK_34>(board); }

HD void C451(B1B2& board) { applyFatColumnMove1<C_MASK_45>(board); }
HD void C452(B1B2& board) { applyFatColumnMove2<C_MASK_45>(board); }
HD void C453(B1B2& board) { applyFatColumnMove3<C_MASK_45>(board); }
HD void C454(B1B2& board) { applyFatColumnMove4<C_MASK_45>(board); }
HD void C455(B1B2& board) { applyFatColumnMove5<C_MASK_45>(board); }











/*
HD void C011(B1B2& board) { applyFatColumnMove1(board, C_MASK_01); }
HD void C012(B1B2& board) { applyFatColumnMove2(board, C_MASK_01); }
HD void C013(B1B2& board) { applyFatColumnMove3(board, C_MASK_01); }
HD void C014(B1B2& board) { applyFatColumnMove4(board, C_MASK_01); }
HD void C015(B1B2& board) { applyFatColumnMove5(board, C_MASK_01); }

HD void C121(B1B2& board) { applyFatColumnMove1(board, C_MASK_12); }
HD void C122(B1B2& board) { applyFatColumnMove2(board, C_MASK_12); }
HD void C123(B1B2& board) { applyFatColumnMove3(board, C_MASK_12); }
HD void C124(B1B2& board) { applyFatColumnMove4(board, C_MASK_12); }
HD void C125(B1B2& board) { applyFatColumnMove5(board, C_MASK_12); }

HD void C231(B1B2& board) { applyFatColumnMove1(board, C_MASK_23); }
HD void C232(B1B2& board) { applyFatColumnMove2(board, C_MASK_23); }
HD void C233(B1B2& board) { applyFatColumnMove3(board, C_MASK_23); }
HD void C234(B1B2& board) { applyFatColumnMove4(board, C_MASK_23); }
HD void C235(B1B2& board) { applyFatColumnMove5(board, C_MASK_23); }

HD void C341(B1B2& board) { applyFatColumnMove1(board, C_MASK_34); }
HD void C342(B1B2& board) { applyFatColumnMove2(board, C_MASK_34); }
HD void C343(B1B2& board) { applyFatColumnMove3(board, C_MASK_34); }
HD void C344(B1B2& board) { applyFatColumnMove4(board, C_MASK_34); }
HD void C345(B1B2& board) { applyFatColumnMove5(board, C_MASK_34); }

HD void C451(B1B2& board) { applyFatColumnMove1(board, C_MASK_45); }
HD void C452(B1B2& board) { applyFatColumnMove2(board, C_MASK_45); }
HD void C453(B1B2& board) { applyFatColumnMove3(board, C_MASK_45); }
HD void C454(B1B2& board) { applyFatColumnMove4(board, C_MASK_45); }
HD void C455(B1B2& board) { applyFatColumnMove5(board, C_MASK_45); }
 */
/*
static constexpr u64 C_MASK_01 = 0'770000'770000'770000;
static constexpr u64 C_MASK_12 = 0'077000'077000'077000;
static constexpr u64 C_MASK_23 = 0'007700'007700'007700;
static constexpr u64 C_MASK_34 = 0'000770'000770'000770;
static constexpr u64 C_MASK_45 = 0'000077'000077'000077;





PERM_MACRO(C_01_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(1);
}

PERM_MACRO(C_01_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(2);
}

PERM_MACRO(C_01_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_01;
    board.addFatY(3);
}

PERM_MACRO(C_01_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(4);
}

PERM_MACRO(C_01_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(5);
}



PERM_MACRO(C_12_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(1);
}

PERM_MACRO(C_12_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(2);
}

PERM_MACRO(C_12_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_12;
    board.addFatY(3);
}

PERM_MACRO(C_12_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(4);
}

PERM_MACRO(C_12_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(5);
}



PERM_MACRO(C_23_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(1);
}

PERM_MACRO(C_23_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(2);
}

PERM_MACRO(C_23_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_23;
    board.addFatY(3);
}

PERM_MACRO(C_23_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(4);
}

PERM_MACRO(C_23_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(5);
}



PERM_MACRO(C_34_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(1);
}

PERM_MACRO(C_34_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(2);
}

PERM_MACRO(C_34_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_34;
    board.addFatY(3);
}

PERM_MACRO(C_34_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(4);
}

PERM_MACRO(C_34_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(5);
}



PERM_MACRO(C_45_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(1);
}

PERM_MACRO(C_45_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(2);
}

PERM_MACRO(C_45_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_45;
    board.addFatY(3);
}

PERM_MACRO(C_45_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(4);
}

PERM_MACRO(C_45_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(5);
}
*/
/*
static constexpr u64 C_MASK_01 = 0'770000'770000'770000;
static constexpr u64 C_MASK_12 = 0'077000'077000'077000;
static constexpr u64 C_MASK_23 = 0'007700'007700'007700;
static constexpr u64 C_MASK_34 = 0'000770'000770'000770;
static constexpr u64 C_MASK_45 = 0'000077'000077'000077;





PERM_MACRO(C_01_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(1);
}

PERM_MACRO(C_01_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(2);
}

PERM_MACRO(C_01_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_01;
    board.addFatY(3);
}

PERM_MACRO(C_01_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_01) & C_MASK_01;
    board.addFatY(4);
}

PERM_MACRO(C_01_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_01) & C_MASK_01;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_01) & C_MASK_01;
    board.addFatY(5);
}



PERM_MACRO(C_12_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(1);
}

PERM_MACRO(C_12_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(2);
}

PERM_MACRO(C_12_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_12;
    board.addFatY(3);
}

PERM_MACRO(C_12_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_12) & C_MASK_12;
    board.addFatY(4);
}

PERM_MACRO(C_12_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_12) & C_MASK_12;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_12) & C_MASK_12;
    board.addFatY(5);
}



PERM_MACRO(C_23_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(1);
}

PERM_MACRO(C_23_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(2);
}

PERM_MACRO(C_23_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_23;
    board.addFatY(3);
}

PERM_MACRO(C_23_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_23) & C_MASK_23;
    board.addFatY(4);
}

PERM_MACRO(C_23_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_23) & C_MASK_23;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_23) & C_MASK_23;
    board.addFatY(5);
}



PERM_MACRO(C_34_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(1);
}

PERM_MACRO(C_34_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(2);
}

PERM_MACRO(C_34_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_34;
    board.addFatY(3);
}

PERM_MACRO(C_34_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_34) & C_MASK_34;
    board.addFatY(4);
}

PERM_MACRO(C_34_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_34) & C_MASK_34;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_34) & C_MASK_34;
    board.addFatY(5);
}



PERM_MACRO(C_45_1) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 18 | board.b2 << 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 18 | b1_temp << 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(1);
}

PERM_MACRO(C_45_2) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 >> 36 | board.b2 << 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 >> 36 | b1_temp << 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(2);
}

PERM_MACRO(C_45_3) {
    const u64 var1 = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ board.b2) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ var1) & C_MASK_45;
    board.addFatY(3);
}

PERM_MACRO(C_45_4) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 36 | board.b2 >> 18) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 36 | b1_temp >> 18) & C_MASK_45) & C_MASK_45;
    board.addFatY(4);
}

PERM_MACRO(C_45_5) {
    const u64 b1_temp = board.b1;
    board.b1 = board.b1 ^ (board.b1 ^ (board.b1 << 18 | board.b2 >> 36) & C_MASK_45) & C_MASK_45;
    board.b2 = board.b2 ^ (board.b2 ^ (board.b2 << 18 | b1_temp >> 36) & C_MASK_45) & C_MASK_45;
    board.addFatY(5);
}
*/