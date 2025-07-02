#include "rotations.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


static constexpr u64 MASK_R0_NT = 0'1777'000000'777777'777777;
static constexpr u64 MASK_R1_NT = 0'1777'777777'000000'777777;
static constexpr u64 MASK_R2_NT = 0'1777'777777'777777'000000;

static constexpr u64 MASK_R0_B1 = 0'0000'777770'000000'000000, MASK_R0_S1 = 0'0000'000007'000000'000000;
static constexpr u64 MASK_R0_B2 = 0'0000'777700'000000'000000, MASK_R0_S2 = 0'0000'000077'000000'000000;
static constexpr u64 MASK_R0_B3 = 0'0000'777000'000000'000000, MASK_R0_S3 = 0'0000'000777'000000'000000;
static constexpr u64 MASK_R0_B4 = 0'0000'770000'000000'000000, MASK_R0_S4 = 0'0000'007777'000000'000000;
static constexpr u64 MASK_R0_B5 = 0'0000'700000'000000'000000, MASK_R0_S5 = 0'0000'077777'000000'000000;

static constexpr u64 MASK_R1_B1 = 0'0000'000000'777770'000000, MASK_R1_S1 = 0'0000'000000'000007'000000;
static constexpr u64 MASK_R1_B2 = 0'0000'000000'777700'000000, MASK_R1_S2 = 0'0000'000000'000077'000000;
static constexpr u64 MASK_R1_B3 = 0'0000'000000'777000'000000, MASK_R1_S3 = 0'0000'000000'000777'000000;
static constexpr u64 MASK_R1_B4 = 0'0000'000000'770000'000000, MASK_R1_S4 = 0'0000'000000'007777'000000;
static constexpr u64 MASK_R1_B5 = 0'0000'000000'700000'000000, MASK_R1_S5 = 0'0000'000000'077777'000000;

static constexpr u64 MASK_R2_B1 = 0'0000'000000'000000'777770, MASK_R2_S1 = 0'0000'000000'000000'000007;
static constexpr u64 MASK_R2_B2 = 0'0000'000000'000000'777700, MASK_R2_S2 = 0'0000'000000'000000'000077;
static constexpr u64 MASK_R2_B3 = 0'0000'000000'000000'777000, MASK_R2_S3 = 0'0000'000000'000000'000777;
static constexpr u64 MASK_R2_B4 = 0'0000'000000'000000'770000, MASK_R2_S4 = 0'0000'000000'000000'007777;
static constexpr u64 MASK_R2_B5 = 0'0000'000000'000000'700000, MASK_R2_S5 = 0'0000'000000'000000'077777;


static constexpr u64 MASK_R01_NT = MASK_R0_NT & MASK_R1_NT;
static constexpr u64 MASK_R01_B1 = MASK_R0_B1 | MASK_R1_B1, MASK_R01_S1 = MASK_R0_S1 | MASK_R1_S1;
static constexpr u64 MASK_R01_B2 = MASK_R0_B2 | MASK_R1_B2, MASK_R01_S2 = MASK_R0_S2 | MASK_R1_S2;
static constexpr u64 MASK_R01_B3 = MASK_R0_B3 | MASK_R1_B3, MASK_R01_S3 = MASK_R0_S3 | MASK_R1_S3;
static constexpr u64 MASK_R01_B4 = MASK_R0_B4 | MASK_R1_B4, MASK_R01_S4 = MASK_R0_S4 | MASK_R1_S4;
static constexpr u64 MASK_R01_B5 = MASK_R0_B5 | MASK_R1_B5, MASK_R01_S5 = MASK_R0_S5 | MASK_R1_S5;

static constexpr u64 MASK_R12_NT = MASK_R1_NT & MASK_R2_NT;
static constexpr u64 MASK_R12_B1 = MASK_R1_B1 | MASK_R2_B1, MASK_R12_S1 = MASK_R1_S1 | MASK_R2_S1;
static constexpr u64 MASK_R12_B2 = MASK_R1_B2 | MASK_R2_B2, MASK_R12_S2 = MASK_R1_S2 | MASK_R2_S2;
static constexpr u64 MASK_R12_B3 = MASK_R1_B3 | MASK_R2_B3, MASK_R12_S3 = MASK_R1_S3 | MASK_R2_S3;
static constexpr u64 MASK_R12_B4 = MASK_R1_B4 | MASK_R2_B4, MASK_R12_S4 = MASK_R1_S4 | MASK_R2_S4;
static constexpr u64 MASK_R12_B5 = MASK_R1_B5 | MASK_R2_B5, MASK_R12_S5 = MASK_R1_S5 | MASK_R2_S5;




#ifdef USE_TRIMMED_ROTATIONS

static constexpr u64 MASKS_RX_NT[5] = {
        MASK_R0_NT, MASK_R1_NT, MASK_R2_NT, MASK_R01_NT, MASK_R12_NT
};
static constexpr u64 MASKS_RX_BXSX_PAIR[50] = {
        MASK_R0_B1, MASK_R0_S1,
        MASK_R0_B2, MASK_R0_S2,
        MASK_R0_B3, MASK_R0_S3,
        MASK_R0_B4, MASK_R0_S4,
        MASK_R0_B5, MASK_R0_S5,

        MASK_R1_B1, MASK_R1_S1,
        MASK_R1_B2, MASK_R1_S2,
        MASK_R1_B3, MASK_R1_S3,
        MASK_R1_B4, MASK_R1_S4,
        MASK_R1_B5, MASK_R1_S5,

        MASK_R2_B1, MASK_R2_S1,
        MASK_R2_B2, MASK_R2_S2,
        MASK_R2_B3, MASK_R2_S3,
        MASK_R2_B4, MASK_R2_S4,
        MASK_R2_B5, MASK_R2_S5,

        MASK_R01_B1, MASK_R01_S1,
        MASK_R01_B2, MASK_R01_S2,
        MASK_R01_B3, MASK_R01_S3,
        MASK_R01_B4, MASK_R01_S4,
        MASK_R01_B5, MASK_R01_S5,

        MASK_R12_B1, MASK_R12_S1,
        MASK_R12_B2, MASK_R12_S2,
        MASK_R12_B3, MASK_R12_S3,
        MASK_R12_B4, MASK_R12_S4,
        MASK_R12_B5, MASK_R12_S5,
};


HD __attribute__((noinline)) void funcB1(B1B2& board, C u8* masks) {
    C u64* BXSXPair = &MASKS_RX_BXSX_PAIR[masks[1]];
    board.b1 = board.b1 & MASKS_RX_NT[masks[0]]
               | (board.b1 & *BXSXPair) >>  masks[2]
               | (board.b1 & *(BXSXPair + 1)) << (18 - masks[2]);
}

HD __attribute__((noinline)) void funcB2(B1B2& board, C u8* masks) {
    C u64* BXSXPair = &MASKS_RX_BXSX_PAIR[masks[1]];
    board.b2 = board.b2 & MASKS_RX_NT[masks[0]]
               | (board.b2 & *BXSXPair) >>  masks[2]
               | (board.b2 & *(BXSXPair + 1)) << (18 - masks[2]);
}

static constexpr u32 RXXX_PER = 3;
static constexpr u8 RXXX_MASKS[RXXX_PER * 25] = {
        0,  0,  3,
        0,  2,  6,
        0,  4,  9,
        0,  6, 12,
        0,  8, 15,
        1, 10,  3,
        1, 12,  6,
        1, 14,  9,
        1, 16, 12,
        1, 18, 15,
        2, 20,  3,
        2, 22,  6,
        2, 24,  9,
        2, 26, 12,
        2, 28, 15,
        3, 30,  3,
        3, 32,  6,
        3, 34,  9,
        3, 36, 12,
        3, 38, 15,
        4, 40,  3,
        4, 42,  6,
        4, 44,  9,
        4, 46, 12,
        4, 48, 15,
};


PERM_MACRO(R01) { funcB1(board, &RXXX_MASKS[RXXX_PER *  0]); } //  0
PERM_MACRO(R02) { funcB1(board, &RXXX_MASKS[RXXX_PER *  1]); } //  1
PERM_MACRO(R03) { funcB1(board, &RXXX_MASKS[RXXX_PER *  2]); } //  2
PERM_MACRO(R04) { funcB1(board, &RXXX_MASKS[RXXX_PER *  3]); } //  3
PERM_MACRO(R05) { funcB1(board, &RXXX_MASKS[RXXX_PER *  4]); } //  4
PERM_MACRO(R11) { funcB1(board, &RXXX_MASKS[RXXX_PER *  5]); } //  5
PERM_MACRO(R12) { funcB1(board, &RXXX_MASKS[RXXX_PER *  6]); } //  6
PERM_MACRO(R13) { funcB1(board, &RXXX_MASKS[RXXX_PER *  7]); } //  7
PERM_MACRO(R14) { funcB1(board, &RXXX_MASKS[RXXX_PER *  8]); } //  8
PERM_MACRO(R15) { funcB1(board, &RXXX_MASKS[RXXX_PER *  9]); } //  9
PERM_MACRO(R21) { funcB1(board, &RXXX_MASKS[RXXX_PER * 10]); } // 10
PERM_MACRO(R22) { funcB1(board, &RXXX_MASKS[RXXX_PER * 11]); } // 11
PERM_MACRO(R23) { funcB1(board, &RXXX_MASKS[RXXX_PER * 12]); } // 12
PERM_MACRO(R24) { funcB1(board, &RXXX_MASKS[RXXX_PER * 13]); } // 13
PERM_MACRO(R25) { funcB1(board, &RXXX_MASKS[RXXX_PER * 14]); } // 14

PERM_MACRO(R31) { funcB2(board, &RXXX_MASKS[RXXX_PER *  0]); } // 15
PERM_MACRO(R32) { funcB2(board, &RXXX_MASKS[RXXX_PER *  1]); } // 16
PERM_MACRO(R33) { funcB2(board, &RXXX_MASKS[RXXX_PER *  2]); } // 17
PERM_MACRO(R34) { funcB2(board, &RXXX_MASKS[RXXX_PER *  3]); } // 18
PERM_MACRO(R35) { funcB2(board, &RXXX_MASKS[RXXX_PER *  4]); } // 19
PERM_MACRO(R41) { funcB2(board, &RXXX_MASKS[RXXX_PER *  5]); } // 20
PERM_MACRO(R42) { funcB2(board, &RXXX_MASKS[RXXX_PER *  6]); } // 21
PERM_MACRO(R43) { funcB2(board, &RXXX_MASKS[RXXX_PER *  7]); } // 22
PERM_MACRO(R44) { funcB2(board, &RXXX_MASKS[RXXX_PER *  8]); } // 23
PERM_MACRO(R45) { funcB2(board, &RXXX_MASKS[RXXX_PER *  9]); } // 24
PERM_MACRO(R51) { funcB2(board, &RXXX_MASKS[RXXX_PER * 10]); } // 25
PERM_MACRO(R52) { funcB2(board, &RXXX_MASKS[RXXX_PER * 11]); } // 26
PERM_MACRO(R53) { funcB2(board, &RXXX_MASKS[RXXX_PER * 12]); } // 27
PERM_MACRO(R54) { funcB2(board, &RXXX_MASKS[RXXX_PER * 13]); } // 28
PERM_MACRO(R55) { funcB2(board, &RXXX_MASKS[RXXX_PER * 14]); } // 29

PERM_MACRO(R011) { board.addFatX(1); funcB1(board, &RXXX_MASKS[RXXX_PER * 15]); }
PERM_MACRO(R012) { board.addFatX(2); funcB1(board, &RXXX_MASKS[RXXX_PER * 16]); }
PERM_MACRO(R013) { board.addFatX(3); funcB1(board, &RXXX_MASKS[RXXX_PER * 17]); }
PERM_MACRO(R014) { board.addFatX(4); funcB1(board, &RXXX_MASKS[RXXX_PER * 18]); }
PERM_MACRO(R015) { board.addFatX(5); funcB1(board, &RXXX_MASKS[RXXX_PER * 19]); }
PERM_MACRO(R121) { board.addFatX(1); funcB1(board, &RXXX_MASKS[RXXX_PER * 20]); }
PERM_MACRO(R122) { board.addFatX(2); funcB1(board, &RXXX_MASKS[RXXX_PER * 21]); }
PERM_MACRO(R123) { board.addFatX(3); funcB1(board, &RXXX_MASKS[RXXX_PER * 22]); }
PERM_MACRO(R124) { board.addFatX(4); funcB1(board, &RXXX_MASKS[RXXX_PER * 23]); }
PERM_MACRO(R125) { board.addFatX(5); funcB1(board, &RXXX_MASKS[RXXX_PER * 24]); }

PERM_MACRO(R231) { board.addFatX(1); R21(board); R31(board); }
PERM_MACRO(R232) { board.addFatX(2); R22(board); R32(board); }
PERM_MACRO(R233) { board.addFatX(3); R23(board); R33(board); }
PERM_MACRO(R234) { board.addFatX(4); R24(board); R34(board); }
PERM_MACRO(R235) { board.addFatX(5); R25(board); R35(board); }

PERM_MACRO(R341) { board.addFatX(1); funcB2(board, &RXXX_MASKS[45]); }
PERM_MACRO(R342) { board.addFatX(2); funcB2(board, &RXXX_MASKS[48]); }
PERM_MACRO(R343) { board.addFatX(3); funcB2(board, &RXXX_MASKS[51]); }
PERM_MACRO(R344) { board.addFatX(4); funcB2(board, &RXXX_MASKS[54]); }
PERM_MACRO(R345) { board.addFatX(5); funcB2(board, &RXXX_MASKS[57]); }

PERM_MACRO(R451) { board.addFatX(1); funcB2(board, &RXXX_MASKS[60]); }
PERM_MACRO(R452) { board.addFatX(2); funcB2(board, &RXXX_MASKS[63]); }
PERM_MACRO(R453) { board.addFatX(3); funcB2(board, &RXXX_MASKS[66]); }
PERM_MACRO(R454) { board.addFatX(4); funcB2(board, &RXXX_MASKS[69]); }
PERM_MACRO(R455) { board.addFatX(5); funcB2(board, &RXXX_MASKS[72]); }




#else

PERM_MACRO(R01) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B1) >>  3 | (board.b1 & MASK_R0_S1) << 15; } //  0
PERM_MACRO(R02) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B2) >>  6 | (board.b1 & MASK_R0_S2) << 12; } //  1
PERM_MACRO(R03) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B3) >>  9 | (board.b1 & MASK_R0_S3) <<  9; } //  2
PERM_MACRO(R04) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B4) >> 12 | (board.b1 & MASK_R0_S4) <<  6; } //  3
PERM_MACRO(R05) { board.b1 = board.b1 & MASK_R0_NT | (board.b1 & MASK_R0_B5) >> 15 | (board.b1 & MASK_R0_S5) <<  3; } //  4
PERM_MACRO(R11) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B1) >>  3 | (board.b1 & MASK_R1_S1) << 15; } //  5
PERM_MACRO(R12) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B2) >>  6 | (board.b1 & MASK_R1_S2) << 12; } //  6
PERM_MACRO(R13) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B3) >>  9 | (board.b1 & MASK_R1_S3) <<  9; } //  7
PERM_MACRO(R14) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B4) >> 12 | (board.b1 & MASK_R1_S4) <<  6; } //  8
PERM_MACRO(R15) { board.b1 = board.b1 & MASK_R1_NT | (board.b1 & MASK_R1_B5) >> 15 | (board.b1 & MASK_R1_S5) <<  3; } //  9
PERM_MACRO(R21) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B1) >>  3 | (board.b1 & MASK_R2_S1) << 15; } // 10
PERM_MACRO(R22) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B2) >>  6 | (board.b1 & MASK_R2_S2) << 12; } // 11
PERM_MACRO(R23) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B3) >>  9 | (board.b1 & MASK_R2_S3) <<  9; } // 12
PERM_MACRO(R24) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B4) >> 12 | (board.b1 & MASK_R2_S4) <<  6; } // 13
PERM_MACRO(R25) { board.b1 = board.b1 & MASK_R2_NT | (board.b1 & MASK_R2_B5) >> 15 | (board.b1 & MASK_R2_S5) <<  3; } // 14

PERM_MACRO(R31) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B1) >>  3 | (board.b2 & MASK_R0_S1) << 15; } // 15
PERM_MACRO(R32) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B2) >>  6 | (board.b2 & MASK_R0_S2) << 12; } // 16
PERM_MACRO(R33) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B3) >>  9 | (board.b2 & MASK_R0_S3) <<  9; } // 17
PERM_MACRO(R34) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B4) >> 12 | (board.b2 & MASK_R0_S4) <<  6; } // 18
PERM_MACRO(R35) { board.b2 = board.b2 & MASK_R0_NT | (board.b2 & MASK_R0_B5) >> 15 | (board.b2 & MASK_R0_S5) <<  3; } // 19
PERM_MACRO(R41) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B1) >>  3 | (board.b2 & MASK_R1_S1) << 15; } // 20
PERM_MACRO(R42) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B2) >>  6 | (board.b2 & MASK_R1_S2) << 12; } // 21
PERM_MACRO(R43) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B3) >>  9 | (board.b2 & MASK_R1_S3) <<  9; } // 22
PERM_MACRO(R44) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B4) >> 12 | (board.b2 & MASK_R1_S4) <<  6; } // 23
PERM_MACRO(R45) { board.b2 = board.b2 & MASK_R1_NT | (board.b2 & MASK_R1_B5) >> 15 | (board.b2 & MASK_R1_S5) <<  3; } // 24
PERM_MACRO(R51) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B1) >>  3 | (board.b2 & MASK_R2_S1) << 15; } // 25
PERM_MACRO(R52) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B2) >>  6 | (board.b2 & MASK_R2_S2) << 12; } // 26
PERM_MACRO(R53) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B3) >>  9 | (board.b2 & MASK_R2_S3) <<  9; } // 27
PERM_MACRO(R54) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B4) >> 12 | (board.b2 & MASK_R2_S4) <<  6; } // 28
PERM_MACRO(R55) { board.b2 = board.b2 & MASK_R2_NT | (board.b2 & MASK_R2_B5) >> 15 | (board.b2 & MASK_R2_S5) <<  3; } // 29


PERM_MACRO(R011) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B1) >> 3 | (board.b1 & MASK_R01_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R012) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B2) >> 6 | (board.b1 & MASK_R01_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R013) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B3) >> 9 | (board.b1 & MASK_R01_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R014) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B4) >> 12 | (board.b1 & MASK_R01_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R015) {
    board.b1 = board.b1 & MASK_R01_NT | (board.b1 & MASK_R01_B5) >> 15 | (board.b1 & MASK_R01_S5) << 3;
    board.addFatX(5);
}

PERM_MACRO(R121) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B1) >> 3 | (board.b1 & MASK_R12_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R122) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B2) >> 6 | (board.b1 & MASK_R12_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R123) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B3) >> 9 | (board.b1 & MASK_R12_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R124) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B4) >> 12 | (board.b1 & MASK_R12_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R125) {
    board.b1 = board.b1 & MASK_R12_NT | (board.b1 & MASK_R12_B5) >> 15 | (board.b1 & MASK_R12_S5) << 3;
    board.addFatX(5);
}

PERM_MACRO(R231) {
    R21(board);
    R31(board);
    board.addFatX(1);
}
PERM_MACRO(R232) {
    R22(board);
    R32(board);
    board.addFatX(2);
}
PERM_MACRO(R233) {
    R23(board);
    R33(board);
    board.addFatX(3);
}
PERM_MACRO(R234) {
    R24(board);
    R34(board);
    board.addFatX(4);
}
PERM_MACRO(R235) {
    R25(board);
    R35(board);
    board.addFatX(5);
}

PERM_MACRO(R341) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B1) >> 3 | (board.b2 & MASK_R01_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R342) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B2) >> 6 | (board.b2 & MASK_R01_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R343) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B3) >> 9 | (board.b2 & MASK_R01_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R344) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B4) >> 12 | (board.b2 & MASK_R01_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R345) {
    board.b2 = board.b2 & MASK_R01_NT | (board.b2 & MASK_R01_B5) >> 15 | (board.b2 & MASK_R01_S5) << 3;
    board.addFatX(5);
}

PERM_MACRO(R451) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B1) >> 3 | (board.b2 & MASK_R12_S1) << 15;
    board.addFatX(1);
}
PERM_MACRO(R452) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B2) >> 6 | (board.b2 & MASK_R12_S2) << 12;
    board.addFatX(2);
}
PERM_MACRO(R453) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B3) >> 9 | (board.b2 & MASK_R12_S3) << 9;
    board.addFatX(3);
}
PERM_MACRO(R454) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B4) >> 12 | (board.b2 & MASK_R12_S4) << 6;
    board.addFatX(4);
}
PERM_MACRO(R455) {
    board.b2 = board.b2 & MASK_R12_NT | (board.b2 & MASK_R12_B5) >> 15 | (board.b2 & MASK_R12_S5) << 3;
    board.addFatX(5);
}

#endif


