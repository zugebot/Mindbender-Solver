#pragma once

#include "board.hpp"


#define PERM_MACRO(name) __host__ __device__ void name(B1B2 &board)


// permutations that are for rows
/**  0*/ PERM_MACRO(R01); /**  1*/ PERM_MACRO(R02); /**  2*/ PERM_MACRO(R03); /**  3*/ PERM_MACRO(R04); /**  4*/ PERM_MACRO(R05);
/**  5*/ PERM_MACRO(R11); /**  6*/ PERM_MACRO(R12); /**  7*/ PERM_MACRO(R13); /**  8*/ PERM_MACRO(R14); /**  9*/ PERM_MACRO(R15);
/** 10*/ PERM_MACRO(R21); /** 11*/ PERM_MACRO(R22); /** 12*/ PERM_MACRO(R23); /** 13*/ PERM_MACRO(R24); /** 14*/ PERM_MACRO(R25);
/** 15*/ PERM_MACRO(R31); /** 16*/ PERM_MACRO(R32); /** 17*/ PERM_MACRO(R33); /** 18*/ PERM_MACRO(R34); /** 19*/ PERM_MACRO(R35);
/** 20*/ PERM_MACRO(R41); /** 21*/ PERM_MACRO(R42); /** 22*/ PERM_MACRO(R43); /** 23*/ PERM_MACRO(R44); /** 24*/ PERM_MACRO(R45);
/** 25*/ PERM_MACRO(R51); /** 26*/ PERM_MACRO(R52); /** 27*/ PERM_MACRO(R53); /** 28*/ PERM_MACRO(R54); /** 29*/ PERM_MACRO(R55);


// permutations that are for columns
/** 32*/ PERM_MACRO(C01); /** 33*/ PERM_MACRO(C02); /** 34*/ PERM_MACRO(C03); /** 35*/ PERM_MACRO(C04); /** 36*/ PERM_MACRO(C05);
/** 37*/ PERM_MACRO(C11); /** 38*/ PERM_MACRO(C12); /** 39*/ PERM_MACRO(C13); /** 40*/ PERM_MACRO(C14); /** 41*/ PERM_MACRO(C15);
/** 42*/ PERM_MACRO(C21); /** 43*/ PERM_MACRO(C22); /** 44*/ PERM_MACRO(C23); /** 45*/ PERM_MACRO(C24); /** 46*/ PERM_MACRO(C25);
/** 47*/ PERM_MACRO(C31); /** 48*/ PERM_MACRO(C32); /** 49*/ PERM_MACRO(C33); /** 50*/ PERM_MACRO(C34); /** 51*/ PERM_MACRO(C35);
/** 52*/ PERM_MACRO(C41); /** 53*/ PERM_MACRO(C42); /** 54*/ PERM_MACRO(C43); /** 55*/ PERM_MACRO(C44); /** 56*/ PERM_MACRO(C45);
/** 57*/ PERM_MACRO(C51); /** 58*/ PERM_MACRO(C52); /** 59*/ PERM_MACRO(C53); /** 60*/ PERM_MACRO(C54); /** 61*/ PERM_MACRO(C55);


// permutations that are special for fat boards
PERM_MACRO(R011); PERM_MACRO(R012); PERM_MACRO(R013); PERM_MACRO(R014); PERM_MACRO(R015);
PERM_MACRO(R121); PERM_MACRO(R122); PERM_MACRO(R123); PERM_MACRO(R124); PERM_MACRO(R125);
PERM_MACRO(R231); PERM_MACRO(R232); PERM_MACRO(R233); PERM_MACRO(R234); PERM_MACRO(R235);
PERM_MACRO(R341); PERM_MACRO(R342); PERM_MACRO(R343); PERM_MACRO(R344); PERM_MACRO(R345);
PERM_MACRO(R451); PERM_MACRO(R452); PERM_MACRO(R453); PERM_MACRO(R454); PERM_MACRO(R455);
PERM_MACRO(C011); PERM_MACRO(C012); PERM_MACRO(C013); PERM_MACRO(C014); PERM_MACRO(C015);
PERM_MACRO(C121); PERM_MACRO(C122); PERM_MACRO(C123); PERM_MACRO(C124); PERM_MACRO(C125);
PERM_MACRO(C231); PERM_MACRO(C232); PERM_MACRO(C233); PERM_MACRO(C234); PERM_MACRO(C235);
PERM_MACRO(C341); PERM_MACRO(C342); PERM_MACRO(C343); PERM_MACRO(C344); PERM_MACRO(C345);
PERM_MACRO(C451); PERM_MACRO(C452); PERM_MACRO(C453); PERM_MACRO(C454); PERM_MACRO(C455);


struct ActStruct {
    Action action{};
    std::array<char, 4> name{};
    u8 index{};
    u8 isColNotFat{};
    u8 tillNext{};
    u8 tillLast{};

    MU ActStruct() = default;

    MU ActStruct(C Action theAction, C u8 theIndex, C u8 theIsColNotFat,
                 C u8 theTillNext, C u8 theTillLast, C char* theName) {
        action = theAction;
        memcpy(name.data(), theName, 4 * sizeof(char));
        index = theIndex;
        isColNotFat = theIsColNotFat;
        tillNext = theTillNext;
        tillLast = theTillLast;
    }
};


#ifdef USE_CUDA
namespace my_cuda {
    // ACT_STRUCT_GPU SECTION
    struct ActStructGPU {
        Action action{};
        MU u8 index{}, isColNotFat{}, tillNext{}, tillLast{};
    };
    MU __constant__ extern ActStructGPU allActStructListGPU[114];

    // R_X_X SECTION
    struct R_X_X_data {
        MU u64 maskRB, maskRS;
        MU i32 rowNum, shiftR, shiftL;
    };
    MU __constant__ extern u64 CUDA_ROW_MASKS[3];
    MU __constant__ extern R_X_X_data CUDA_RXX_DATA[15];
    MU __device__ void R_X_X(B1B2& board, int idx);
    MU __device__ void R_X_X_copy(C B1B2* src, B1B2* dest, u32 idx);

    // C_X_X SECTION
    MU __constant__ extern u64 CUDA_COL_MASKS[6];
    MU __constant__ extern u32 CUDA_COL_OFF1[9];
    struct PrecomputedIdx { u8 div5, mod5; };
    MU __constant__ extern PrecomputedIdx PRECOMPUTED_IDX[30];
    MU __device__ void C_X_X(B1B2& board, int idx);
    MU __device__ void C_X_X_copy(C B1B2* src, B1B2* dest, u32 idx);
}
#endif


MU extern ActStruct allActStructList[114];
MU extern u8 fatActionsIndexes[25][48];

extern void applyMoves(Board& board, C Memory& memory);
extern void applyFatMoves(Board& board, C Memory& memory);

extern Board makeBoardWithMoves(C Board& board, C Memory& memory);
extern Board makeBoardWithFatMoves(C Board& board, C Memory& memory);




