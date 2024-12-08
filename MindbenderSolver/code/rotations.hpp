#pragma once

#include "board.hpp"


#define PERM_MACRO(name) __host__ __device__ void name(B1B2 &board)


// permutations that are for rows
PERM_MACRO(R_0_1); PERM_MACRO(R_0_2); PERM_MACRO(R_0_3); PERM_MACRO(R_0_4); PERM_MACRO(R_0_5);
PERM_MACRO(R_1_1); PERM_MACRO(R_1_2); PERM_MACRO(R_1_3); PERM_MACRO(R_1_4); PERM_MACRO(R_1_5);
PERM_MACRO(R_2_1); PERM_MACRO(R_2_2); PERM_MACRO(R_2_3); PERM_MACRO(R_2_4); PERM_MACRO(R_2_5);
PERM_MACRO(R_3_1); PERM_MACRO(R_3_2); PERM_MACRO(R_3_3); PERM_MACRO(R_3_4); PERM_MACRO(R_3_5);
PERM_MACRO(R_4_1); PERM_MACRO(R_4_2); PERM_MACRO(R_4_3); PERM_MACRO(R_4_4); PERM_MACRO(R_4_5);
PERM_MACRO(R_5_1); PERM_MACRO(R_5_2); PERM_MACRO(R_5_3); PERM_MACRO(R_5_4); PERM_MACRO(R_5_5);


// permutations that are for columns
PERM_MACRO(C_0_1); PERM_MACRO(C_0_2); PERM_MACRO(C_0_3); PERM_MACRO(C_0_4); PERM_MACRO(C_0_5);
PERM_MACRO(C_1_1); PERM_MACRO(C_1_2); PERM_MACRO(C_1_3); PERM_MACRO(C_1_4); PERM_MACRO(C_1_5);
PERM_MACRO(C_2_1); PERM_MACRO(C_2_2); PERM_MACRO(C_2_3); PERM_MACRO(C_2_4); PERM_MACRO(C_2_5);
PERM_MACRO(C_3_1); PERM_MACRO(C_3_2); PERM_MACRO(C_3_3); PERM_MACRO(C_3_4); PERM_MACRO(C_3_5);
PERM_MACRO(C_4_1); PERM_MACRO(C_4_2); PERM_MACRO(C_4_3); PERM_MACRO(C_4_4); PERM_MACRO(C_4_5);
PERM_MACRO(C_5_1); PERM_MACRO(C_5_2); PERM_MACRO(C_5_3); PERM_MACRO(C_5_4); PERM_MACRO(C_5_5);


// permutations that are special for fat boards
PERM_MACRO(R_01_1); PERM_MACRO(R_01_2); PERM_MACRO(R_01_3); PERM_MACRO(R_01_4); PERM_MACRO(R_01_5);
PERM_MACRO(R_12_1); PERM_MACRO(R_12_2); PERM_MACRO(R_12_3); PERM_MACRO(R_12_4); PERM_MACRO(R_12_5);
PERM_MACRO(R_23_1); PERM_MACRO(R_23_2); PERM_MACRO(R_23_3); PERM_MACRO(R_23_4); PERM_MACRO(R_23_5);
PERM_MACRO(R_34_1); PERM_MACRO(R_34_2); PERM_MACRO(R_34_3); PERM_MACRO(R_34_4); PERM_MACRO(R_34_5);
PERM_MACRO(R_45_1); PERM_MACRO(R_45_2); PERM_MACRO(R_45_3); PERM_MACRO(R_45_4); PERM_MACRO(R_45_5);
PERM_MACRO(C_01_1); PERM_MACRO(C_01_2); PERM_MACRO(C_01_3); PERM_MACRO(C_01_4); PERM_MACRO(C_01_5);
PERM_MACRO(C_12_1); PERM_MACRO(C_12_2); PERM_MACRO(C_12_3); PERM_MACRO(C_12_4); PERM_MACRO(C_12_5);
PERM_MACRO(C_23_1); PERM_MACRO(C_23_2); PERM_MACRO(C_23_3); PERM_MACRO(C_23_4); PERM_MACRO(C_23_5);
PERM_MACRO(C_34_1); PERM_MACRO(C_34_2); PERM_MACRO(C_34_3); PERM_MACRO(C_34_4); PERM_MACRO(C_34_5);
PERM_MACRO(C_45_1); PERM_MACRO(C_45_2); PERM_MACRO(C_45_3); PERM_MACRO(C_45_4); PERM_MACRO(C_45_5);


typedef void (*Action)(B1B2 &);


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

    MU __constant__ extern ActStructGPU allActStructListGPU[110];

    // R_X_X SECTION

    struct R_X_X_data {
        MU u64 maskRB, maskRS;
        MU i32 rowNum, shiftR, shiftL;
    };

    MU __constant__ extern u64 CUDA_ROW_MASKS[3];

    MU __constant__ extern R_X_X_data CUDA_RXX_DATA[15];

    MU __device__ void R_X_X(B1B2& board, int idx);

    // C_X_X SECTION


    MU __constant__ extern u64 CUDA_COL_MASKS[6];

    MU __constant__ extern u32 CUDA_COL_OFF1[9];

    struct PrecomputedIdx { u8 div5, mod5; };

    MU __constant__ extern PrecomputedIdx PRECOMPUTED_IDX[30];

    MU __device__ void C_X_X(B1B2& board, int idx);


}

#endif



MU extern ActStruct allActStructList[110];

MU extern u8 fatActionsIndexes[25][48];

extern void applyMoves(Board& board, C Memory& memory);
extern void applyFatMoves(Board& board, C Memory& memory);

extern Board makeBoardWithMoves(C Board& board, C Memory& memory);
extern Board makeBoardWithFatMoves(C Board& board, C Memory& memory);




