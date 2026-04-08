#pragma once
// code/rotations_cuda.hpp

#include "board.hpp"


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
    MU __device__ void R_X_X_copy(const B1B2* src, B1B2* dest, u32 idx);

    // C_X_X SECTION
    MU __constant__ extern u64 CUDA_COL_MASKS[6];
    MU __constant__ extern u32 CUDA_COL_OFF1[9];
    struct PrecomputedIdx { u8 div5, mod5; };
    MU __constant__ extern PrecomputedIdx PRECOMPUTED_IDX[30];
    MU __device__ void C_X_X(B1B2& board, int idx);
    MU __device__ void C_X_X_copy(const B1B2* src, B1B2* dest, u32 idx);
}
#endif
