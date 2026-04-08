// code/rotations_cuda.cpp
#include "rotations_cuda.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


#ifdef USE_CUDA

namespace my_cuda {

    MU __constant__ u64 CUDA_ROW_MASKS[3] = {
            0'1777'000000'777777'777777,
            0'1777'777777'000000'777777,
            0'1777'777777'777777'000000,
    };

    MU __constant__ R_X_X_data CUDA_RXX_DATA[15] = {
            {0'0000'777770'000000'000000, 0'0000'000007'000000'000000, 0,  3, 15},
            {0'0000'777700'000000'000000, 0'0000'000077'000000'000000, 0,  6, 12},
            {0'0000'777000'000000'000000, 0'0000'000777'000000'000000, 0,  9,  9},
            {0'0000'770000'000000'000000, 0'0000'007777'000000'000000, 0, 12,  6},
            {0'0000'700000'000000'000000, 0'0000'077777'000000'000000, 0, 15,  3},
            {0'0000'000000'777770'000000, 0'0000'000000'000007'000000, 1,  3, 15},
            {0'0000'000000'777700'000000, 0'0000'000000'000077'000000, 1,  6, 12},
            {0'0000'000000'777000'000000, 0'0000'000000'000777'000000, 1,  9,  9},
            {0'0000'000000'770000'000000, 0'0000'000000'007777'000000, 1, 12,  6},
            {0'0000'000000'700000'000000, 0'0000'000000'077777'000000, 1, 15,  3},
            {0'0000'000000'000000'777770, 0'0000'000000'000000'000007, 2,  3, 15},
            {0'0000'000000'000000'777700, 0'0000'000000'000000'000077, 2,  6, 12},
            {0'0000'000000'000000'777000, 0'0000'000000'000000'000777, 2,  9,  9},
            {0'0000'000000'000000'770000, 0'0000'000000'000000'007777, 2, 12,  6},
            {0'0000'000000'000000'700000, 0'0000'000000'000000'077777, 2, 15,  3},
    };

    MU __device__ void R_X_X(B1B2& board, i32 idx) {

        const bool is_R3_to_r5 = idx >= 15;
        R_X_X_data data = CUDA_RXX_DATA[idx - 15 * is_R3_to_r5];

        u64* addr = &board.b1 + is_R3_to_r5;
        *addr = (*addr & CUDA_ROW_MASKS[data.rowNum])
                | ((*addr & data.maskRB) >> data.shiftR)
                | ((*addr & data.maskRS) << data.shiftL);
    }


    // New R_X_X function that copies and modifies
    MU __device__ void R_X_X_copy(const B1B2* src, B1B2* dest, u32 idx) {
        // Copy the source board to destination
        *dest = *src;

        // Determine if we are operating on R3 to R5 based on idx
        // 0b11111111111111111000000000000000
        bool is_R3_to_r5 = idx >= 15;

        // Fetch the appropriate R_X_X_data
        R_X_X_data data = CUDA_RXX_DATA[idx - 15 * is_R3_to_r5];

        // Compute the address based on is_R3_to_r5
        u64* addr = &dest->b1 + is_R3_to_r5;

        // Apply the permutation on the destination board
        *addr = (*addr & CUDA_ROW_MASKS[data.rowNum])
                | ((*addr & data.maskRB) >> data.shiftR)
                | ((*addr & data.maskRS) << data.shiftL);
    }


    MU __constant__ u64 CUDA_COL_MASKS[6] = {
            0'700000'700000'700000,
            0'070000'070000'070000,
            0'007000'007000'007000,
            0'000700'000700'000700,
            0'000070'000070'000070,
            0'000007'000007'000007,
    };

    MU __constant__ u32 CUDA_COL_OFF1[9] = {36, 18, 0, 36, 18, 36, 54, 18, 36};

    MU __constant__ PrecomputedIdx PRECOMPUTED_IDX[30] = {
            {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},
            {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},
            {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4},
            {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},
            {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4},
            {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4},
    };

    MU __device__ void C_X_X(B1B2 &board, i32 idx) {
        const u8 mod5 = PRECOMPUTED_IDX[idx].mod5;
        const u64 C_MASK_X = CUDA_COL_MASKS[PRECOMPUTED_IDX[idx].div5];


        const u64 vars[2] = {board.b1 & C_MASK_X, board.b2 & C_MASK_X};

        const i32 idx1 = (mod5 <= 2) ? 1 : 0;
        const u64 *ptr1 = &vars[idx1];
        const u64 *ptr2 = &vars[1 - idx1];

        board.b1 = ((*ptr1 << CUDA_COL_OFF1[mod5] | *ptr2 >>
                                                            (CUDA_COL_OFF1 + 4)[mod5]) & C_MASK_X) | board.b1 & ~C_MASK_X;
        board.b2 = ((*ptr2 << CUDA_COL_OFF1[mod5] | *ptr1 >>
                                                            (CUDA_COL_OFF1 + 4)[mod5]) & C_MASK_X) | board.b2 & ~C_MASK_X;
    }


    // New C_X_X function that copies and modifies
    MU __device__ void C_X_X_copy(const B1B2* src, B1B2* dest, u32 idx) {
        // Copy the source board to destination
        *dest = *src;

        // Retrieve precomputed indices
        u8 mod5 = PRECOMPUTED_IDX[idx].mod5;
        u64 C_MASK_X = CUDA_COL_MASKS[PRECOMPUTED_IDX[idx].div5];

        // Extract relevant parts of the board
        u64 vars[2] = {dest->b1 & C_MASK_X, dest->b2 & C_MASK_X};

        // Determine indices based on mod5
        i32 idx1 = (mod5 <= 2) ? 1 : 0;
        u64* ptr1 = &vars[idx1];
        u64* ptr2 = &vars[1 - idx1];

        // Perform the permutation on the destination board
        dest->b1 = ((*ptr1 << CUDA_COL_OFF1[mod5] | *ptr2 >>
                                                            CUDA_COL_OFF1[mod5 + 4]) & C_MASK_X) | (dest->b1 & ~C_MASK_X);
        dest->b2 = ((*ptr2 << CUDA_COL_OFF1[mod5] | *ptr1 >>
                                                            CUDA_COL_OFF1[mod5 + 4]) & C_MASK_X) | (dest->b2 & ~C_MASK_X);
    }


    MU __constant__ ActStructGPU allActStructListGPU[114] = {
            // {action, index, isColNotFat, tillNext, tillLast}
            {R01,    0, 2, 5, 0}, {R02,    1, 2, 4, 1}, {R03,    2, 2, 3, 2}, {R04,    3, 2, 2, 3}, {R05,    4, 2, 1, 4},
            {R11,    5, 2, 5, 0}, {R12,    6, 2, 4, 1}, {R13,    7, 2, 3, 2}, {R14,    8, 2, 2, 3}, {R15,    9, 2, 1, 4},
            {R21,   10, 2, 5, 0}, {R22,   11, 2, 4, 1}, {R23,   12, 2, 3, 2}, {R24,   13, 2, 2, 3}, {R25,   14, 2, 1, 4},
            {R31,   15, 2, 5, 0}, {R32,   16, 2, 4, 1}, {R33,   17, 2, 3, 2}, {R34,   18, 2, 2, 3}, {R35,   19, 2, 1, 4},
            {R41,   20, 2, 5, 0}, {R42,   21, 2, 4, 1}, {R43,   22, 2, 3, 2}, {R44,   23, 2, 2, 3}, {R45,   24, 2, 1, 4},
            {R51,   25, 2, 5, 0}, {R52,   26, 2, 4, 1}, {R53,   27, 2, 3, 2}, {R54,   28, 2, 2, 3}, {R55,   29, 2, 1, 4},
            {nullptr,0, 0, 0, 0},
            {nullptr,0, 0, 0, 0},

            {C01,   32, 1, 5, 0}, {C02,   33, 1, 4, 1}, {C03,   34, 1, 3, 2}, {C04,   35, 1, 2, 3}, {C05,   36, 1, 1, 4},
            {C11,   37, 1, 5, 0}, {C12,   38, 1, 4, 1}, {C13,   39, 1, 3, 2}, {C14,   40, 1, 2, 3}, {C15,   41, 1, 1, 4},
            {C21,   42, 1, 5, 0}, {C22,   43, 1, 4, 1}, {C23,   44, 1, 3, 2}, {C24,   45, 1, 2, 3}, {C25,   46, 1, 1, 4},
            {C31,   47, 1, 5, 0}, {C32,   48, 1, 4, 1}, {C33,   49, 1, 3, 2}, {C34,   50, 1, 2, 3}, {C35,   51, 1, 1, 4},
            {C41,   52, 1, 5, 0}, {C42,   53, 1, 4, 1}, {C43,   54, 1, 3, 2}, {C44,   55, 1, 2, 3}, {C45,   56, 1, 1, 4},
            {C51,   57, 1, 5, 0}, {C52,   58, 1, 4, 1}, {C53,   59, 1, 3, 2}, {C54,   60, 1, 2, 3}, {C55,   61, 1, 1, 4},
            {nullptr,0, 0, 0, 0},
            {nullptr,0, 0, 0, 0},

            {R011,  64, 0, 4, 0}, {R012,  65, 0, 3, 1}, {R013,  66, 0, 2, 2}, {R014,  67, 0, 1, 3}, {R015,  68, 0, 0, 4},
            {R121,  69, 0, 4, 0}, {R122,  70, 0, 3, 1}, {R123,  71, 0, 2, 2}, {R124,  72, 0, 1, 3}, {R125,  73, 0, 0, 4},
            {R231,  74, 0, 4, 0}, {R232,  75, 0, 3, 1}, {R233,  76, 0, 2, 2}, {R234,  77, 0, 1, 3}, {R235,  78, 0, 0, 4},
            {R341,  79, 0, 4, 0}, {R342,  80, 0, 3, 1}, {R343,  81, 0, 2, 2}, {R344,  82, 0, 1, 3}, {R345,  83, 0, 0, 4},
            {R451,  84, 0, 4, 0}, {R452,  85, 0, 3, 1}, {R453,  86, 0, 2, 2}, {R454,  87, 0, 1, 3}, {R455,  88, 0, 0, 4},
            {C011,  89, 0, 4, 0}, {C012,  90, 0, 3, 1}, {C013,  91, 0, 2, 2}, {C014,  92, 0, 1, 3}, {C015,  93, 0, 0, 4},
            {C121,  94, 0, 4, 0}, {C122,  95, 0, 3, 1}, {C123,  96, 0, 2, 2}, {C124,  97, 0, 1, 3}, {C125,  98, 0, 0, 4},
            {C231,  99, 0, 4, 0}, {C232, 100, 0, 3, 1}, {C233, 101, 0, 2, 2}, {C234, 102, 0, 1, 3}, {C235, 103, 0, 0, 4},
            {C341, 104, 0, 4, 0}, {C342, 105, 0, 3, 1}, {C343, 106, 0, 2, 2}, {C344, 107, 0, 1, 3}, {C345, 108, 0, 0, 4},
            {C451, 109, 0, 4, 0}, {C452, 110, 0, 3, 1}, {C453, 111, 0, 2, 2}, {C454, 112, 0, 1, 3}, {C455, 113, 0, 0, 4},
    };

}
#endif