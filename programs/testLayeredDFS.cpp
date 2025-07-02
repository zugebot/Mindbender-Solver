#include <cmath>
#include "include.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif


#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); } \
    } while(0)
#define CUDA_MALLOC(type, ptr, size) CUDA_CHECK(cudaMalloc(ptr, (size) * sizeof(type)))
#define CUDA_FREE(ptr) CUDA_CHECK(cudaFree(ptr))
#define CUDA_DEV_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#define CUDA_LAST_ERR() CUDA_CHECK(cudaGetLastError())
#define CUDA_H_TO_D(type, d_ptr, h_ptr, size) \
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, (size) * sizeof(type), cudaMemcpyHostToDevice))
#define CUDA_D_TO_H(type, d_ptr, h_ptr, size) \
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, (size) * sizeof(type), cudaMemcpyDeviceToHost))
#define CUDA_KERNEL_OPT(array_size, block_size) \
    (((array_size) + (block_size) - 1) / (block_size)), (block_size)
#define CUDA_LAST_ERROR_TIME_SYNC(message) { \
    CUDA_LAST_ERR(); C Timer sync; CUDA_DEV_SYNC(); \
    std::cout << (message) << sync.getSeconds() << "\n"; }



/**
 * I need it so that
 * all boards that have a row perm done to them
 * are placed first
 * then
 * all boards that have a col perm done to them
 * are placed last
 *
 * maybe I could do it like
 *
 *
 */
class scanSizesAndOffsets {
public:
    u32* row_ScanSizes;
    u32* row_ScanOffsets;

    u32* col_ScanSizes;
    u32* col_ScanOffsets;

    u32* other_ScanSizes;
    u32* other_ScanOffsets;
};





__global__ void kernelCalcOffsetIndex(
        int numBoards,
        Board *inPtr,
        u32 *offsetPtr)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) { return; }

    inPtr[idx].setRowColCC(&offsetPtr[2 * idx]);
}


__global__ void kernelGenBoardsFromOffsets(
        int numCounts,
        Board* inPtr,
        Board* outPtr,

        C u32* scanSizes,
        C u32* scanOffsets)
{
    u32 globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    u32 warpId = globalThreadId / 32;
    if (warpId >= numCounts) return;

    u32 laneId = threadIdx.x % 32;
    u32 boardId = warpId / 2;

    Board srcBoard = inPtr[boardId];

    // Determine the number of unique states to generate
    u32 numUnique = scanSizes[warpId];
    u32 startOffset = scanOffsets[warpId];
    u32 startLaneId = 30 - numUnique + laneId;

    // Mask out threads beyond max count
    if (laneId < numUnique) {

        bool isRow = (warpId % 2) == 0;
        if (isRow) {
            outPtr[startOffset + laneId].memory = srcBoard.memory;
            my_cuda::R_X_X_copy(&srcBoard, &outPtr[startOffset + laneId], startLaneId);
            outPtr[startOffset + laneId].memory.setNextNMove<1>(startLaneId);
        } else {
            outPtr[startOffset + laneId].memory = srcBoard.memory;
            my_cuda::C_X_X_copy(&srcBoard, &outPtr[startOffset + laneId], startLaneId);
            outPtr[startOffset + laneId].memory.setNextNMove<1>(startLaneId + 32);
        }
    }

}


int main() {
    std::cout << "entered 'main()'.\n";
    // 13-1
    C Board board = BoardLookup::getBoardPair("4-4")->getStartState();
    Board solve = board;

    solve.doMoves({R41, C55, R22, R15 /*C34, R22, R44, C55*/});
    solve.memory.setNextMoves({20, 61, 11, 9 /*50, 11, 23, 61*/});


    static constexpr u32 ARRAY_SIZE = 4'000'000;
    static constexpr bool PRINT_ARRAYS = false;


    Board *h_boards_in;

    Board *d_boards_in, *d_boards_out;
    u32 *d_cc_sizes, *d_scan_offsets;


    // allocate all the memory
    h_boards_in = new Board[ARRAY_SIZE];

    CUDA_MALLOC(Board, &d_boards_in, ARRAY_SIZE);
    CUDA_MALLOC(Board, &d_boards_out, 60 * ARRAY_SIZE);

    CUDA_MALLOC(u32, &d_cc_sizes, 2 * ARRAY_SIZE);
    CUDA_MALLOC(u32, &d_scan_offsets, 2 * ARRAY_SIZE);


    // setup data for testing out the kernels
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_boards_in[i] = solve;
        /*
        if constexpr (PRINT_ARRAYS) {
            std::cout << i << ": " << h_boards_in[i].memory.asmStringForwards() << "\n";
        }
         */
    }
    CUDA_H_TO_D(Board, d_boards_in, h_boards_in, ARRAY_SIZE);


    // call kernel #1 to calculate children sizes
    kernelCalcOffsetIndex<<<CUDA_KERNEL_OPT(ARRAY_SIZE, 128)>>>
            (ARRAY_SIZE, d_boards_in, d_cc_sizes);
    CUDA_LAST_ERROR_TIME_SYNC("Kernel #1 Time: ");


    // exclusive scan
    thrust::device_ptr<u32> dev_counts(d_cc_sizes);
    thrust::device_ptr<u32> dev_scan(d_scan_offsets);
    thrust::exclusive_scan(dev_counts, dev_counts + 2 * ARRAY_SIZE, dev_scan);
    u32 h_last_scan_offset = 0;
    CUDA_D_TO_H(u32, &h_last_scan_offset, &d_scan_offsets[2 * ARRAY_SIZE - 1], 1);
    MU u32 FINAL_OUTPUT_SIZE = h_last_scan_offset + h_boards_in[ARRAY_SIZE - 1].getColCC();
    //if constexpr (PRINT_ARRAYS) {
    // std::cout << "Output Board Size: " << FINAL_OUTPUT_SIZE << "\n"; //}

    // call kernel #2 to create children in output board
    kernelGenBoardsFromOffsets<<<CUDA_KERNEL_OPT(64 * ARRAY_SIZE, 128)>>>
            (2 * ARRAY_SIZE, d_boards_in, d_boards_out, d_cc_sizes, d_scan_offsets);
    CUDA_LAST_ERROR_TIME_SYNC("Kernel #2 Time: ");



    // show GPU Results
    //if constexpr (PRINT_ARRAYS) {
        u32 MAX_SIZE = 20;
        auto* h_boards_out = new Board[MAX_SIZE];
        CUDA_D_TO_H(Board, h_boards_out, d_boards_out, MAX_SIZE);
        for (int i = 0; i < MAX_SIZE; i++) {
            std::cout << i << ": " << h_boards_out[i].memory.asmStringForwards() << "\n";
        }
        std::cout << std::flush;

/*
        std::cout << "GPU Offsets: [";
        u32* h_scan_offsets = new u32[2 * ARRAY_SIZE];
        CUDA_D_TO_H(u32, h_scan_offsets, d_scan_offsets, 2 * ARRAY_SIZE);
        for (int i = 0; i < 2 * ARRAY_SIZE; i++) {
            std::cout << h_scan_offsets[i];
            if (i != 2 * ARRAY_SIZE - 1) std::cout << ", ";
            if (i % 10 == 9 && i != 2 * ARRAY_SIZE - 1) { std::cout << "\n              "; }
        }
        std::cout << "]" << std::endl;
        delete[] h_boards_out;
    */
    //}


    // deallocate all memory
    delete[] h_boards_in;

    CUDA_FREE(d_boards_in);
    CUDA_FREE(d_boards_out);
    CUDA_FREE(d_cc_sizes);
    CUDA_FREE(d_scan_offsets);

    return 0;
}
