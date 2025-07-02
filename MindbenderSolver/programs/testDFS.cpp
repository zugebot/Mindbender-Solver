#include "MindbenderSolver/include.hpp"

#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)





template<int DEPTH>
struct RefState {
    B1B2 start;
    B1B2 end;
    int count = 0;

    int DEPTHS_COUNT[DEPTH + 1] = {};
    int states_traversed = 0;

    RefState() = default;
};



template<int CUR_DEPTH, int MAX_DEPTH, bool ROW_TRUE>
HD void recursive_helper(
        RefState<MAX_DEPTH>& theState, C B1B2 theBoard, C int theNext);









#ifdef __CUDA_ARCH__
#define FUNC_DEF auto func = my_cuda::allActStructListGPU[actIndex]
#else
#define FUNC_DEF auto func = allActStructList[actIndex]
#endif

#define LOOP_DEF(ROW_TRUE_BOOL) \
    FUNC_DEF; \
    nextBoard = theBoard; \
    func.action(nextBoard); \
                                \
    if (nextBoard == theBoard) { continue; } \
                                \
    if constexpr (DIFF_DEPTH > 0 && DIFF_DEPTH < 6) { \
        if (nextBoard.getScore3Till<DIFF_DEPTH>(theState.end)) { continue; } } \
                                \
    recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, ROW_TRUE_BOOL>( \
            theState, nextBoard, func.index + func.tillNext)




template<int CUR_DEPTH, int MAX_DEPTH, bool ROW_TRUE>
HD void recursive_helper(RefState<MAX_DEPTH>& theState, C B1B2 theBoard, C int theNext) {
    ++theState.DEPTHS_COUNT[CUR_DEPTH];
    ++theState.states_traversed;

    if constexpr (CUR_DEPTH < MAX_DEPTH) {
        static constexpr int DIFF_DEPTH = MAX_DEPTH - CUR_DEPTH;


        B1B2 nextBoard;

        for (int actIndex = ROW_TRUE ? theNext : 0;
             actIndex < 30; ++actIndex) { LOOP_DEF(true); }

        for (int actIndex = ROW_TRUE ? 32 : theNext;
             actIndex < 62; ++actIndex) { LOOP_DEF(false); }


    } else if constexpr (CUR_DEPTH == MAX_DEPTH) {
        if EXPECT_FALSE(theBoard == theState.end) {
            theState.count++;
        }
    }
}


template <int MAX_DEPTH>
__global__ void cudaRecursiveKernel(
        RefState<MAX_DEPTH>* states, int numThreads) {
    C u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {
        RefState<MAX_DEPTH>& state = states[idx];
        recursive_helper<0, MAX_DEPTH, true>(state, state.start, 0);
    }
}



MU __global__ void testRXX(Board theBoardPtr[30]) {
    C i32 idx = static_cast<i32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 30) {
        my_cuda::R_X_X(theBoardPtr[idx], idx);
    }
}


MU __global__ void testCXX(Board theBoardPtr[30]) {
    C i32 idx = static_cast<i32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < 30) {
        my_cuda::C_X_X(theBoardPtr[idx], idx);
    }
}





int main() {

    /*
    Board rxxBoard({
        6, 0, 0, 0, 0, 0,
        0, 6, 6, 6, 6, 6,
        0, 6, 6, 6, 6, 6,
        0, 6, 6, 6, 6, 6,
        0, 6, 6, 6, 6, 6,
        0, 6, 6, 6, 6, 6,
    });

    C_0_3(rxxBoard);
    std::cout << rxxBoard.toStringSingle({}) << "\n";
    C_0_3(rxxBoard);



    const int numBoards = 30;
    Board h_boards[numBoards];

    // Initialize each Board in the array to the input board
    for(int i = 0; i < numBoards; ++i) {
        h_boards[i] = rxxBoard;
    }

    // Step 4: Allocate device memory for Boards
    Board* d_boards;
    cudaMalloc(&d_boards, sizeof(Board) * numBoards);


    // Step 5: Copy host Boards to device
    cudaMemcpy(d_boards, h_boards, sizeof(Board) * numBoards, cudaMemcpyHostToDevice);


    // Step 6: Define grid and block sizes
    int threadsPerBlock = 32;
    int blocksPerGrid = (numBoards + threadsPerBlock - 1) / threadsPerBlock;

    // Step 7: Launch the testRXX kernel
    testCXX<<<blocksPerGrid, threadsPerBlock>>>(d_boards);


    // Step 9: Wait for GPU to finish
    cudaDeviceSynchronize();

    // Step 10: Copy results back to host
    cudaMemcpy(h_boards, d_boards, sizeof(Board) * numBoards, cudaMemcpyDeviceToHost);


    // Step 11: Print results for verification
    std::cout << "\nPermuted Boards:" << std::endl;
    for(int i = 0; i < numBoards; ++i) {
        char str[5] = {0};
        memcpy(str, &allActStructList[i + 30].name, 4);
        std::cout << str << "\n";
        Board temp = h_boards[i];
        std::cout << temp.toStringSingle({}) << "\n";
        // std::cout << "Board " << i << ": b1 = " << h_boards[i].b1 << ", b2 = " << h_boards[i].b2 << std::endl;
    }

    // Step 12: Free device memory
    cudaFree(d_boards);

    return 0;
    */



    std::cout << "starting" << std::endl;
    // 13-1
    C Board board = BoardLookup::getBoardPair("4-4")->getStartState();
    Board solve = board;

    solve.doMoves({R_4_1, C_5_5, R_2_2, R_1_5, C_3_4, R_2_2, R_4_4,});

    Memory mem({20, 61, 11, 9, 50, 11, 23});
    std::cout << mem.asmStringForwards() << "\n";



    constexpr int DEPTH = 7;
    constexpr int NUM_THREADS = 64 * 1;
    constexpr int BLOCK_SIZE = 64;
    constexpr int GRID_SIZE = (NUM_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;


    Timer cpu_only;
    auto cpu_state = RefState<DEPTH>();
    cpu_state.start = static_cast<B1B2>(solve);
    cpu_state.end = static_cast<B1B2>(board);
    recursive_helper<0, DEPTH, true>(cpu_state, cpu_state.start, 0);
    std::cout << "cpu only time: " << cpu_only.getSeconds() << "\n";
    std::cout << "Solves: " << cpu_state.count << std::endl;
    std::cout << "Traversed: " << cpu_state.states_traversed << std::endl;
    for (int i = 0; i < DEPTH + 1; ++i) {
        std::cout << cpu_state.DEPTHS_COUNT[i];
        if (i != DEPTH) {  std::cout << ", "; }
    }
    std::cout << "\n";


    std::cout << "Total Threads: " << NUM_THREADS << std::endl;
    // Host-side setup
    auto* h_states = new RefState<DEPTH>[NUM_THREADS];
    size_t STATE_SIZE = NUM_THREADS * sizeof(RefState<DEPTH>);

    // Initialize state for each thread
    for (int i = 0; i < NUM_THREADS; ++i) {
        h_states[i].start = static_cast<B1B2>(solve);
        h_states[i].end = static_cast<B1B2>(board);
    }

    // Device-side memory allocation
    RefState<DEPTH>* d_states;
    C Timer allocT;
    CUDA_CHECK(cudaMalloc(&d_states, STATE_SIZE));
    std::cout << "Alloc: " << allocT.getSeconds() << std::endl;

    // Copy states to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states, STATE_SIZE, cudaMemcpyHostToDevice));


    // Launch kernel
    cudaRecursiveKernel<DEPTH><<<GRID_SIZE, BLOCK_SIZE>>>(d_states, NUM_THREADS);

    CUDA_CHECK(cudaGetLastError());


    C Timer syncT;
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Synchronize: " << syncT.getSeconds() << std::endl;

    // Copy states back to host
    CUDA_CHECK(cudaMemcpy(h_states, d_states, STATE_SIZE, cudaMemcpyDeviceToHost));




    std::cout << "Time: " << syncT.getSeconds() << std::endl;
    std::cout << "Solves: " << h_states[0].count << std::endl;
    std::cout << "Traversed: " << h_states[0].states_traversed << std::endl;
    for (int i = 0; i < DEPTH + 1; ++i) {
        std::cout << h_states[0].DEPTHS_COUNT[i];
        if (i != DEPTH) {  std::cout << ", "; }
    }

    // Cleanup
    delete[] h_states;
    CUDA_CHECK(cudaFree(d_states));

    return 0;
}




/*
int main() {
    C Board board = BoardLookup::getBoardPair("13-1")->getStartState();
    Board solve = board;

    R_4_1(solve); // 0
    C_5_5(solve); // 1
    R_2_2(solve); // 2
    R_1_5(solve); // 3
    C_3_4(solve); // 4
    R_2_2(solve); // 5
    R_4_4(solve); // 6


    static constexpr int DEPTH = 7;
    static auto state = RefState<DEPTH>();
    state.start = static_cast<B1B2>(solve);
    state.end = static_cast<B1B2>(board);



    JVec<Board> boards;
    Perms<Board>::reserveForDepth(board, boards, 5);
    Perms<Board>::toDepthFromLeft::funcPtrs[5](board, boards, board.getHashFunc());
    std::cout << "[Arr] Length: " << boards.size() << std::endl;
    std::set<Board> boardSet;
    for (int i = 0; i < boards.size(); i++) {
        Board bi = boards[i];
        boardSet.insert(bi);
    }
    std::cout << "[Set] Length: " << boardSet.size() << std::endl;



    double timeTaken;
    recursive<DEPTH>(state, timeTaken);

    std::cout << "Time: " << timeTaken << std::endl;
    std::cout << "Depth: " << DEPTH << std::endl;
    std::cout << "Solves: " << state.count << std::endl;
    std::cout << "Traversed: " << state.states_traversed << std::endl;
    std::cout << "Total States: " << pow(60, DEPTH) << std::endl;
    // std::cout << "GetScore3: " << GET_SCORE_3_CALLS << std::endl;

    std::cout << "Depths: [";
    for (int i = 0; i < DEPTH + 1; ++i) {
        std::cout << state.DEPTHS_COUNT[i];
        if (i != DEPTH) {  std::cout << ", "; }
    }
    std::cout << "]\n";

    return 0;
}*/