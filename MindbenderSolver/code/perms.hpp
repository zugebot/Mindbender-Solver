#pragma once

#include "board.hpp"
#include "rotations.hpp"

#include "MindbenderSolver/utils/jvec.hpp"

#include <array>
#include <vector>
#include <unordered_map>


static constexpr u64 BOARD_PRE_MAX_MALLOC_SIZES[8] = {
        1, 60, 2550, 104000, 4245000, 173325000, 7076687500, 288933750000,};


/**
 * these are an upper-limit for each depth,
 * it's dependent on the fat location
 *
 * so a good upper limit for guessing more is
 * 1: 48
 * 2: 27.5
 * 3: 27.577272727
 * 4: 27.503104225
 * 5: 27.4814706423
 * SIZE = 48 * 27.6 ^ (depth - 1)
 */
static constexpr u64 BOARD_FAT_MAX_MALLOC_SIZES[8] = {
        1, 48, 1320, 36402, 1001168, 27513569, 0, 0};


// ######################################################################################
// #                        PERMUTATION [0 -> DEPTH] NORMAL                             #
// ######################################################################################


template<int MAX_DEPTH>
class Ref {
public:
    std::array<int, MAX_DEPTH> dir_seq = {};
    std::array<int, MAX_DEPTH> sect_seq = {};
    std::array<int, MAX_DEPTH> base_seq = {};
    std::array<u64, MAX_DEPTH> cur_seq = {};
    std::array<bool, MAX_DEPTH> checkRC_seq = {}; // first index not used
    std::array<u8, MAX_DEPTH> intersect_seq = {}; // first index not used
    Memory::HasherPtr hasher{};
};

template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
static void make_perm_list_inner(C Board &board_in, JVec<Memory> &boards_out,
    Ref<MAX_DEPTH> &ref, u64 move_prev, int& count);

template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS,
         bool CHECK_SIM, bool CHANGE_SECT_START, bool SECT_ASCENDING>
static void make_perm_list_outer(C Board &board_in, JVec<Memory> &boards_out,
    Ref<MAX_DEPTH> &ref, int& count);

/// Entry point function
template<int MAX_DEPTH, bool CHECK_CROSS = true, bool CHECK_SIM = true,
         bool CHANGE_SECT_START = true, bool SECT_ASCENDING = true>
void make_perm_list(C Board &board_in, JVec<Memory> &boards_out, Memory::HasherPtr hasher);



// ######################################################################################
// #                          PERMUTATION [0 -> DEPTH] FAT                              #
// ######################################################################################


extern u32 MAKE_FAT_PERM_LIST_HELPER_CALLS;
extern u32 MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS;
extern u32 MAKE_FAT_PERM_LIST_HELPER_FOUND_SIMILAR;

template<int CUR_DEPTH, int MAX_DEPTH, bool MOVES_ASCENDING, bool DIRECTION>
static void make_fat_perm_list_helper(
        C Board &board, JVec<Memory> &boards_out, u32 &count, Memory::HasherPtr hasher,
        u64 move, C ActStruct&, u8 startIndex, u8 endIndex);

/// Entry point function
template<int DEPTH, bool MOVES_ASCENDING=true>
void make_fat_perm_list(C Board& board_in, JVec<Memory> &boards_out, Memory::HasherPtr hasher);


// ######################################################################################
// #                      PERMUTATION [N -> N + DEPTH] NORMAL                           #
// ######################################################################################

template<bool CHECK_CROSS=true, bool CHECK_SIM=true>
void make_permutation_list_depth_plus_one(
    C JVec<Board> &boards_in, JVec<Board> &boards_out, Board::HasherPtr hasher);

template<bool CHECK_CROSS=true, bool CHECK_SIM=true, u32 BUFFER_SIZE=33'554'432>
void make_permutation_list_depth_plus_one_buffered(C std::string& root_path,
    C JVec<Board> &boards_in, JVec<Board> &boards_buffer, Board::HasherPtr hasher);




class Perms {
    static constexpr u32 PTR_LIST_SIZE = 6;
public:
    typedef void (*toDepthFuncPtr_t)(C Board &, JVec<Memory> &, Memory::HasherPtr);
    typedef void (*toDepthPlusOneFuncPtr_t)(C JVec<Board> &, JVec<Board> &, Board::HasherPtr);
    typedef void (*toDepthPlusOneFuncBufferedPtr_t)(C std::string&, C JVec<Board> &, JVec<Board> &, Board::HasherPtr);
    typedef std::unordered_map<u32, std::vector<std::pair<u32, u32>>> depthMap_t;

    static C depthMap_t depthMap;
    static toDepthFuncPtr_t toDepthFromLeftFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFromRightFuncPtrs[PTR_LIST_SIZE];

    static toDepthFuncPtr_t toDepthFromLeftFatFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFromRightFatFuncPtrs[PTR_LIST_SIZE];

    static toDepthPlusOneFuncPtr_t toDepthPlusOneFuncPtr;
    static toDepthPlusOneFuncBufferedPtr_t toDepthPlusOneBufferedFuncPtr;

    MU static void reserveForDepth(C Board &board_in, JVec<Memory>& boards_out, u32 depth);
    MU static void reserveForDepth(C Board &board_in, JVec<Board>& boards_out, u32 depth);

    template<bool SECT_ASCENDING>
    MU static void getDepthFunc(C Board &board_in, JVec<Memory> &boards_out, u32 depth, bool shouldResize = true);
    MU static void getDepthPlus1Func(C JVec<Board>& boards_in, JVec<Board>& boards_out, bool shouldResize = true);
    MU static void getDepthPlus1BufferedFunc(C std::string& root_path, C JVec<Board>& boards_in, JVec<Board>& boards_buffer, int depth);
};


template<bool SECT_ASCENDING = true>
void Perms::getDepthFunc(C Board& board_in, JVec<Memory> &boards_out, C u32 depth, C bool shouldResize) {
    if (depth >= PTR_LIST_SIZE) { return; }
    if (shouldResize) { reserveForDepth(board_in, boards_out, depth); }

    boards_out.resize(boards_out.capacity());
    C Memory::HasherPtr hasher = Memory::getHashFunc(board_in);

    if (board_in.getFatBool()) {
        constexpr auto FUNC_DIR = SECT_ASCENDING ? toDepthFromLeftFatFuncPtrs : toDepthFromRightFatFuncPtrs;
        FUNC_DIR[depth](board_in, boards_out, hasher);
    } else {
        constexpr auto FUNC_DIR = SECT_ASCENDING ? toDepthFromLeftFuncPtrs : toDepthFromRightFuncPtrs;
        FUNC_DIR[depth](board_in, boards_out, hasher);
    }
}


#include "perms_fat.tpp"
#include "perms_nrm.tpp"