#pragma once

#include "board.hpp"
#include "rotations.hpp"
#include "reference.hpp"

#include "MindbenderSolver/utils/jvec.hpp"


#include <array>
#include <vector>
#include <unordered_map>


static constexpr u64 BOARD_PRE_MAX_MALLOC_SIZES[8] = {
        1, 60, 2550, 104000, 4245000, 173325000, 7076687500, 288933750000,};


// static constexpr u64 BOARD_FAT_MAX_MALLOC_SIZES[8] = {
//         1, 48, 2304, 110592, 5308416, 254803968, 12230590464, 587068342272};

static constexpr u64 BOARD_FAT_MAX_MALLOC_SIZES[8] = {
        1, 48, 1320, 36402, 1001168, 27513569, 0, 0};

template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
static void make_perm_list_inner(
        const Board &board_in,
        JVec<HashMem> &boards_out,
        Ref<MAX_DEPTH> &ref, c_u64 move_prev, int& count);

template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS,
        bool CHECK_SIM, bool CHANGE_SECT_START, bool SECT_ASCENDING>
static void make_perm_list_outer(
        const Board &board_in,
        JVec<HashMem> &boards_out,
        Ref<MAX_DEPTH> &ref, int& count);

/// Entry point function
template<int MAX_DEPTH, bool CHECK_CROSS = true, bool CHECK_SIM = true,
         bool CHANGE_SECT_START = true, bool SECT_ASCENDING = true>
void make_perm_list(
        const Board &board_in,
        JVec<HashMem> &boards_out,
        HashMem::HasherPtr hasher);


extern u32 MAKE_FAT_PERM_LIST_HELPER_CALLS;
extern u32 MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS;
extern u32 MAKE_FAT_PERM_LIST_HELPER_FOUND_SIMILAR;


template<int CUR_DEPTH, int MAX_DEPTH, bool MOVES_ASCENDING, bool DIRECTION>
static void make_fat_perm_list_helper(
        const Board &board,
        JVec<HashMem> &boards_out,
        u32 &count,
        HashMem::HasherPtr hasher,
        u64 move,
        const ActStruct&,
        u8 startIndex,
        u8 endIndex);


/// Entry point function
template<int DEPTH, bool MOVES_ASCENDING=true>
void make_fat_perm_list(
        const Board& board_in,
        JVec<HashMem> &boards_out,
        HashMem::HasherPtr hasher);



template<bool CHECK_CROSS=true, bool CHECK_SIM=true>
void make_permutation_list_depth_plus_one(
    const JVec<Board> &boards_in, JVec<Board> &boards_out, Board::HasherPtr hasher);

template<bool CHECK_CROSS=true, bool CHECK_SIM=true, u32 BUFFER_SIZE=33'554'432>
void make_permutation_list_depth_plus_one_buffered(const std::string& root_path,
    const JVec<Board> &boards_in, JVec<Board> &boards_out, Board::HasherPtr hasher);


class Perms {
    static constexpr u32 PTR_LIST_SIZE = 6;
public:
    typedef void (*toDepthFuncPtr_t)(const Board &, JVec<HashMem> &, HashMem::HasherPtr);
    typedef void (*toDepthPlusOneFuncPtr_t)(const JVec<Board> &, JVec<Board> &, Board::HasherPtr);
    typedef void (*toDepthPlusOneFuncBufferedPtr_t)(const std::string&, const JVec<Board> &, JVec<Board> &, Board::HasherPtr);
    typedef std::unordered_map<u32, std::vector<std::pair<u32, u32>>> depthMap_t;

    static const depthMap_t depthMap;
    static toDepthFuncPtr_t toDepthFromLeftFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFromRightFuncPtrs[PTR_LIST_SIZE];

    static toDepthFuncPtr_t toDepthFromLeftFatFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFromRightFatFuncPtrs[PTR_LIST_SIZE];

    static toDepthPlusOneFuncPtr_t toDepthPlusOneFuncPtr;
    static toDepthPlusOneFuncBufferedPtr_t toDepthPlusOneBufferedFuncPtr;

    MU static void reserveForDepth(const Board &board_in, JVec<HashMem>& boards_out, u32 depth);
    MU static void reserveForDepth(const Board &board_in, JVec<Board>& boards_out, u32 depth);

    template<bool SECT_ASCENDING>
    MU static void getDepthFunc(const Board &board_in, JVec<HashMem> &boards_out, u32 depth, bool shouldResize = true);
    MU static void getDepthPlus1Func(const JVec<Board>& boards_in, JVec<Board>& boards_out, bool shouldResize = true);
    MU static void getDepthPlus1BufferedFunc(const std::string& root_path, const JVec<Board>& boards_in, JVec<Board>& board_buffer, int depth);
};


template<bool SECT_ASCENDING = true>
void Perms::getDepthFunc(const Board& board_in, JVec<HashMem> &boards_out, c_u32 depth, c_bool shouldResize) {
    if (depth >= PTR_LIST_SIZE) { return; }
    if (shouldResize) { reserveForDepth(board_in, boards_out, depth); }

    boards_out.resize(boards_out.capacity());
    const HashMem::HasherPtr hasher = HashMem::getHashFunc(board_in);
    if (board_in.getFatBool()) {
        if constexpr (SECT_ASCENDING) {
            toDepthFromLeftFatFuncPtrs[depth](board_in, boards_out, hasher);
        } else {
            toDepthFromRightFatFuncPtrs[depth](board_in, boards_out, hasher);
        }
    } else {
        if constexpr (SECT_ASCENDING) {
            toDepthFromLeftFuncPtrs[depth](board_in, boards_out, hasher);
        } else {
            toDepthFromRightFuncPtrs[depth](board_in, boards_out, hasher);
        }
    }
}


#include "perms.tpp"