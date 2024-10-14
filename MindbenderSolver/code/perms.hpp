#pragma once

#include "board.hpp"
#include "rotations.hpp"

#include "reference.hpp"

#include <array>
#include <vector>
#include <unordered_map>


static constexpr u64 BOARD_PRE_MAX_MALLOC_SIZES[8] = {
        1, 60, 2550, 104000, 4245000, 173325000, 7076687500, 288933750000,};


static constexpr u64 BOARD_FAT_MAX_MALLOC_SIZES[8] = {
        1, 48, 2304, 110592, 5308416, 254803968, 12230590464, 587068342272};


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
static void make_perm_list_inner(
        const Board &board_in,
        std::vector<HashMem> &boards_out,
        Ref<MAX_DEPTH> &ref, c_u64 move_prev, int& count);

template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
static void make_perm_list_outer(
        const Board &board_in,
        std::vector<HashMem> &boards_out,
        Ref<MAX_DEPTH> &ref, int& count);

/// Entry point function
template<int MAX_DEPTH, bool CHECK_CROSS = true, bool CHECK_SIM = true, bool CHANGE_SECT_START = true>
void make_perm_list(
        const Board &board_in,
        std::vector<HashMem> &boards_out,
        HashMem::HasherPtr hasher);


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_SIM>
static void make_fat_perm_list_recursive_helper(
        const Board &board,
        std::vector<HashMem> &boards_out,
        HashMem::HasherPtr hasher, u64 move);

/// Entry point function
template<int DEPTH, bool CHECK_SIM = true>
void make_fat_perm_list(
        const Board& board_in,
        std::vector<HashMem> &boards_out,
        HashMem::HasherPtr hasher);



template<bool CHECK_CROSS=true, bool CHECK_SIM=true>
void make_permutation_list_depth_plus_one(
    const std::vector<Board> &boards_in, std::vector<Board> &boards_out, Board::HasherPtr hasher);

template<bool CHECK_CROSS=true, bool CHECK_SIM=true, u32 BUFFER_SIZE=33'554'432>
void make_permutation_list_depth_plus_one_buffered(const std::string& root_path,
    const std::vector<Board> &boards_in, std::vector<Board> &boards_out, Board::HasherPtr hasher);


class Perms {
    static constexpr u32 PTR_LIST_SIZE = 6;
public:
    typedef void (*toDepthFuncPtr_t)(const Board &, std::vector<HashMem> &, const HashMem::HasherPtr);
    typedef void (*toDepthPlusOneFuncPtr_t)(const std::vector<Board> &, std::vector<Board> &, Board::HasherPtr);
    typedef void (*toDepthPlusOneFuncBufferedPtr_t)(const std::string&, const std::vector<Board> &, std::vector<Board> &, Board::HasherPtr);
    typedef std::unordered_map<u32, std::vector<std::pair<u32, u32>>> depthMap_t;

    static const depthMap_t depthMap;
    static toDepthFuncPtr_t toDepthFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFatFuncPtrs[PTR_LIST_SIZE];
    static toDepthPlusOneFuncPtr_t toDepthPlusOneFuncPtr;
    static toDepthPlusOneFuncBufferedPtr_t toDepthPlusOneBufferedFuncPtr;

    MU static void reserveForDepth(const Board &board_in, std::vector<HashMem>& boards_out, u32 depth);
    MU static void reserveForDepth(const Board &board_in, std::vector<Board>& boards_out, u32 depth);

    MU static void getDepthFunc(const Board &board_in, std::vector<HashMem> &boards_out, u32 depth, bool shouldResize = true);
    MU static void getDepthPlus1Func(const std::vector<Board>& boards_in, std::vector<Board>& boards_out, bool shouldResize = true);
    MU static void getDepthPlus1BufferedFunc(const std::string& root_path, const std::vector<Board>& boards_in, std::vector<Board>& boards_out, int depth);
};


#include "perms.tpp"