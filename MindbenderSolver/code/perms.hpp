#pragma once

#include "board.hpp"

#include <vector>
#include <unordered_map>


static constexpr u64 BOARD_PRE_ALLOC_SIZES[7] = {
        1,
        60,
        2550,
        104000,
        4245000,
        173325000,
        7076687500,
        // 288933750000,
        // 11796869531250,
        // 481654101562500,
        // 19665443613281250,
        // 802919920312500000,
};


static constexpr u64 BOARD_FAT_PRE_ALLOC_SIZES[7] = {
        1,
        48,
        2304,
        110592,
        5308416,
        254803968,
        12230590464,
};


static constexpr u64 FAT_PERM_COUNT = 48;
static constexpr bool FAT_CHECK_SIMILAR = true;

#define FAT_PERM_FUNC_DECLARE(name) \
template<bool CHECK_SIMILAR=FAT_CHECK_SIMILAR> \
void name(vecBoard_t& boards_out, const Board &board_in, Board::HasherPtr hasher);
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_0)
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_1)
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_2)
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_3)
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_4)
FAT_PERM_FUNC_DECLARE(make_fat_permutation_list_depth_5)


static constexpr bool NORM_CHECK_INTERSECTION = true;
static constexpr bool NORM_CHECK_SIMILAR = true;

#define NORM_PERM_FUNC_DECLARE(name) \
template<bool CHECK_INTERSECTION=NORM_CHECK_INTERSECTION, bool CHECK_SIMILAR=NORM_CHECK_SIMILAR> \
void name(vecBoard_t& boards_out, const Board &board_in, Board::HasherPtr hasher);
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_0)
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_1)
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_2)
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_3)
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_4)
NORM_PERM_FUNC_DECLARE(make_permutation_list_depth_5)


template<bool CHECK_INTERSECTION=true, bool CHECK_SIMILAR=true>
void make_permutation_list_depth_plus_one(const vecBoard_t &boards_in,
    vecBoard_t &boards_out, Board::HasherPtr hasher);

template<bool CHECK_INTERSECTION=true, bool CHECK_SIMILAR=true, u32 BUFFER_SIZE=33'554'432>
void make_permutation_list_depth_plus_one_buffered(const std::string& root_path,
    const vecBoard_t &boards_in, vecBoard_t &boards_out, Board::HasherPtr hasher);






class Permutations {
    static constexpr u32 PTR_LIST_SIZE = 6;
public:
    typedef void (*toDepthFuncPtr_t)(vecBoard_t &, const Board &, Board::HasherPtr);
    typedef void (*toDepthPlusOneFuncPtr_t)(const vecBoard_t &, vecBoard_t &, Board::HasherPtr);
    typedef void (*toDepthPlusOneFuncBufferedPtr_t)(const std::string&, const vecBoard_t &, vecBoard_t &, Board::HasherPtr);
    typedef std::unordered_map<u32, std::vector<std::pair<u32, u32>>> depthMap_t;

    static const depthMap_t depthMap;
    static toDepthFuncPtr_t toDepthFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFatFuncPtrs[PTR_LIST_SIZE];
    static toDepthPlusOneFuncPtr_t toDepthPlusOneFuncPtr;
    static toDepthPlusOneFuncBufferedPtr_t toDepthPlusOneBufferedFuncPtr;

    MU static void reserveForDepth(const Board &board_in, vecBoard_t& boards_out, u32 depth, bool isFat = false);
    MU static void getDepthFunc(const Board &board_in, vecBoard_t &boards_out, u32 depth, bool shouldResize = true);
    MU static void getDepthPlus1Func(const vecBoard_t& boards_in, vecBoard_t& boards_out, bool shouldResize = true);
    MU static void getDepthPlus1BufferedFunc(const std::string& root_path, const vecBoard_t& boards_in, vecBoard_t& boards_out, int depth);
};



