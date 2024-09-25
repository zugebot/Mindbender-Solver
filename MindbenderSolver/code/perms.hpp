#pragma once

#include "board.hpp"

#include <vector>
#include <unordered_map>


static constexpr u64 BOARD_PRE_ALLOC_SIZES[6] = {
        1,
        60,
        2550,
        104000,
        4245000,
        173325000,
        // 7076687500,
        // 288933750000,
        // 11796869531250,
        // 481654101562500,
        // 19665443613281250,
        // 802919920312500000,
};


static constexpr u64 BOARD_FAT_PRE_ALLOC_SIZES[6] = {
        1,
        48,
        2304,
        110592,
        5308416,
        254803968,
};


class Permutations {
    static constexpr u32 PTR_LIST_SIZE = 6;
public:
    typedef void (*toDepthFuncPtr_t)(vecBoard_t &, const Board &, u32);
    typedef void (*toDepthPlusOneFuncPtr_t)(const vecBoard_t &, vecBoard_t &, u32);
    typedef std::unordered_map<u32, std::vector<std::pair<u32, u32>>> depthMap_t;

    static const depthMap_t depthMap;
    static toDepthFuncPtr_t toDepthFuncPtrs[PTR_LIST_SIZE];
    static toDepthFuncPtr_t toDepthFatFuncPtrs[PTR_LIST_SIZE];
    static toDepthPlusOneFuncPtr_t toDepthPlusOneFuncPtr;

    MU static void reserveForDepth(const Board& board_in, vecBoard_t& boards_out, u32 depth, bool isFat = false);
    MU static void getDepthFunc(const Board &board_in, vecBoard_t &boards_out, u32 depth, bool shouldResize = true);
    MU static void getDepthPlus1Func(const vecBoard_t& boards_in, vecBoard_t& boards_out, bool shouldResize = true);
};



