#pragma once

#include "board.hpp"
#include <vector>

#include <unordered_map>

typedef std::vector<Board> (*MakePermFuncArray)(Board&, u32);
extern MakePermFuncArray makePermutationListFuncs[];
extern MakePermFuncArray make2PermutationListFuncs[];


typedef std::vector<Board> (*MakePermFuncArray)(Board&, u32);
extern MakePermFuncArray makeFatPermutationListFuncs[];



static const std::unordered_map<int, std::vector<std::pair<int, int>>> permutationDepthMap = {
        {1, {{1, 0}, {0, 1}}},
        {2, {{1, 1}, {2, 0}, {0, 2}}},
        {3, {{2, 1}, {1, 2}, {3, 0}, {0, 3}}},
        {4, {{2, 2}, {3, 1}, {1, 3}, {4, 0}, {0, 4}}},
        {5, {{3, 2}, {3, 2}, {4, 1}, {1, 4}, {5, 0}, {0, 5}}},
        {6, {{3, 3}, {4, 2}, {2, 4}, {5, 1}, {1, 5}}},
        {7, {{4, 3}, {3, 4}, {5, 2}, {2, 5}}},
        {8, {{4, 4}, {5, 3}, {3, 5}}},
        {9, {{5, 4},{4, 5}}},
        {10, {{5, 5}}},
};





