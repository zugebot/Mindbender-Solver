#pragma once

#include "board.hpp"
#include <vector>


typedef std::vector<Board> (*make_permutation_list_func)(Board&);
extern make_permutation_list_func makePermutationListFuncs[];
