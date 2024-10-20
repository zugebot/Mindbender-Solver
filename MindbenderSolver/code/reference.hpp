#pragma once

#include "board.hpp"
#include "rotations.hpp"

#include <vector>
#include <array>


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






