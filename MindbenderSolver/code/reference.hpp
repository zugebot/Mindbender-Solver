#pragma once

#include "board.hpp"
#include "rotations.hpp"

#include <vector>
#include <array>


typedef Board PERMOBJ_t;
typedef const Board c_PERMOBJ_t;
typedef std::vector<PERMOBJ_t> vec_PERMOBJ_t;
typedef const std::vector<PERMOBJ_t> c_vec_PERMOBJ_t;


template<int MAX_DEPTH>
class Ref {
public:
    std::array<int, MAX_DEPTH> dir_seq = {};
    std::array<int, MAX_DEPTH> sect_seq = {};
    std::array<int, MAX_DEPTH> base_seq = {};
    std::array<u64, MAX_DEPTH> cur_seq = {};
    std::array<bool, MAX_DEPTH> checkRC_seq = {}; // first index not used
    std::array<u8, MAX_DEPTH> intersect_seq = {}; // first index not used
    c_PERMOBJ_t::HasherPtr hasher{};
};






