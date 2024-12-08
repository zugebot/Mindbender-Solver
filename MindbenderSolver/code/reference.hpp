#pragma once

#include "allowed_type.hpp"


template<typename T, int MAX_DEPTH>
class Ref {
    static_assert(AllowedPermsType<T>, "T must be Memory or Board");

public:
    std::array<int, MAX_DEPTH> dir_seq = {};
    std::array<int, MAX_DEPTH> sect_seq = {};
    std::array<int, MAX_DEPTH> base_seq = {};
    std::array<u64, MAX_DEPTH> cur_seq = {};
    std::array<bool, MAX_DEPTH> checkRC_seq = {}; // first index not used
    std::array<u8, MAX_DEPTH> intersect_seq = {}; // first index not used
    typename T::HasherPtr hasher{};
};