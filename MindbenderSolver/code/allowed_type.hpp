#pragma once

#include "memory.hpp"


#ifdef USE_CUDA
// C++17 version because my GPU is ASS
template<typename T>
struct IsAllowedPermsType {
    static constexpr bool value =
            std::is_same_v<T, Memory> || std::is_same_v<T, Board>;
};

template<typename T>
constexpr bool AllowedPermsType = IsAllowedPermsType<T>::value;

#else
// C++20 concept implementation
template<typename T>
concept AllowedPermsType = std::is_same_v<T, Memory> || std::is_same_v<T, Board>;

#endif