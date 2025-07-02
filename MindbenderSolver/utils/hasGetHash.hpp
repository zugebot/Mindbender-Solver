#pragma once

#include "MindbenderSolver/utils/processor.hpp"


#ifdef USE_CUDA
template <typename T, typename = void>
struct HasGetHash : std::false_type {};

// Specialization for types with a `getHash()` method returning `u64`
template <typename T>
struct HasGetHash<T, std::void_t<decltype(std::declval<T>().getHash())>>
    : std::is_same<decltype(std::declval<T>().getHash()), uint64_t> {};

template <typename T>
constexpr bool HasGetHash_v = HasGetHash<T>::value;

#else
#include <concepts>
template<typename T>
concept HasGetHash_v = requires(T a) { { a.getHash() } -> std::same_as<u64>; };

#endif