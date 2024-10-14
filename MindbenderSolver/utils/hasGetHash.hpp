#pragma once

#include <concepts>

#include "MindbenderSolver/utils/processor.hpp"

template<typename T>
concept HasGetHash = requires(T a) { { a.getHash() } -> std::same_as<u64>; };