#pragma once
// ml_torch/src/common/constants.hpp

#include <cstdint>

namespace mindbender_ml {

// Board dimensions
inline constexpr int kBoardRows = 6;
inline constexpr int kBoardCols = 6;
inline constexpr int kBoardSize = kBoardRows * kBoardCols;
inline constexpr int kMaxColors = 8;

// Model input/output
inline constexpr int kMaxColorValue = 8;
inline constexpr int kInputChannels = 18;  // 8 one-hot (state) + 8 one-hot (goal) + 2 (equality + color_count_normalized)
inline constexpr int kMaxDepthClass = 13;  // Predict depth 0..13
inline constexpr int kOutputClasses = kMaxDepthClass + 1;

// Training data generation
inline constexpr int kSyntheticDataMinDepth = 7;      // Generate solutions starting at depth 7
inline constexpr int kSyntheticDataMaxDepth = 10;     // Up to depth 10
inline constexpr int kDefaultNumSyntheticPuzzles = 100;
inline constexpr uint64_t kDefaultSeed = 1337;

} // namespace mindbender_ml

