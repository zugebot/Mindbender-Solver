#pragma once
// ml_torch/src/common/board_encoder.hpp

#include <torch/torch.h>
#include "code/board.hpp"
#include "ml_torch/src/common/constants.hpp"

namespace mindbender_ml {

/**
 * Encode a (state, goal) board pair into a PyTorch tensor
 * 
 * Output shape: [kInputChannels, kBoardRows, kBoardCols] = [18, 6, 6]
 * 
 * Channels:
 *   0-7:   One-hot encoding of state board colors (color c → channel c)
 *   8-15:  One-hot encoding of goal board colors
 *   16:    Equality mask: 1.0 if state[i] == goal[i], else 0.0
 *   17:    Color count normalized to [0, 1]
 */
inline torch::Tensor encodeBoardPair(
    const B1B2& state_b1b2,
    const B1B2& goal_b1b2,
    int color_count
) {
    torch::Tensor tensor = torch::zeros({kInputChannels, kBoardRows, kBoardCols}, torch::kFloat32);
    
    auto accessor = tensor.accessor<float, 3>();
    
    // Extract colors from both boards
    for (int i = 0; i < kBoardSize; ++i) {
        int row = i / kBoardCols;
        int col = i % kBoardCols;
        
        // Get colors (0-7)
        uint8_t x = col;
        uint8_t y = row;
        const uint8_t state_color = state_b1b2.getColor(x, y);
        const uint8_t goal_color = goal_b1b2.getColor(x, y);
        const int state_idx = static_cast<int>(state_color) - 1;
        const int goal_idx = static_cast<int>(goal_color) - 1;
        
        // One-hot state (channels 0-7)
        if (state_idx >= 0 && state_idx < kMaxColors) {
            accessor[state_idx][row][col] = 1.0f;
        }
        
        // One-hot goal (channels 8-15)
        if (goal_idx >= 0 && goal_idx < kMaxColors) {
            accessor[8 + goal_idx][row][col] = 1.0f;
        }
        
        // Equality mask (channel 16)
        if (state_color == goal_color) {
            accessor[16][row][col] = 1.0f;
        }
    }
    
    // Color count normalized (channel 17, broadcast to all cells)
    const float color_norm = static_cast<float>(color_count) / static_cast<float>(kMaxColors);
    tensor[17].fill_(color_norm);
    
    return tensor;
}

} // namespace mindbender_ml

