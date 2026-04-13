#pragma once
// ml_torch/src/model/distance_net.hpp

#include <torch/torch.h>
#include "ml_torch/src/common/constants.hpp"

namespace mindbender_ml {

/**
 * CNN model to predict remaining moves from a (state, goal) board pair.
 * 
 * Input:  [batch, 18, 6, 6]
 * Output: [batch, kOutputClasses] logits for depth prediction
 * 
 * Architecture:
 *   - 3 conv blocks with GroupNorm (batch-size safe)
 *   - Global self-attention mixing across all 36 cells
 *   - 2 fully connected layers
 *   - Output: logits for classes 0..kMaxDepthClass
 */
class DistanceNetImpl : public torch::nn::Module {
public:
    DistanceNetImpl();
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::GroupNorm gn1, gn2, gn3;
    torch::nn::Conv2d q_proj, k_proj, v_proj;
    torch::nn::Linear fc1, fc2;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(DistanceNet);

} // namespace mindbender_ml

