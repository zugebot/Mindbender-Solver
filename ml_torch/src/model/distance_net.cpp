// ml_torch/src/model/distance_net.cpp

#include "distance_net.hpp"

#include <cmath>

#include <torch/torch.h>

namespace mindbender_ml {

DistanceNetImpl::DistanceNetImpl()
    : conv1(torch::nn::Conv2dOptions(18, 64, 3).padding(1)),
      conv2(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
      conv3(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
      gn1(torch::nn::GroupNormOptions(8, 64)),
      gn2(torch::nn::GroupNormOptions(8, 128)),
      gn3(torch::nn::GroupNormOptions(8, 128)),
      q_proj(torch::nn::Conv2dOptions(128, 128, 1)),
      k_proj(torch::nn::Conv2dOptions(128, 128, 1)),
      v_proj(torch::nn::Conv2dOptions(128, 128, 1)),
      fc1(128, 256),
      fc2(256, kOutputClasses),
      dropout(0.3) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("gn1", gn1);
    register_module("gn2", gn2);
    register_module("gn3", gn3);
    register_module("q_proj", q_proj);
    register_module("k_proj", k_proj);
    register_module("v_proj", v_proj);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("dropout", dropout);
}

torch::Tensor DistanceNetImpl::forward(torch::Tensor x) {
    // x: [batch, 18, 6, 6]

    // Local feature extraction.
    x = conv1(x);
    x = gn1(x);
    x = torch::relu(x);

    x = conv2(x);
    x = gn2(x);
    x = torch::relu(x);

    // Global all-to-all mixing: every cell attends to every other cell.
    const auto b = x.size(0);
    const auto c = x.size(1);
    const auto h = x.size(2);
    const auto w = x.size(3);
    const auto p = h * w;

    auto q = q_proj(x).view({b, c, p}).transpose(1, 2); // [B, P, C]
    auto k = k_proj(x).view({b, c, p}).transpose(1, 2); // [B, P, C]
    auto v = v_proj(x).view({b, c, p}).transpose(1, 2); // [B, P, C]

    auto scores = torch::bmm(q, k.transpose(1, 2)) / std::sqrt(static_cast<double>(c)); // [B, P, P]
    auto attn = torch::softmax(scores, -1);
    auto mixed = torch::bmm(attn, v).transpose(1, 2).contiguous().view({b, c, h, w});
    x = x + mixed;

    // Refinement + global pooling.
    x = conv3(x);
    x = gn3(x);
    x = torch::relu(x);
    x = torch::adaptive_avg_pool2d(x, {1, 1});

    x = x.view({x.size(0), -1});

    x = fc1(x);
    x = torch::relu(x);
    x = dropout(x);
    x = fc2(x);

    return x; // Raw logits
}

} // namespace mindbender_ml

