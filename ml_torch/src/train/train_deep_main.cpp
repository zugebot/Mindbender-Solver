// ml_torch/src/train/train_deep_main.cpp

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/torch.h>

#include "include/nlohmann/json.hpp"
#include "ml_torch/src/common/board_encoder.hpp"
#include "ml_torch/src/common/constants.hpp"
#include "ml_torch/src/model/distance_net.hpp"

namespace mindbender_ml {

struct TrainConfig {
    std::string dataset_path;
    std::string output_model_path;
    int batch_size = 64;
    int num_epochs = 50;
    double learning_rate = 5e-4;
    double weight_decay = 1e-4;
    double val_ratio = 0.1;
    uint64_t split_seed = 1337;
    double label_smoothing = 0.05;
    double ce_loss_weight = 1.0;
    double reg_loss_weight = 0.5;
    bool use_cuda = false;
    bool verbose = true;
};

struct CsvRow {
    uint64_t state_b1 = 0;
    uint64_t state_b2 = 0;
    uint64_t goal_b1 = 0;
    uint64_t goal_b2 = 0;
    int remaining_depth = 0;
    int color_count = 0;
    bool is_fat = false;
};

struct RowKey {
    uint64_t state_b1 = 0;
    uint64_t state_b2 = 0;
    uint64_t goal_b1 = 0;
    uint64_t goal_b2 = 0;
    int remaining_depth = 0;
    int color_count = 0;
    bool is_fat = false;

    bool operator==(const RowKey& o) const {
        return state_b1 == o.state_b1 &&
               state_b2 == o.state_b2 &&
               goal_b1 == o.goal_b1 &&
               goal_b2 == o.goal_b2 &&
               remaining_depth == o.remaining_depth &&
               color_count == o.color_count &&
               is_fat == o.is_fat;
    }
};

struct RowKeyHash {
    std::size_t operator()(const RowKey& k) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](const uint64_t v) {
            h ^= static_cast<std::size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        };
        mix(k.state_b1);
        mix(k.state_b2);
        mix(k.goal_b1);
        mix(k.goal_b2);
        mix(static_cast<uint64_t>(k.remaining_depth));
        mix(static_cast<uint64_t>(k.color_count));
        mix(static_cast<uint64_t>(k.is_fat ? 1 : 0));
        return h;
    }
};

struct SampleKey {
    uint64_t state_b1 = 0;
    uint64_t state_b2 = 0;
    uint64_t goal_b1 = 0;
    uint64_t goal_b2 = 0;
    int color_count = 0;
    bool is_fat = false;

    bool operator==(const SampleKey& o) const {
        return state_b1 == o.state_b1 &&
               state_b2 == o.state_b2 &&
               goal_b1 == o.goal_b1 &&
               goal_b2 == o.goal_b2 &&
               color_count == o.color_count &&
               is_fat == o.is_fat;
    }
};

struct SampleKeyHash {
    std::size_t operator()(const SampleKey& k) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](const uint64_t v) {
            h ^= static_cast<std::size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        };
        mix(k.state_b1);
        mix(k.state_b2);
        mix(k.goal_b1);
        mix(k.goal_b2);
        mix(static_cast<uint64_t>(k.color_count));
        mix(static_cast<uint64_t>(k.is_fat ? 1 : 0));
        return h;
    }
};

static SampleKey makeSampleKey(const CsvRow& r) {
    return {r.state_b1, r.state_b2, r.goal_b1, r.goal_b2, r.color_count, r.is_fat};
}

static std::vector<CsvRow> loadCsvDataset(const std::string& path) {
    std::vector<CsvRow> data;
    std::unordered_set<RowKey, RowKeyHash> seen;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open dataset: " + path);
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::string tok;
        std::vector<std::string> tokens;
        while (std::getline(iss, tok, ',')) {
            tokens.push_back(tok);
        }
        if (tokens.size() < 7) {
            continue;
        }

        CsvRow row;
        row.state_b1 = std::stoull(tokens[0], nullptr, 0);
        row.state_b2 = std::stoull(tokens[1], nullptr, 0);
        row.goal_b1 = std::stoull(tokens[2], nullptr, 0);
        row.goal_b2 = std::stoull(tokens[3], nullptr, 0);
        row.remaining_depth = std::stoi(tokens[4]);
        row.color_count = std::stoi(tokens[5]);
        row.is_fat = std::stoi(tokens[6]) != 0;

        if (row.remaining_depth < 0 || row.remaining_depth > kMaxDepthClass) {
            continue;
        }

        if (row.color_count < 1 || row.color_count > 8) {
            continue;
        }

        const RowKey key{row.state_b1, row.state_b2, row.goal_b1, row.goal_b2, row.remaining_depth, row.color_count, row.is_fat};
        if (!seen.insert(key).second) {
            continue;
        }

        data.push_back(row);
    }

    return data;
}

static void splitIndicesByBoardKey(const std::vector<CsvRow>& rows,
                                   const double val_ratio,
                                   const uint64_t seed,
                                   std::vector<std::size_t>& train_idx,
                                   std::vector<std::size_t>& val_idx) {
    train_idx.clear();
    val_idx.clear();
    if (rows.empty()) {
        return;
    }

    std::unordered_map<SampleKey, std::vector<std::size_t>, SampleKeyHash> groups;
    groups.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        groups[makeSampleKey(rows[i])].push_back(i);
    }

    std::vector<std::vector<std::size_t>> grouped;
    grouped.reserve(groups.size());
    for (auto& kv : groups) {
        grouped.push_back(std::move(kv.second));
    }

    std::mt19937_64 rng(seed);
    std::shuffle(grouped.begin(), grouped.end(), rng);

    const std::size_t target_val = static_cast<std::size_t>(static_cast<double>(rows.size()) * val_ratio);
    std::size_t current_val = 0;
    for (const auto& g : grouped) {
        if (current_val < target_val) {
            val_idx.insert(val_idx.end(), g.begin(), g.end());
            current_val += g.size();
        } else {
            train_idx.insert(train_idx.end(), g.begin(), g.end());
        }
    }

    if (train_idx.empty() && !val_idx.empty()) {
        train_idx.push_back(val_idx.back());
        val_idx.pop_back();
    }
}

static std::vector<std::size_t> buildLabelHistogram(const std::vector<int64_t>& labels,
                                                    const std::vector<std::size_t>& idx) {
    std::vector<std::size_t> hist(static_cast<std::size_t>(kMaxDepthClass + 1), 0);
    for (const std::size_t i : idx) {
        const int64_t y = labels[i];
        if (y >= 0 && y <= kMaxDepthClass) {
            ++hist[static_cast<std::size_t>(y)];
        }
    }
    return hist;
}

static torch::Tensor buildClassWeights(const std::vector<int64_t>& labels,
                                       const std::vector<std::size_t>& train_idx,
                                       const torch::Device& device) {
    const auto hist = buildLabelHistogram(labels, train_idx);
    const std::size_t class_count = hist.size();
    const double total = static_cast<double>(train_idx.size());

    std::vector<float> w(class_count, 0.0f);
    double sum_nonzero = 0.0;
    std::size_t nonzero = 0;
    for (std::size_t c = 0; c < class_count; ++c) {
        if (hist[c] == 0) {
            continue;
        }
        const double wc = total / (static_cast<double>(class_count) * static_cast<double>(hist[c]));
        w[c] = static_cast<float>(wc);
        sum_nonzero += wc;
        ++nonzero;
    }

    if (nonzero > 0 && sum_nonzero > 0.0) {
        const double mean_nonzero = sum_nonzero / static_cast<double>(nonzero);
        for (float& wc : w) {
            if (wc > 0.0f) {
                wc = static_cast<float>(wc / mean_nonzero);
            }
        }
    }

    return torch::tensor(w, torch::dtype(torch::kFloat32)).to(device);
}

static void buildTensorDataset(const std::vector<CsvRow>& rows,
                               std::vector<torch::Tensor>& inputs,
                               std::vector<int64_t>& labels) {
    inputs.clear();
    labels.clear();
    inputs.reserve(rows.size());
    labels.reserve(rows.size());

    for (const auto& row : rows) {
        const B1B2 state(row.state_b1, row.state_b2);
        const B1B2 goal(row.goal_b1, row.goal_b2);
        inputs.push_back(encodeBoardPair(state, goal, row.color_count));
        labels.push_back(static_cast<int64_t>(row.remaining_depth));
    }
}

struct EvalMetrics {
    double top1_acc = 0.0;
    double argmax_mae = 0.0;
    double expected_mae = 0.0;
};

static EvalMetrics evaluateModel(
        DistanceNet& model,
        const torch::Device& device,
        const std::vector<torch::Tensor>& inputs,
        const std::vector<int64_t>& labels,
        const std::vector<std::size_t>& idx,
        const int batch_size) {
    if (idx.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;
    model->eval();

    std::size_t correct = 0;
    std::size_t total = 0;
    double argmax_mae_sum = 0.0;
    double expected_mae_sum = 0.0;
    const auto class_ids = torch::arange(0, kMaxDepthClass + 1, torch::dtype(torch::kFloat32)).to(device);

    for (std::size_t start = 0; start < idx.size(); start += static_cast<std::size_t>(batch_size)) {
        const std::size_t end = std::min(start + static_cast<std::size_t>(batch_size), idx.size());

        std::vector<torch::Tensor> batch_tensors;
        std::vector<int64_t> batch_labels;
        batch_tensors.reserve(end - start);
        batch_labels.reserve(end - start);

        for (std::size_t i = start; i < end; ++i) {
            const std::size_t j = idx[i];
            batch_tensors.push_back(inputs[j]);
            batch_labels.push_back(labels[j]);
        }

        const auto x = torch::stack(batch_tensors).to(device);
        const auto y = torch::tensor(batch_labels, torch::kLong).to(device);
        const auto logits = model->forward(x);
        const auto pred = logits.argmax(1);
        const auto probs = torch::softmax(logits, 1);
        const auto expected = (probs * class_ids).sum(1);

        correct += static_cast<std::size_t>(pred.eq(y).sum().item<int64_t>());
        total += (end - start);

        const auto pred_float = pred.to(torch::kFloat32);
        const auto y_float = y.to(torch::kFloat32);
        argmax_mae_sum += torch::abs(pred_float - y_float).sum().item<double>();
        expected_mae_sum += torch::abs(expected - y_float).sum().item<double>();
    }

    EvalMetrics out;
    out.top1_acc = total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
    out.argmax_mae = total == 0 ? 0.0 : argmax_mae_sum / static_cast<double>(total);
    out.expected_mae = total == 0 ? 0.0 : expected_mae_sum / static_cast<double>(total);
    return out;
}

static bool trainModel(const TrainConfig& cfg) {
    const auto rows = loadCsvDataset(cfg.dataset_path);
    if (rows.empty()) {
        std::cerr << "[ERROR] dataset has no valid rows\n";
        return false;
    }

    std::vector<torch::Tensor> inputs;
    std::vector<int64_t> labels;
    buildTensorDataset(rows, inputs, labels);

    std::vector<std::size_t> train_idx;
    std::vector<std::size_t> val_idx;
    splitIndicesByBoardKey(rows, cfg.val_ratio, cfg.split_seed, train_idx, val_idx);
    if (train_idx.empty()) {
        std::cerr << "[ERROR] train split is empty\n";
        return false;
    }

    const torch::Device device(cfg.use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    DistanceNet model;
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(cfg.learning_rate).weight_decay(cfg.weight_decay));
    const auto class_weights = buildClassWeights(labels, train_idx, device);
    const auto class_ids = torch::arange(0, kMaxDepthClass + 1, torch::dtype(torch::kFloat32)).to(device);

    if (cfg.verbose) {
        std::cout << "[Dataset] rows=" << rows.size() << " train=" << train_idx.size() << " val=" << val_idx.size() << "\n";
        std::cout << "[Device] " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        const auto train_hist = buildLabelHistogram(labels, train_idx);
        std::cout << "[Train histogram]";
        for (std::size_t d = 0; d < train_hist.size(); ++d) {
            if (train_hist[d] > 0) {
                std::cout << " d" << d << '=' << train_hist[d];
            }
        }
        std::cout << "\n";
    }

    std::mt19937_64 epoch_rng(cfg.split_seed + 17);
    const bool has_val = !val_idx.empty();
    double best_val_expected_mae = std::numeric_limits<double>::infinity();
    bool saved_best = false;

    for (int epoch = 0; epoch < cfg.num_epochs; ++epoch) {
        std::shuffle(train_idx.begin(), train_idx.end(), epoch_rng);
        model->train();

        double loss_sum = 0.0;
        double ce_sum = 0.0;
        double reg_sum = 0.0;
        std::size_t batches = 0;

        for (std::size_t start = 0; start < train_idx.size(); start += static_cast<std::size_t>(cfg.batch_size)) {
            const std::size_t end = std::min(start + static_cast<std::size_t>(cfg.batch_size), train_idx.size());

            std::vector<torch::Tensor> batch_tensors;
            std::vector<int64_t> batch_labels;
            batch_tensors.reserve(end - start);
            batch_labels.reserve(end - start);

            for (std::size_t i = start; i < end; ++i) {
                const std::size_t j = train_idx[i];
                batch_tensors.push_back(inputs[j]);
                batch_labels.push_back(labels[j]);
            }

            const auto x = torch::stack(batch_tensors).to(device);
            const auto y = torch::tensor(batch_labels, torch::kLong).to(device);

            const auto logits = model->forward(x);
            const auto ce = torch::nn::functional::cross_entropy(
                    logits,
                    y,
                    torch::nn::functional::CrossEntropyFuncOptions()
                            .weight(class_weights)
                            .label_smoothing(cfg.label_smoothing));

            const auto probs = torch::softmax(logits, 1);
            const auto expected = (probs * class_ids).sum(1);
            const auto y_float = y.to(torch::kFloat32);
            const auto reg = torch::nn::functional::smooth_l1_loss(expected, y_float);

            const auto loss = cfg.ce_loss_weight * ce + cfg.reg_loss_weight * reg;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            loss_sum += loss.item<double>();
            ce_sum += ce.item<double>();
            reg_sum += reg.item<double>();
            ++batches;
        }

        EvalMetrics val_metrics{};
        if (has_val) {
            val_metrics = evaluateModel(model, device, inputs, labels, val_idx, cfg.batch_size);
            if (val_metrics.expected_mae < best_val_expected_mae) {
                best_val_expected_mae = val_metrics.expected_mae;
                torch::save(model, cfg.output_model_path);
                saved_best = true;
            }
        }

        if (cfg.verbose) {
            std::cout << "epoch " << (epoch + 1) << "/" << cfg.num_epochs
                      << " loss=" << std::fixed << std::setprecision(4)
                      << (batches == 0 ? 0.0 : loss_sum / static_cast<double>(batches))
                      << " ce=" << std::setprecision(4) << (batches == 0 ? 0.0 : ce_sum / static_cast<double>(batches))
                      << " reg=" << std::setprecision(4) << (batches == 0 ? 0.0 : reg_sum / static_cast<double>(batches));
            if (has_val) {
                std::cout << " val_acc=" << std::setprecision(3) << (100.0 * val_metrics.top1_acc) << "%"
                          << " val_mae_argmax=" << std::setprecision(3) << val_metrics.argmax_mae
                          << " val_mae_expected=" << std::setprecision(3) << val_metrics.expected_mae;
            }
            std::cout << "\n";
        }
    }

    if (!saved_best) {
        torch::save(model, cfg.output_model_path);
    }
    if (cfg.verbose) {
        if (saved_best) {
            std::cout << "[Saved best by val expected MAE] " << cfg.output_model_path << "\n";
        } else {
            std::cout << "[Saved final] " << cfg.output_model_path << "\n";
        }
    }
    return true;
}

} // namespace mindbender_ml

int trainDeep(const nlohmann::json& cfg_json) {
    using namespace mindbender_ml;
    TrainConfig cfg;

    cfg.dataset_path = cfg_json.value("dataset_path", std::string());
    cfg.output_model_path = cfg_json.value("output_model_path", std::string());
    cfg.batch_size = std::max(1, cfg_json.value("batch_size", cfg.batch_size));
    cfg.num_epochs = std::max(1, cfg_json.value("num_epochs", cfg.num_epochs));
    cfg.learning_rate = cfg_json.value("learning_rate", cfg.learning_rate);
    cfg.weight_decay = cfg_json.value("weight_decay", cfg.weight_decay);
    cfg.val_ratio = std::clamp(cfg_json.value("val_ratio", cfg.val_ratio), 0.0, 0.9);
    cfg.split_seed = cfg_json.value("split_seed", cfg.split_seed);
    cfg.label_smoothing = std::clamp(cfg_json.value("label_smoothing", cfg.label_smoothing), 0.0, 0.2);
    cfg.ce_loss_weight = std::max(0.0, cfg_json.value("ce_loss_weight", cfg.ce_loss_weight));
    cfg.reg_loss_weight = std::max(0.0, cfg_json.value("reg_loss_weight", cfg.reg_loss_weight));

    if (cfg_json.contains("use_cuda")) {
        const auto& v = cfg_json["use_cuda"];
        cfg.use_cuda = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
    }

    if (cfg_json.contains("verbose")) {
        const auto& v = cfg_json["verbose"];
        cfg.verbose = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
    }

    if (cfg.dataset_path.empty() || cfg.output_model_path.empty()) {
        std::cerr << "[ERROR] train config requires: dataset_path, output_model_path\n";
        return 1;
    }

    try {
        return trainModel(cfg) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}



