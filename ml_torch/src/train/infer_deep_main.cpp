#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "include/nlohmann/json.hpp"
#include "ml_torch/src/common/board_encoder.hpp"
#include "ml_torch/src/common/constants.hpp"
#include "ml_torch/src/model/distance_net.hpp"

namespace mindbender_ml {

    struct InferConfig {
        std::string dataset_path;
        std::string model_path;
        int batch_size = 128;
        double val_ratio = 0.1;
        uint64_t split_seed = 1337;
        int samples_to_print = 25;
        bool use_cuda = false;
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

    static std::vector<CsvRow> loadCsvDataset(const std::string& path) {
        std::vector<CsvRow> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("cannot open dataset: " + path);
        }

        std::string line;
        std::getline(file, line);

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
            data.push_back(row);
        }
        return data;
    }

    static std::vector<std::size_t> buildValSplit(const std::size_t n, const double val_ratio, const uint64_t seed) {
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
        const std::size_t val_count = static_cast<std::size_t>(static_cast<double>(n) * val_ratio);
        idx.resize(std::min(val_count, idx.size()));
        return idx;
    }

} // namespace mindbender_ml

int inferDeep(const nlohmann::json& cfg_json) {
    using namespace mindbender_ml;
    InferConfig cfg;

    cfg.dataset_path = cfg_json.value("dataset_path", std::string());
    cfg.model_path = cfg_json.value("model_path", std::string());
    cfg.batch_size = std::max(1, cfg_json.value("batch_size", cfg.batch_size));
    cfg.val_ratio = std::clamp(cfg_json.value("val_ratio", cfg.val_ratio), 0.0, 0.9);
    cfg.split_seed = cfg_json.value("split_seed", cfg.split_seed);
    cfg.samples_to_print = std::max(1, cfg_json.value("samples_to_print", cfg.samples_to_print));

    if (cfg_json.contains("use_cuda")) {
        const auto& v = cfg_json["use_cuda"];
        cfg.use_cuda = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
    }

    if (cfg.dataset_path.empty() || cfg.model_path.empty()) {
        std::cerr << "[ERROR] infer config requires: dataset_path, model_path\n";
        return 1;
    }

    try {
        const auto rows = loadCsvDataset(cfg.dataset_path);
        if (rows.empty()) {
            std::cerr << "[ERROR] dataset empty\n";
            return 1;
        }

        auto val_idx = buildValSplit(rows.size(), cfg.val_ratio, cfg.split_seed);
        if (val_idx.empty()) {
            std::cerr << "[ERROR] holdout split empty\n";
            return 1;
        }

        const torch::Device device(cfg.use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        DistanceNet model;
        torch::load(model, cfg.model_path);
        model->to(device);
        model->eval();

        std::size_t correct = 0;
        double mae_sum = 0.0;
        std::vector<std::tuple<int, int, double>> samples;

        torch::NoGradGuard no_grad;

        for (std::size_t start = 0; start < val_idx.size(); start += static_cast<std::size_t>(cfg.batch_size)) {
            const std::size_t end = std::min(start + static_cast<std::size_t>(cfg.batch_size), val_idx.size());

            std::vector<torch::Tensor> batch_tensors;
            std::vector<int64_t> labels;
            batch_tensors.reserve(end - start);
            labels.reserve(end - start);

            for (std::size_t i = start; i < end; ++i) {
                const auto& row = rows[val_idx[i]];
                batch_tensors.push_back(encodeBoardPair(B1B2(row.state_b1, row.state_b2), B1B2(row.goal_b1, row.goal_b2), row.color_count));
                labels.push_back(row.remaining_depth);
            }

            const auto x = torch::stack(batch_tensors).to(device);
            const auto y = torch::tensor(labels, torch::kLong).to(device);
            const auto logits = model->forward(x);
            const auto probs = torch::softmax(logits, 1);
            const auto pred = probs.argmax(1);

            correct += static_cast<std::size_t>(pred.eq(y).sum().item<int64_t>());
            mae_sum += torch::abs(pred.to(torch::kFloat32) - y.to(torch::kFloat32)).sum().item<double>();

            if (static_cast<int>(samples.size()) < cfg.samples_to_print) {
                const auto k = static_cast<int64_t>(std::min<std::size_t>(cfg.samples_to_print - samples.size(), end - start));
                for (int64_t i = 0; i < k; ++i) {
                    const int truth = static_cast<int>(y[i].item<int64_t>());
                    const int guess = static_cast<int>(pred[i].item<int64_t>());
                    const double conf = probs[i][guess].item<double>();
                    samples.emplace_back(truth, guess, conf);
                }
            }
        }

        const double acc = static_cast<double>(correct) / static_cast<double>(val_idx.size());
        const double mae = mae_sum / static_cast<double>(val_idx.size());

        std::cout << "[Holdout] size=" << val_idx.size()
                  << " acc=" << std::fixed << std::setprecision(3) << (100.0 * acc) << "%"
                  << " mae=" << std::setprecision(3) << mae << "\n";

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto& [truth, guess, conf] = samples[i];
            std::cout << "sample[" << i << "] truth=" << truth
                      << " pred=" << guess
                      << " conf=" << std::setprecision(3) << conf << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}