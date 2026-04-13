#include <fstream>
#include <iostream>
#include <string>

#include "include/nlohmann/json.hpp"

int trainDeep(const nlohmann::json& cfg_json);
int inferDeep(const nlohmann::json& cfg_json);

int main(int argc, char** argv) {
    const std::string config_path = (argc >= 2) ? argv[1] : "config_ml_torch.json";

    std::ifstream in(config_path);
    if (!in.is_open()) {
        std::cerr << "[ERROR] cannot open config file: " << config_path << "\n";
        return 1;
    }

    nlohmann::json root;
    try {
        in >> root;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] invalid config JSON: " << e.what() << "\n";
        return 1;
    }

    const std::string mode = root.value("mode", std::string());
    if (mode == "train") {
        if (!root.contains("train") || !root["train"].is_object()) {
            std::cerr << "[ERROR] config missing object: train\n";
            return 1;
        }
        return trainDeep(root["train"]);
    }

    if (mode == "infer") {
        if (!root.contains("infer") || !root["infer"].is_object()) {
            std::cerr << "[ERROR] config missing object: infer\n";
            return 1;
        }
        return inferDeep(root["infer"]);
    }

    std::cerr << "[ERROR] config field 'mode' must be 'train' or 'infer'\n";
    std::cerr << "[Hint] using config file: " << config_path << "\n";
    return 1;
}

