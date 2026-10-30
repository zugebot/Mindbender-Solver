#pragma once
// src/solver/ssd/cli.hpp
//
// Command-line flags -> Config (documented at the top of
// src/programs/solvers/experimentDiskDepth6.cpp).

#include "src/solver/ssd/common.hpp"

namespace ssd {

    inline Config parseArgs(const int argc, char** argv) {
        Config cfg;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + a);
                }
                return argv[++i];
            };
            if (a == "--out") cfg.outRoot = next();
            else if (a == "--batch") cfg.batchStates = std::stoull(next());
            else if (a == "--threads") cfg.threads = static_cast<u32>(std::stoul(next()));
            else if (a == "--seed-depth") cfg.seedDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--reverse") cfg.reverse = true;
            else if (a == "--rebuild") cfg.rebuild = true;
            else if (a == "--no-recover") cfg.recover = false;
            else if (a == "--batched") cfg.batched = true;
            else if (a == "--ram-gb") cfg.ramBytes = static_cast<u64>(std::stod(next()) * 1e9);
            else if (a == "--no-large-pages") cfg.largePages = false;
            else if (a == "--solutions") cfg.solutionsFile = next();
            else if (a == "--d7-sample") cfg.d7Sample = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-pass-budget") cfg.d7PassBudget = std::stoull(next());
            else if (a == "--table-depth") cfg.tableDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-partitions") cfg.d7Partitions = static_cast<u32>(std::stoul(next()));
            else if (a == "--no-goal-filter") cfg.goalFilter = false;
            else if (a == "--recover-stream-depth") cfg.recoverStreamDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-counts-ram") cfg.d7CountsInRam = true;
            else if (a == "--seed-last-move") cfg.seedLastMove = true;
            else if (a == "--no-seed-last-move") cfg.seedLastMove = false;
            else if (a == "--d7-first-partition") cfg.d7FirstPartition = static_cast<u32>(std::stoul(next()));
            else if (a == "--progress-file") cfg.progressFile = next();
            else if (a == "--progress-interval") cfg.progressInterval = std::stod(next());
            else if (a == "--no-calibrate") cfg.calibrate = false;
            else if (a == "--ram-build") cfg.ramBuild = true;
            else if (a == "--disk-table") cfg.diskTable = true;
            else if (a == "--calibrate-seconds") cfg.calibrateSeconds = std::stod(next());
            else if (a == "--bound-calibrate-seconds") cfg.boundCalibrateSeconds = std::stod(next());
            else if (a == "--bound-kind") {
                const std::string v = next();
                if (v != "mix" && v != "rotation") {
                    throw std::runtime_error("--bound-kind must be mix or rotation");
                }
                cfg.boundKind = v == "mix" ? lane_bound::Kind::Mix : lane_bound::Kind::Rotation;
            }
            else if (a == "--bound") {
                const std::string v = next();
                cfg.boundMode = v == "auto" ? -1 : v == "off" ? 0 : v == "leaf" ? 1 : v == "both" ? 2
                              : v == "pdb" ? 3 : v == "pdbleaf" ? 4 : v == "parent" ? 5 : v == "pdbparent" ? 6 : -2;
                if (cfg.boundMode == -2) {
                    throw std::runtime_error("--bound must be auto, off, leaf, both, pdb, pdbleaf, parent or pdbparent");
                }
                if (cfg.boundMode == 3 || cfg.boundMode == 4 || cfg.boundMode == 6) {
                    cfg.pdbMode = 1;
                }
            }
            else if (a == "--pdb") {
                const std::string v = next();
                cfg.pdbMode = v == "auto" ? -1 : v == "off" ? 0 : v == "on" ? 1 : -2;
                if (cfg.pdbMode == -2) {
                    throw std::runtime_error("--pdb must be auto, off or on");
                }
            }
            else if (a == "--pdb-max-mentries") cfg.pdbMaxEntries = std::stoull(next()) * 1'000'000ULL;
            else if (a == "--stream-depth") cfg.streamDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--shard") {
                const std::string v = next();
                const std::size_t slash = v.find('/');
                if (slash == std::string::npos) {
                    throw std::runtime_error("--shard must be k/N");
                }
                cfg.shardIndex = static_cast<u32>(std::stoul(v.substr(0, slash)));
                cfg.shardCount = static_cast<u32>(std::stoul(v.substr(slash + 1)));
                if (cfg.shardCount == 0 || cfg.shardIndex >= cfg.shardCount) {
                    throw std::runtime_error("--shard k/N needs 0 <= k < N");
                }
            }
            else if (a == "--middles-out") cfg.middlesOut = next();
            else if (a == "--recover-from") cfg.recoverFrom.push_back(next());
            else if (a == "--probe-variant") {
                const std::string v = next();
                const auto colon = v.find(':');
                cfg.forcedGroup = std::stoi(v.substr(0, colon));
                cfg.forcedPipeline = colon == std::string::npos ? 0 : std::stoi(v.substr(colon + 1));
            }
            else cfg.puzzle = a;
        }
        if (cfg.threads == 0) {
            cfg.threads = std::max(1U, std::thread::hardware_concurrency());
        }
        if (cfg.tableDepth != 6 && cfg.tableDepth != 7) {
            throw std::runtime_error("--table-depth must be 6 or 7");
        }
        if (cfg.tableDepth == 7 && (cfg.reverse || cfg.batched)) {
            throw std::runtime_error("--table-depth 7 supports forward fingerprint mode only");
        }
        cfg.d7FirstPartition = std::min<u32>(cfg.d7FirstPartition, PARTITION_COUNT - 1);
        cfg.d7Partitions = std::clamp<u32>(cfg.d7Partitions, 1, PARTITION_COUNT - cfg.d7FirstPartition);
        return cfg;
    }

} // namespace ssd
