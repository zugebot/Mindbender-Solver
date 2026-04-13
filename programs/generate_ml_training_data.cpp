// programs/generate_ml_training_data.cpp
#include "code/include.hpp"
#include "code/solver/solver_frontier.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "include/nlohmann/json.hpp"

namespace {
constexpr int kMlTargetDepthDefault = 10;
constexpr int kMlMaxDepthLabel = 13;
constexpr const char* kConfigPath = "config_ml_training_data.json";
constexpr int kNormalNoneMoveCount = 60;
constexpr int kFatNoneMoveCount = 48;
}

struct DataGenConfig {
    std::string output_path = "train/ml_training_data.csv";
    std::string base_puzzle = "12-1";
    int num_puzzles = 100;
    int target_depth = kMlTargetDepthDefault;
    int scramble_extra_moves = 1;
    std::vector<int> color_counts = {6, 6, 6, 6, 6, 6};
    bool verbose = true;
    uint64_t seed = 1337;
    int threads = 8;
};

static u32 pickRandomNoneMoveIndex(const Board& board, std::mt19937_64& rng) {
    if (board.getFatBool()) {
        std::uniform_int_distribution<> dist(0, kFatNoneMoveCount - 1);
        return fatActionsIndexes[board.getFatXY()][dist(rng)];
    }

    std::uniform_int_distribution<> dist(0, kNormalNoneMoveCount - 1);
    const int idx = dist(rng);
    if (idx < static_cast<int>(NORMAL_ROW_MOVE_COUNT)) {
        return static_cast<u32>(idx);
    }

    const int col_idx = idx - static_cast<int>(NORMAL_ROW_MOVE_COUNT);
    return NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT + static_cast<u32>(col_idx);
}

static Board generateStartByRandomWalk(const Board& goal, const int target_depth, std::mt19937_64& rng) {
    Board start = goal;
    for (int i = 0; i < target_depth; ++i) {
        const u32 move_idx = pickRandomNoneMoveIndex(start, rng);
        allActStructList[move_idx].action(start);
    }
    return start;
}

static bool hasExactDepthSolutionShallow(const Board& start, const Board& goal, const int depth) {
    FrontierBuilderB1B2 builder(start);
    JVec<B1B2> frontier;
    JVec<u64> hashes;
    builder.buildExactNoneDepth(depth, frontier, hashes);

    const B1B2 goal_state = goal.asB1B2();
    for (const auto& s : frontier) {
        if (s == goal_state) {
            return true;
        }
    }
    return false;
}

static void collectNoneChildren(const Board& parent, std::vector<Board>& out) {
    out.clear();

    if (parent.getFatBool()) {
        const u8* actions = fatActionsIndexes[parent.getFatXY()];
        out.reserve(kFatNoneMoveCount);
        for (int i = 0; i < kFatNoneMoveCount; ++i) {
            Board child = parent;
            allActStructList[actions[i]].action(child);
            if (child == parent) {
                continue;
            }
            out.push_back(child);
        }
        return;
    }

    out.reserve(kNormalNoneMoveCount);
    for (u32 act = 0; act < NORMAL_ROW_MOVE_COUNT; ++act) {
        Board child = parent;
        allActStructList[act].action(child);
        if (child == parent) {
            continue;
        }
        out.push_back(child);
    }

    for (u32 act = NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT;
         act < NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT + NORMAL_COL_MOVE_COUNT;
         ++act) {
        Board child = parent;
        allActStructList[act].action(child);
        if (child == parent) {
            continue;
        }
        out.push_back(child);
    }
}

template<bool DEBUG>
static int runSolverForEstimatedDepth(
        BoardSolverFrontier& solver,
        const int estimatedDepth,
        const int threads,
        const BoardSolverFrontier::SearchDirection searchDirection,
        const bool ensureNoLowerSolutions,
        const bool enableDepth5RightCache
) {
    switch (estimatedDepth) {
        case 3:
            return solver.findSolutionsFrontierThreaded<1, 1, 1, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 4:
            return solver.findSolutionsFrontierThreaded<1, 1, 2, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 5:
            return solver.findSolutionsFrontierThreaded<1, 1, 3, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 6:
            return solver.findSolutionsFrontierThreaded<1, 1, 4, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 7:
            return solver.findSolutionsFrontierThreaded<1, 2, 4, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 8:
            return solver.findSolutionsFrontierThreaded<1, 3, 4, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 9:
            return solver.findSolutionsFrontierThreaded<1, 4, 4, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 10:
            return solver.findSolutionsFrontierThreaded<1, 4, 5, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 11:
            return solver.findSolutionsFrontierThreaded<1, 5, 5, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 12:
            return solver.findSolutionsFrontierThreaded<2, 5, 5, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        case 13:
            return solver.findSolutionsFrontierThreaded<3, 5, 5, DEBUG>(threads, searchDirection, ensureNoLowerSolutions, enableDepth5RightCache);
        default:
            return -1;
    }
}

static void emitCsvRow(std::ofstream& output_file,
                       const Board& state,
                       const Board& goal,
                       const int remaining_depth,
                       const int color_count,
                       const bool is_fat) {
    output_file << std::hex
                << "0x" << state.b1 << ","
                << "0x" << state.b2 << ","
                << "0x" << goal.b1 << ","
                << "0x" << goal.b2 << ","
                << std::dec
                << remaining_depth << ","
                << color_count << ","
                << (is_fat ? 1 : 0) << "\n";
}

static bool loadConfigFromJson(DataGenConfig& cfg, std::string& loadedPath) {
    std::ifstream in(kConfigPath);
    if (!in.is_open()) {
        return false;
    }

    try {
        nlohmann::json j;
        in >> j;

        if (const auto it = j.find("output_path"); it != j.end() && it->is_string()) {
            cfg.output_path = it->get<std::string>();
        }
        if (const auto it = j.find("base_puzzle"); it != j.end() && it->is_string()) {
            cfg.base_puzzle = it->get<std::string>();
        }
        if (const auto it = j.find("num_puzzles"); it != j.end() && it->is_number_integer()) {
            cfg.num_puzzles = std::max(1, it->get<int>());
        }
        if (const auto it = j.find("target_depth"); it != j.end() && it->is_number_integer()) {
            cfg.target_depth = std::max(0, it->get<int>());
        }
        if (const auto it = j.find("scramble_extra_moves"); it != j.end() && it->is_number_integer()) {
            cfg.scramble_extra_moves = std::max(0, std::min(16, it->get<int>()));
        }
        if (const auto it = j.find("threads"); it != j.end() && it->is_number_integer()) {
            cfg.threads = std::max(1, it->get<int>());
        }
        if (const auto it = j.find("seed"); it != j.end() && it->is_number_unsigned()) {
            cfg.seed = it->get<uint64_t>();
        }
        if (const auto it = j.find("verbose"); it != j.end() && it->is_boolean()) {
            cfg.verbose = it->get<bool>();
        }
        if (const auto it = j.find("color_counts"); it != j.end() && it->is_array()) {
            std::vector<int> parsed;
            for (const auto& v : *it) {
                if (v.is_number_integer()) {
                    const int c = v.get<int>();
                    if (c >= 0) {
                        parsed.push_back(c);
                    }
                }
            }
            if (!parsed.empty()) {
                cfg.color_counts = parsed;
            }
        }

        loadedPath = kConfigPath;
        return true;
    } catch (...) {
        return false;
    }
}

int main() {
    DataGenConfig config;
    std::string loadedConfigPath;
    if (!loadConfigFromJson(config, loadedConfigPath)) {
        std::cerr << "[ERROR] Missing/invalid config JSON. Create:\n"
                  << "  - " << kConfigPath << "\n";
        return 1;
    }

    const auto base_pair = BoardLookup::getBoardPair(config.base_puzzle);
    if (!base_pair) {
        std::cerr << "[ERROR] Invalid base_puzzle: " << config.base_puzzle << "\n";
        return 1;
    }

    const Board fixed_goal = base_pair->getEndState();
    const int color_count = static_cast<int>(fixed_goal.getColorCount());
    const bool is_fat = fixed_goal.getFatBool();

    if (config.target_depth < 0 || config.target_depth > 13) {
        std::cerr << "[ERROR] target_depth must be in [0,13].\n";
        return 1;
    }

    const int scramble_depth = std::min(64, config.target_depth + config.scramble_extra_moves);

    if (config.verbose) {
        std::cout << "\n========================================\n"
                  << "  ML TRAINING DATA GENERATOR\n"
                  << "========================================\n\n"
                  << "[Config file] " << loadedConfigPath << "\n"
                  << "[Config]\n"
                  << "  Output: " << config.output_path << "\n"
                  << "  Base puzzle: " << config.base_puzzle << "\n"
                  << "  Puzzles: " << config.num_puzzles << "\n"
                  << "  Target depth: " << config.target_depth << "\n"
                  << "  Scramble extra moves: " << config.scramble_extra_moves << "\n"
                  << "  Effective scramble depth: " << scramble_depth << "\n"
                  << "  Threads: " << config.threads << "\n"
                  << "  Color counts: ";
        for (const int c : config.color_counts) {
            std::cout << c << " ";
        }
        std::cout << "\n\n";
    }

    std::mt19937_64 rng(config.seed);
    std::ofstream output_file(config.output_path);
    if (!output_file.is_open()) {
        std::cerr << "[ERROR] Cannot open output file: " << config.output_path << "\n";
        return 1;
    }

    output_file << "start_board_b1,start_board_b2,goal_board_b1,goal_board_b2,"
                << "remaining_depth,color_count,is_fat\n";

    std::size_t total_data_points = 0;
    std::size_t total_accepted = 0;
    std::size_t total_attempts = 0;
    Timer total_timer;

    auto printProgress = [&]() {
        if (!config.verbose) {
            return;
        }

        tcout.setProgressPrefix(total_accepted, static_cast<std::size_t>(config.num_puzzles));
        tcout << "attempts=" << total_attempts << ", data=" << total_data_points << '\n';
    };

    while (total_accepted < static_cast<std::size_t>(config.num_puzzles)) {
        ++total_attempts;
        try {
            const Board goal = fixed_goal;
            const Board start = generateStartByRandomWalk(fixed_goal, scramble_depth, rng);

            BoardPair pair("ML", start, goal);
            BoardSolverFrontier solver(&pair);
            solver.setWriteSolutionsToFile(false);

            int exact_count = 0;
            if (config.target_depth == 0) {
                exact_count = (start == goal) ? 1 : 0;
            } else if (config.target_depth <= 2) {
                if (start == goal) {
                    continue;
                }

                if (config.target_depth == 2) {
                    FrontierBuilderB1B2 depth1Builder(start);
                    JVec<B1B2> depth1Frontier;
                    JVec<u64> depth1Hashes;
                    depth1Builder.buildExactNoneDepth(1, depth1Frontier, depth1Hashes);
                    const B1B2 goal_state_depth1 = goal.asB1B2();
                    bool hasDepth1 = false;
                    for (const auto& s : depth1Frontier) {
                        if (s == goal_state_depth1) {
                            hasDepth1 = true;
                            break;
                        }
                    }
                    if (hasDepth1) {
                        continue;
                    }
                }

                FrontierBuilderB1B2 builder(start);
                JVec<B1B2> frontier;
                JVec<u64> hashes;
                builder.buildExactNoneDepth(config.target_depth, frontier, hashes);
                const B1B2 goal_state = goal.asB1B2();
                for (const auto& s : frontier) {
                    if (s == goal_state) {
                        exact_count = 1;
                        break;
                    }
                }
            } else {
                exact_count = runSolverForEstimatedDepth<false>(
                        solver,
                        config.target_depth,
                        config.threads,
                        BoardSolverFrontier::SearchDirection::Auto,
                        true,
                        true
                );
            }

            if (exact_count <= 0) {
                printProgress();
                continue;
            }

            if (config.target_depth <= 2) {
                std::unordered_set<u64> emitted_hashes;

                const u64 start_hash = StateHash::computeHash(start);
                if (emitted_hashes.insert(start_hash).second) {
                    emitCsvRow(output_file, start, goal, config.target_depth, color_count, is_fat);
                    ++total_data_points;
                }

                if (config.target_depth == 2) {
                    std::vector<Board> children;
                    collectNoneChildren(start, children);
                    for (const auto& child : children) {
                        if (!hasExactDepthSolutionShallow(child, goal, 1)) {
                            continue;
                        }

                        const u64 child_hash = StateHash::computeHash(child);
                        if (!emitted_hashes.insert(child_hash).second) {
                            continue;
                        }

                        emitCsvRow(output_file, child, goal, 1, color_count, is_fat);
                        ++total_data_points;
                    }
                }

                ++total_accepted;
                printProgress();
                continue;
            }

            std::unordered_set<u64> emitted_hashes;
            const auto solutions = solver.getExpandedSolutionsList();

            for (const auto& solution : solutions) {
                std::vector<u8> moves;
                try {
                    moves = is_fat ? Memory::parseFatMoveString(solution) : Memory::parseNormMoveString(solution);
                } catch (...) {
                    continue;
                }

                if (static_cast<int>(moves.size()) != config.target_depth) {
                    continue;
                }

                Board cursor = start;
                const u64 start_hash = StateHash::computeHash(cursor);
                if (emitted_hashes.insert(start_hash).second) {
                    emitCsvRow(output_file, cursor, goal, config.target_depth, color_count, is_fat);
                    ++total_data_points;
                }

                for (int step = 0; step < static_cast<int>(moves.size()); ++step) {
                    allActStructList[moves[step]].action(cursor);
                    const int remaining = config.target_depth - (step + 1);
                    if (remaining <= 0) {
                        continue;
                    }

                    const u64 hash = StateHash::computeHash(cursor);
                    if (!emitted_hashes.insert(hash).second) {
                        continue;
                    }

                    emitCsvRow(output_file, cursor, goal, remaining, color_count, is_fat);
                    ++total_data_points;
                }
            }

            ++total_accepted;
            printProgress();
        } catch (const std::exception& e) {
            if (config.verbose) {
                std::cerr << "  [Warning] Attempt " << total_attempts << " failed: " << e.what() << "\n";
            }
            printProgress();
        }
    }

    output_file.close();

    if (config.verbose) {
        tcout.clearLinePrefix();
        std::cout << "\n[Complete]\n"
                  << "  Total time: " << std::fixed << std::setprecision(2) << total_timer.getSeconds() << "s\n"
                  << "  Attempts: " << total_attempts << "\n"
                  << "  Accepted puzzles: " << total_accepted << "\n"
                  << "  Data points generated: " << total_data_points << "\n"
                  << "  Output: " << config.output_path << "\n\n";
    }

    return 0;
}

