#include "code/include.hpp"
#include "code/solver/frontier_builder.hpp"
#include "code/heuristics/heuristics.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
# include <windows.h>
static void force_utf8_console()
{
    // 65001 == CP_UTF8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // Enable ANSI / VT processing so UTF-8 + colours work in Win10+
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#endif

namespace {

    constexpr int kSeedDepth = 2;
    constexpr int kMaxLookaheadDepth = 3;
    constexpr int kLookaheadDepthCount = kMaxLookaheadDepth + 1;
    constexpr std::size_t kPlacementLineWidth = 100;

    constexpr int kMinTrainingPuzzleLength = 9;
    constexpr std::size_t kHeuristicCount = heur::kHeuristicCount;
    constexpr std::size_t kLearnedDepthFeatureCount = kHeuristicCount * static_cast<std::size_t>(kLookaheadDepthCount);
    constexpr std::size_t kLearnedDeltaFeatureCount = kHeuristicCount * static_cast<std::size_t>(kLookaheadDepthCount - 1);
    constexpr std::size_t kLearnedFeatureCount = kLearnedDepthFeatureCount + kLearnedDeltaFeatureCount;

    enum class BenchmarkScope {
        SinglePuzzle,
        AllNonFat
    };

    enum class LearnedModelMode {
        Disabled,
        LoadFromDisk,
        TrainAndBenchmark,
        TrainOnly
    };

    struct RunConfig {
        BenchmarkScope scope;
        LearnedModelMode modelMode;
    };

    // Default is equivalent to: kRunAllNonFatPuzzles=true, kTrainLearnedBlendModel=true, kTrainOnly=false.
    constexpr RunConfig kRunConfig{
            BenchmarkScope::AllNonFat,
            LearnedModelMode::LoadFromDisk
            // LearnedModelMode::TrainAndBenchmark
    };

    const fs::path kLevelsFinalDir = "levels_final";
    const fs::path kLearnedModelFile = "train/learned_blend_model.txt";
    constexpr const char* kBlendedHeuristicName = "blended_weighted_norm";
    constexpr const char* kLearnedHeuristicName = "learned_multi_depth_logreg";
    constexpr auto kBlendWeights = heur::kBlendWeights;

    // constexpr const char* kPuzzleName = "7-4";
    // const fs::path kSolutionsFile = "levels_final/7-4_c12_2696.txt";

    // constexpr const char* kPuzzleName = "9-2";
    // const fs::path kSolutionsFile = "levels_final/9-2_c12_1004.txt";

    constexpr const char* kPuzzleName = "20-1";
    const fs::path kSolutionsFile = "levels_final/20-1_c12_3155.txt";

    // constexpr const char* kPuzzleName = "14-2";
    // const fs::path kSolutionsFile = "levels_final/14-2_c12_8.txt";

    constexpr auto kHeuristicNames = heur::kHeuristicNames;

    struct SeedEval {
        B1B2 seed{};
        u64 seedHash = 0;
        std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount> lookaheadScores{};
        bool isSolutionSeed = false;
    };

    struct BenchmarkTarget {
        std::string puzzleName;
        fs::path solutionsFile;
    };

    struct LearnedBlendModel {
        std::array<double, kLearnedFeatureCount> weights{};
        double bias = 0.0;
        bool valid = false;
    };

    LearnedBlendModel gLearnedModel;

    static const char* toString(const BenchmarkScope scope) {
        switch (scope) {
            case BenchmarkScope::SinglePuzzle: return "single-puzzle";
            case BenchmarkScope::AllNonFat: return "all-non-fat";
        }
        return "unknown";
    }

    static const char* toString(const LearnedModelMode mode) {
        switch (mode) {
            case LearnedModelMode::Disabled: return "disabled";
            case LearnedModelMode::LoadFromDisk: return "load-from-disk";
            case LearnedModelMode::TrainAndBenchmark: return "train-and-benchmark";
            case LearnedModelMode::TrainOnly: return "train-only";
        }
        return "unknown";
    }

    static std::string trimCopy(const std::string& s) {
        const std::size_t first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const std::size_t last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    static std::vector<std::string> readNonEmptyLines(const fs::path& filePath) {
        std::vector<std::string> lines;
        std::ifstream in(filePath);
        if (!in.is_open()) {
            return lines;
        }

        std::string line;
        while (std::getline(in, line)) {
            line = trimCopy(line);
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    static double sigmoidStable(const double x) {
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        const double z = std::exp(x);
        return z / (1.0 + z);
    }

    static void computeDepthMinMax(const std::vector<SeedEval>& rows,
                                   std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount>& mins,
                                   std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount>& maxs) {
        for (int d = 0; d < kLookaheadDepthCount; ++d) {
            mins[static_cast<std::size_t>(d)].fill(1000000000);
            maxs[static_cast<std::size_t>(d)].fill(-1000000000);
        }

        for (const auto& row : rows) {
            for (int d = 0; d < kLookaheadDepthCount; ++d) {
                const auto& scores = row.lookaheadScores[static_cast<std::size_t>(d)];
                auto& dMin = mins[static_cast<std::size_t>(d)];
                auto& dMax = maxs[static_cast<std::size_t>(d)];
                for (std::size_t h = 0; h < kHeuristicCount; ++h) {
                    dMin[h] = std::min(dMin[h], scores[h]);
                    dMax[h] = std::max(dMax[h], scores[h]);
                }
            }
        }
    }

    static std::array<double, kLearnedFeatureCount> makeCombinedFeatures(
            const SeedEval& row,
            const std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount>& mins,
            const std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount>& maxs) {
        std::array<double, kLearnedFeatureCount> f{};
        std::array<std::array<double, kHeuristicCount>, kLookaheadDepthCount> normByDepth{};

        std::size_t offset = 0;
        for (int d = 0; d < kLookaheadDepthCount; ++d) {
            const auto& scores = row.lookaheadScores[static_cast<std::size_t>(d)];
            const auto& dMin = mins[static_cast<std::size_t>(d)];
            const auto& dMax = maxs[static_cast<std::size_t>(d)];
            auto& nDepth = normByDepth[static_cast<std::size_t>(d)];
            for (std::size_t h = 0; h < kHeuristicCount; ++h) {
                const double n = (dMax[h] > dMin[h])
                                         ? (static_cast<double>(scores[h] - dMin[h]) / static_cast<double>(dMax[h] - dMin[h]))
                                         : 0.5;
                nDepth[h] = n;
                f[offset + h] = n;
            }
            offset += kHeuristicCount;
        }

        for (int d = 1; d < kLookaheadDepthCount; ++d) {
            const auto& prev = normByDepth[static_cast<std::size_t>(d - 1)];
            const auto& cur = normByDepth[static_cast<std::size_t>(d)];
            for (std::size_t h = 0; h < kHeuristicCount; ++h) {
                f[offset + h] = 0.5 + 0.5 * (cur[h] - prev[h]);
            }
            offset += kHeuristicCount;
        }

        if (offset != kLearnedFeatureCount) {
            f.fill(0.5);
        }
        return f;
    }
    static_assert(kLookaheadDepthCount >= 2, "learned model expects at least depths 0 and 1");

    static bool saveLearnedModel(const LearnedBlendModel& model, const fs::path& filePath) {
        try {
            if (filePath.has_parent_path()) {
                fs::create_directories(filePath.parent_path());
            }
        } catch (...) {
            return false;
        }

        std::ofstream out(filePath);
        if (!out.is_open()) {
            return false;
        }

        out << "mindbender_blend_model_v1\n";
        out << "feature_count " << kLearnedFeatureCount << "\n";
        out << std::setprecision(17) << "bias " << model.bias << "\n";
        out << "weights";
        for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
            out << ' ' << model.weights[i];
        }
        out << "\n";
        return true;
    }

    static bool loadLearnedModel(const fs::path& filePath, LearnedBlendModel& modelOut) {
        std::ifstream in(filePath);
        if (!in.is_open()) {
            return false;
        }

        std::string magic;
        std::getline(in, magic);
        if (magic != "mindbender_blend_model_v1") {
            return false;
        }

        std::string token;
        std::size_t featureCount = 0;
        in >> token >> featureCount;
        if (!in || token != "feature_count" || featureCount != kLearnedFeatureCount) {
            return false;
        }

        in >> token >> modelOut.bias;
        if (!in || token != "bias") {
            return false;
        }

        in >> token;
        if (!in || token != "weights") {
            return false;
        }
        for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
            in >> modelOut.weights[i];
            if (!in) {
                return false;
            }
        }
        modelOut.valid = true;
        return true;
    }

    static std::vector<std::size_t> buildOrderByLearnedModel(const std::vector<SeedEval>& rows,
                                                             const LearnedBlendModel& model) {
        std::vector<std::size_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0);

        std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount> mins{}, maxs{};
        computeDepthMinMax(rows, mins, maxs);

        auto score = [&](const SeedEval& row) {
            const auto f = makeCombinedFeatures(row, mins, maxs);
            double z = model.bias;
            for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
                z += model.weights[i] * f[i];
            }
            return z;
        };

        std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
            const double sa = score(rows[a]);
            const double sb = score(rows[b]);
            if (sa != sb) {
                return sa > sb;
            }
            if (rows[a].seedHash != rows[b].seedHash) {
                return rows[a].seedHash < rows[b].seedHash;
            }
            return a < b;
        });
        return order;
    }

    static std::vector<std::size_t> buildOrderByHeuristic(const std::vector<SeedEval>& rows,
                                                          const int heuristicIndex,
                                                          const int lookaheadDepth) {
        std::vector<std::size_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
            const int sa = rows[a].lookaheadScores[lookaheadDepth][heuristicIndex];
            const int sb = rows[b].lookaheadScores[lookaheadDepth][heuristicIndex];
            if (sa != sb) {
                return sa < sb;
            }
            if (rows[a].seedHash != rows[b].seedHash) {
                return rows[a].seedHash < rows[b].seedHash;
            }
            return a < b;
        });

        return order;
    }

    static std::vector<std::size_t> buildOrderByBlendedHeuristic(const std::vector<SeedEval>& rows,
                                                                 const int lookaheadDepth) {
        std::vector<std::size_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0);

        std::array<int, kHeuristicCount> mins{};
        std::array<int, kHeuristicCount> maxs{};
        mins.fill(1000000000);
        maxs.fill(-1000000000);

        for (const auto& row : rows) {
            const auto& scores = row.lookaheadScores[lookaheadDepth];
            for (std::size_t h = 0; h < kHeuristicCount; ++h) {
                mins[h] = std::min(mins[h], scores[h]);
                maxs[h] = std::max(maxs[h], scores[h]);
            }
        }

        auto blendedScore = [&](const SeedEval& row) {
            const auto& scores = row.lookaheadScores[lookaheadDepth];
            double sum = 0.0;
            for (std::size_t h = 0; h < kHeuristicCount; ++h) {
                const int lo = mins[h];
                const int hi = maxs[h];
                const double norm = (hi > lo)
                                            ? (static_cast<double>(scores[h] - lo) / static_cast<double>(hi - lo))
                                            : 0.5;
                sum += kBlendWeights[h] * norm;
            }
            return sum;
        };

        std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
            const double sa = blendedScore(rows[a]);
            const double sb = blendedScore(rows[b]);
            if (sa != sb) {
                return sa < sb;
            }
            if (rows[a].seedHash != rows[b].seedHash) {
                return rows[a].seedHash < rows[b].seedHash;
            }
            return a < b;
        });

        return order;
    }

    static std::vector<std::size_t> buildOracleOrder(const std::vector<SeedEval>& rows) {
        std::vector<std::size_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
            if (rows[a].isSolutionSeed != rows[b].isSolutionSeed) {
                return rows[a].isSolutionSeed;
            }
            return a < b;
        });

        return order;
    }

    static u64 topKPositiveCount(const std::vector<SeedEval>& rows,
                                 const std::vector<std::size_t>& order,
                                 const std::size_t k) {
        const std::size_t n = std::min(k, order.size());
        u64 total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            total += rows[order[i]].isSolutionSeed ? 1ULL : 0ULL;
        }
        return total;
    }

    static double weightedRankScore(const std::vector<SeedEval>& rows,
                                    const std::vector<std::size_t>& order) {
        if (order.empty()) {
            return 0.0;
        }

        const double n = static_cast<double>(order.size());
        double weighted = 0.0;
        double maxWeighted = 0.0;

        for (std::size_t i = 0; i < order.size(); ++i) {
            const double w = rows[order[i]].isSolutionSeed ? 1.0 : 0.0;
            weighted += w * (n - static_cast<double>(i));
            maxWeighted += w * n;
        }

        if (maxWeighted <= 0.0) {
            return 0.0;
        }
        return weighted / maxWeighted;
    }

    static std::vector<std::size_t> collectPositiveRanks(const std::vector<SeedEval>& rows,
                                                         const std::vector<std::size_t>& order) {
        std::vector<std::size_t> ranks;
        ranks.reserve(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (rows[order[i]].isSolutionSeed) {
                ranks.push_back(i + 1);
            }
        }
        return ranks;
    }

    static std::vector<std::size_t> buildBinCounts(const std::vector<std::size_t>& ranks,
                                                   const std::size_t listSize,
                                                   const std::size_t width = 100) {
        std::vector<std::size_t> counts(width, 0);
        if (width == 0 || listSize == 0) {
            return counts;
        }
        for (const std::size_t rank1 : ranks) {
            const std::size_t zero = rank1 - 1;
            std::size_t bin = (zero * width) / listSize;
            if (bin >= width) {
                bin = width - 1;
            }
            ++counts[bin];
        }
        return counts;
    }

    static std::string buildAsciiHitLine(const std::vector<std::size_t>& ranks,
                                         const std::size_t listSize,
                                         const std::size_t width = 100) {
        if (width == 0) {
            return "";
        }
        const std::string emptyCell = "░";
        const std::string filledCell = "█";
        const std::vector<std::size_t> counts = buildBinCounts(ranks, listSize, width);
        std::vector<std::string> cells(width, emptyCell);
        for (std::size_t i = 0; i < width; ++i) {
            if (counts[i] > 0) {
                cells[i] = filledCell;
            }
        }

        std::string line;
        line.reserve(width * emptyCell.size());
        for (const auto& c : cells) {
            line += c;
        }
        return line;
    }

    static std::string buildColoredCompressionLine(const std::vector<std::size_t>& ranks,
                                                   const std::size_t listSize,
                                                   const std::size_t width = 100) {
        if (width == 0) {
            return "";
        }

        const std::vector<std::size_t> counts = buildBinCounts(ranks, listSize, width);
        std::size_t maxCount = 0;
        for (const std::size_t c : counts) {
            if (c > maxCount) {
                maxCount = c;
            }
        }

        static const int kHeat[7] = {46, 82, 118, 190, 214, 202, 196};
        std::string out;
        out.reserve(width * 12);
        for (const std::size_t c : counts) {
            if (c == 0) {
                out += "\x1b[38;5;238m░\x1b[0m";
                continue;
            }

            std::size_t idx = 0;
            if (maxCount > 1) {
                idx = ((c - 1) * 6) / (maxCount - 1);
                if (idx > 6) {
                    idx = 6;
                }
            }
            out += "\x1b[38;5;";
            out += std::to_string(kHeat[idx]);
            out += "m█\x1b[0m";
        }
        return out;
    }

    static std::string ranksToString(const std::vector<std::size_t>& ranks) {
        if (ranks.empty()) {
            return "-";
        }
        std::ostringstream out;
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << ranks[i];
        }
        return out.str();
    }

    static double meanRank(const std::vector<std::size_t>& ranks) {
        if (ranks.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (const std::size_t r : ranks) {
            total += static_cast<double>(r);
        }
        return total / static_cast<double>(ranks.size());
    }

    static double meanReciprocalRank(const std::vector<std::size_t>& ranks) {
        if (ranks.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (const std::size_t r : ranks) {
            total += 1.0 / static_cast<double>(r);
        }
        return total / static_cast<double>(ranks.size());
    }

    static std::size_t adjacencyGapPenalty(const std::vector<std::size_t>& ranks,
                                           const std::size_t listSize) {
        if (ranks.empty()) {
            return listSize + 1;
        }

        std::size_t penalty = 0;
        std::size_t prev = 0;
        for (const std::size_t rank : ranks) {
            if (rank > prev + 1) {
                penalty += (rank - prev - 1);
            }
            prev = rank;
        }
        return penalty;
    }

    static void markSolutionSeedsFromFile(const Board& start,
                                          const std::vector<std::string>& solutionLines,
                                          std::vector<SeedEval>& rows) {
        std::unordered_map<u64, std::vector<std::size_t>> seedBuckets;
        seedBuckets.reserve(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            seedBuckets[rows[i].seedHash].push_back(i);
        }

        for (const std::string& line : solutionLines) {
            std::vector<u8> moves;
            try {
                moves = Memory::parseNormMoveString(line);
            } catch (...) {
                continue;
            }
            if (moves.size() < static_cast<std::size_t>(kSeedDepth)) {
                continue;
            }

            Board cur = start;
            for (int i = 0; i < kSeedDepth; ++i) {
                allActStructList[moves[static_cast<std::size_t>(i)]].action(cur);
            }

            const B1B2 b = cur.asB1B2();
            const u64 h = StateHash::computeHash(b);
            const auto it = seedBuckets.find(h);
            if (it == seedBuckets.end()) {
                continue;
            }

            for (const std::size_t idx : it->second) {
                if (rows[idx].seed == b) {
                    rows[idx].isSolutionSeed = true;
                }
            }
        }
    }

    static bool parsePuzzleNameFromLevelsFinalFile(const fs::path& filePath, std::string& puzzleNameOut) {
        static const std::regex pat(R"(^([0-9]+-[0-9]+)_c[0-9]+_[0-9]+\.txt$)");
        std::smatch m;
        const std::string name = filePath.filename().string();
        if (!std::regex_match(name, m, pat) || m.size() < 2) {
            return false;
        }
        puzzleNameOut = m[1].str();
        return true;
    }

    static bool parsePuzzleLengthFromPuzzleName(const std::string& puzzleName, int& lengthOut) {
        const std::size_t dash = puzzleName.find('-');
        if (dash == std::string::npos || dash == 0) {
            return false;
        }
        try {
            lengthOut = std::stoi(puzzleName.substr(0, dash));
        } catch (...) {
            return false;
        }
        return lengthOut > 0;
    }

    static std::vector<BenchmarkTarget> filterTrainingTargetsByMinPuzzleLength(
            const std::vector<BenchmarkTarget>& allTargets,
            const int minPuzzleLength,
            std::size_t& skippedOut) {
        skippedOut = 0;
        std::vector<BenchmarkTarget> filtered;
        filtered.reserve(allTargets.size());

        for (const auto& t : allTargets) {
            int puzzleLength = 0;
            if (!parsePuzzleLengthFromPuzzleName(t.puzzleName, puzzleLength)) {
                ++skippedOut;
                continue;
            }
            if (puzzleLength < minPuzzleLength) {
                ++skippedOut;
                continue;
            }
            filtered.push_back(t);
        }
        return filtered;
    }

    static std::vector<BenchmarkTarget> collectAllNonFatTargetsFromLevelsFinal() {
        std::vector<BenchmarkTarget> out;
        if (!fs::exists(kLevelsFinalDir) || !fs::is_directory(kLevelsFinalDir)) {
            return out;
        }

        for (const auto& entry : fs::directory_iterator(kLevelsFinalDir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") {
                continue;
            }

            std::string puzzleName;
            if (!parsePuzzleNameFromLevelsFinalFile(entry.path(), puzzleName)) {
                continue;
            }

            const BoardPair* pair = BoardLookup::getBoardPair(puzzleName.c_str());
            if (pair == nullptr) {
                continue;
            }
            if (pair->getStartState().getFatBool() || pair->getEndState().getFatBool()) {
                continue;
            }

            out.push_back({puzzleName, entry.path()});
        }

        std::sort(out.begin(), out.end(), [](const BenchmarkTarget& a, const BenchmarkTarget& b) {
            if (a.puzzleName != b.puzzleName) {
                return a.puzzleName < b.puzzleName;
            }
            return a.solutionsFile.string() < b.solutionsFile.string();
        });
        return out;
    }

    static bool collectRowsForTarget(const std::string& puzzleName,
                                     const fs::path& solutionsFile,
                                     std::vector<SeedEval>& rowsOut) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzleName.c_str());
        if (pair == nullptr) {
            return false;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        if (start.getFatBool() || goal.getFatBool()) {
            return false;
        }

        StateHash::refreshHashFunc(start);

        JVec<B1B2> leftSeeds;
        JVec<u64> leftSeedHashes;
        buildUniqueNoneDepthFrontierB1B2<kSeedDepth>(start, leftSeeds, leftSeedHashes, true);

        const std::vector<std::string> solutionLines = readNonEmptyLines(solutionsFile);
        if (solutionLines.empty()) {
            return false;
        }

        rowsOut.assign(leftSeeds.size(), {});
        for (std::size_t i = 0; i < leftSeeds.size(); ++i) {
            rowsOut[i].seed = leftSeeds[i];
            rowsOut[i].seedHash = leftSeedHashes[i];
        }
        markSolutionSeedsFromFile(start, solutionLines, rowsOut);

        const std::size_t seedCount = leftSeeds.size();
        const unsigned int hw = std::thread::hardware_concurrency();
        const std::size_t workerCount = std::max<std::size_t>(1, (hw == 0) ? 8 : hw);
        std::atomic<std::size_t> nextIndex{0};

        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (std::size_t w = 0; w < workerCount; ++w) {
            workers.emplace_back([&]() {
                while (true) {
                    const std::size_t i = nextIndex.fetch_add(1, std::memory_order_relaxed);
                    if (i >= seedCount) {
                        break;
                    }

                    Board seedBoard = makeBoardFromState(leftSeeds[i]);
                    for (int depth = 0; depth <= kMaxLookaheadDepth; ++depth) {
                        rowsOut[i].lookaheadScores[static_cast<std::size_t>(depth)] =
                                heur::evaluateLookaheadMinScoresAtDepth(seedBoard, goal, depth);
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        return true;
    }

    static bool trainLearnedBlendModel(const std::vector<BenchmarkTarget>& targets,
                                       LearnedBlendModel& outModel) {
        struct Sample {
            std::array<double, kLearnedFeatureCount> x{};
            double y = 0.0;
        };
        std::vector<Sample> samples;
        samples.reserve(500000);

        for (const auto& target : targets) {
            std::vector<SeedEval> rows;
            if (!collectRowsForTarget(target.puzzleName, target.solutionsFile, rows) || rows.empty()) {
                continue;
            }

            std::array<std::array<int, kHeuristicCount>, kLookaheadDepthCount> mins{}, maxs{};
            computeDepthMinMax(rows, mins, maxs);

            for (const auto& row : rows) {
                Sample s;
                s.x = makeCombinedFeatures(row, mins, maxs);
                s.y = row.isSolutionSeed ? 1.0 : 0.0;
                samples.push_back(s);
            }
        }

        if (samples.empty()) {
            return false;
        }

        double posCount = 0.0;
        for (const auto& s : samples) {
            posCount += s.y;
        }
        const double negCount = static_cast<double>(samples.size()) - posCount;
        if (posCount <= 0.0 || negCount <= 0.0) {
            return false;
        }
        const double posWeight = negCount / posCount;

        outModel.weights.fill(0.0);
        outModel.bias = 0.0;

        constexpr int kIters = 900;
        constexpr double kL2 = 1e-4;
        double lr = 0.15;

        for (int iter = 0; iter < kIters; ++iter) {
            std::array<double, kLearnedFeatureCount> gradW{};
            double gradB = 0.0;

            for (const auto& s : samples) {
                double z = outModel.bias;
                for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
                    z += outModel.weights[i] * s.x[i];
                }
                const double p = sigmoidStable(z);
                const double sampleW = (s.y > 0.5) ? posWeight : 1.0;
                const double err = (p - s.y) * sampleW;

                for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
                    gradW[i] += err * s.x[i];
                }
                gradB += err;
            }

            const double invN = 1.0 / static_cast<double>(samples.size());
            for (std::size_t i = 0; i < kLearnedFeatureCount; ++i) {
                gradW[i] = gradW[i] * invN + kL2 * outModel.weights[i];
                outModel.weights[i] -= lr * gradW[i];
            }
            gradB *= invN;
            outModel.bias -= lr * gradB;

            if ((iter + 1) % 150 == 0) {
                lr *= 0.7;
            }
        }

        outModel.valid = true;
        return true;
    }

} // namespace

static int runBenchmarkForTarget(const std::string& puzzleName, const fs::path& solutionsFile) {
    const BoardPair* pair = BoardLookup::getBoardPair(puzzleName.c_str());
    if (pair == nullptr) {
        std::cerr << "[ERROR] Puzzle not found: " << puzzleName << "\n";
        return 1;
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();

    if (start.getFatBool() || goal.getFatBool()) {
        std::cerr << "[ERROR] This benchmark currently expects a non-fat puzzle.\n";
        return 1;
    }

    StateHash::refreshHashFunc(start);

    JVec<B1B2> leftSeeds;
    JVec<u64> leftSeedHashes;
    buildUniqueNoneDepthFrontierB1B2<kSeedDepth>(start, leftSeeds, leftSeedHashes, true);
    tcout << "seed(" << kSeedDepth << ") unique size: " << leftSeeds.size() << '\n';

    const std::vector<std::string> solutionLines = readNonEmptyLines(solutionsFile);
    if (solutionLines.empty()) {
        std::cerr << "[ERROR] Failed to read solutions file or file is empty: " << solutionsFile.string() << "\n";
        return 1;
    }
    tcout << "solutions loaded: " << solutionLines.size() << " from " << solutionsFile.string() << '\n';

    std::vector<SeedEval> rows;
    rows.resize(leftSeeds.size());

    for (std::size_t i = 0; i < leftSeeds.size(); ++i) {
        rows[i].seed = leftSeeds[i];
        rows[i].seedHash = leftSeedHashes[i];
    }

    markSolutionSeedsFromFile(start, solutionLines, rows);

    {
        const std::size_t seedCount = leftSeeds.size();
        const unsigned int hw = std::thread::hardware_concurrency();
        const std::size_t workerCount = std::max<std::size_t>(1, (hw == 0) ? 8 : hw);
        std::atomic<std::size_t> nextIndex{0};
        std::atomic<std::size_t> processed{0};
        std::mutex progressMutex;

        std::vector<std::thread> workers;
        workers.reserve(workerCount);

        for (std::size_t w = 0; w < workerCount; ++w) {
            workers.emplace_back([&]() {
                while (true) {
                    const std::size_t i = nextIndex.fetch_add(1, std::memory_order_relaxed);
                    if (i >= seedCount) {
                        break;
                    }

                    Board seedBoard = makeBoardFromState(leftSeeds[i]);
                    for (int depth = 0; depth <= kMaxLookaheadDepth; ++depth) {
                        rows[i].lookaheadScores[static_cast<std::size_t>(depth)] =
                                heur::evaluateLookaheadMinScoresAtDepth(seedBoard, goal, depth);
                    }

                    const std::size_t done = processed.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 100 == 0 || done == seedCount) {
                        std::lock_guard<std::mutex> lock(progressMutex);
                        tcout << "processed seeds: " << done << "/" << seedCount << '\n';
                    }
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }
    }

    const std::vector<std::size_t> oracleOrder = buildOracleOrder(rows);
    const std::size_t n = rows.size();

    u64 totalPositives = 0;
    for (const auto& r : rows) {
        totalPositives += r.isSolutionSeed ? 1ULL : 0ULL;
    }

    const std::size_t topK1 = std::min<std::size_t>(16, n);
    const std::size_t topK3 = std::min<std::size_t>(64, n);
    const std::size_t recallK1 = std::min<std::size_t>(100, n);
    const std::size_t recallK2 = std::min<std::size_t>(250, n);
    const std::size_t recallK3 = std::min<std::size_t>(500, n);
    const std::size_t placementWidth = std::max<std::size_t>(1, std::min<std::size_t>(kPlacementLineWidth, n));

    const u64 oracleTopK1 = topKPositiveCount(rows, oracleOrder, topK1);
    const u64 oracleTopK3 = topKPositiveCount(rows, oracleOrder, topK3);
    const double oracleWeighted = weightedRankScore(rows, oracleOrder);

    std::cout << "\n=== testHeuristicsOnLeftSeed (single known-solution file) ===\n";
    std::cout << "puzzle: " << puzzleName << "\n";
    std::cout << "solutions file: " << solutionsFile.string() << "\n";
    std::cout << "seed count: " << n << "\n";
    std::cout << "positive seeds (appear at depth " << kSeedDepth << " in listed solutions): " << totalPositives << "\n";
    std::cout << "topK baseline K=" << topK1 << "," << topK3 << "\n";
    std::cout << "recall K=" << recallK1 << "," << recallK2 << "," << recallK3 << "\n";
    std::cout << "lookahead depths evaluated: 0.." << kMaxLookaheadDepth << "\n\n";

    struct HeuristicReport {
        int heuristicIndex = 0;
        std::string label;
        std::vector<std::size_t> ranks;
        double r1 = 0.0;
        double r3 = 0.0;
        double wrNorm = 0.0;
        double recall1 = 0.0;
        double recall2 = 0.0;
        double recall3 = 0.0;
        std::size_t adjGap = 0;
        double meanR = 0.0;
        double meanRPct = 0.0;
        double mrr = 0.0;
        std::string firstRank;
        std::string medianRank;
        std::string lastRank;
    };

    std::vector<std::vector<HeuristicReport>> depthReports(static_cast<std::size_t>(kLookaheadDepthCount));
    const std::vector<std::size_t> oracleRanks = collectPositiveRanks(rows, oracleOrder);

    for (int depth = 0; depth <= kMaxLookaheadDepth; ++depth) {
        auto& reports = depthReports[static_cast<std::size_t>(depth)];
        reports.reserve(kHeuristicCount + 2);

        for (int h = 0; h < static_cast<int>(kHeuristicCount); ++h) {
            const auto order = buildOrderByHeuristic(rows, h, depth);

            const u64 gotTopK1 = topKPositiveCount(rows, order, topK1);
            const u64 gotTopK3 = topKPositiveCount(rows, order, topK3);
            const double r1 = (oracleTopK1 == 0) ? 0.0 : static_cast<double>(gotTopK1) / static_cast<double>(oracleTopK1);
            const double r3 = (oracleTopK3 == 0) ? 0.0 : static_cast<double>(gotTopK3) / static_cast<double>(oracleTopK3);

            const double wr = weightedRankScore(rows, order);
            const double wrNorm = (oracleWeighted <= 0.0) ? 0.0 : (wr / oracleWeighted);
            const std::vector<std::size_t> ranks = collectPositiveRanks(rows, order);
            const u64 gotRecallK1 = topKPositiveCount(rows, order, recallK1);
            const u64 gotRecallK2 = topKPositiveCount(rows, order, recallK2);
            const u64 gotRecallK3 = topKPositiveCount(rows, order, recallK3);

            const double recall1 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK1) / static_cast<double>(totalPositives);
            const double recall2 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK2) / static_cast<double>(totalPositives);
            const double recall3 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK3) / static_cast<double>(totalPositives);
            const double meanR = meanRank(ranks);
            const double meanRPct = (n == 0) ? 0.0 : (meanR / static_cast<double>(n));
            const double mrr = meanReciprocalRank(ranks);
            const std::size_t adjGap = adjacencyGapPenalty(ranks, n);

            reports.push_back({
                    h,
                    kHeuristicNames[static_cast<std::size_t>(h)],
                    ranks,
                    r1,
                    r3,
                    wrNorm,
                    recall1,
                    recall2,
                    recall3,
                    adjGap,
                    meanR,
                    meanRPct,
                    mrr,
                    ranks.empty() ? "-" : std::to_string(ranks.front()),
                    ranks.empty() ? "-" : std::to_string(ranks[ranks.size() / 2]),
                    ranks.empty() ? "-" : std::to_string(ranks.back())
            });
        }

        {
            const auto order = buildOrderByBlendedHeuristic(rows, depth);

            const u64 gotTopK1 = topKPositiveCount(rows, order, topK1);
            const u64 gotTopK3 = topKPositiveCount(rows, order, topK3);
            const double r1 = (oracleTopK1 == 0) ? 0.0 : static_cast<double>(gotTopK1) / static_cast<double>(oracleTopK1);
            const double r3 = (oracleTopK3 == 0) ? 0.0 : static_cast<double>(gotTopK3) / static_cast<double>(oracleTopK3);

            const double wr = weightedRankScore(rows, order);
            const double wrNorm = (oracleWeighted <= 0.0) ? 0.0 : (wr / oracleWeighted);
            const std::vector<std::size_t> ranks = collectPositiveRanks(rows, order);
            const u64 gotRecallK1 = topKPositiveCount(rows, order, recallK1);
            const u64 gotRecallK2 = topKPositiveCount(rows, order, recallK2);
            const u64 gotRecallK3 = topKPositiveCount(rows, order, recallK3);

            const double recall1 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK1) / static_cast<double>(totalPositives);
            const double recall2 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK2) / static_cast<double>(totalPositives);
            const double recall3 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK3) / static_cast<double>(totalPositives);
            const double meanR = meanRank(ranks);
            const double meanRPct = (n == 0) ? 0.0 : (meanR / static_cast<double>(n));
            const double mrr = meanReciprocalRank(ranks);
            const std::size_t adjGap = adjacencyGapPenalty(ranks, n);

            reports.push_back({
                    static_cast<int>(kHeuristicCount),
                    kBlendedHeuristicName,
                    ranks,
                    r1,
                    r3,
                    wrNorm,
                    recall1,
                    recall2,
                    recall3,
                    adjGap,
                    meanR,
                    meanRPct,
                    mrr,
                    ranks.empty() ? "-" : std::to_string(ranks.front()),
                    ranks.empty() ? "-" : std::to_string(ranks[ranks.size() / 2]),
                    ranks.empty() ? "-" : std::to_string(ranks.back())
            });
        }

        if (gLearnedModel.valid) {
            const auto order = buildOrderByLearnedModel(rows, gLearnedModel);

            const u64 gotTopK1 = topKPositiveCount(rows, order, topK1);
            const u64 gotTopK3 = topKPositiveCount(rows, order, topK3);
            const double r1 = (oracleTopK1 == 0) ? 0.0 : static_cast<double>(gotTopK1) / static_cast<double>(oracleTopK1);
            const double r3 = (oracleTopK3 == 0) ? 0.0 : static_cast<double>(gotTopK3) / static_cast<double>(oracleTopK3);

            const double wr = weightedRankScore(rows, order);
            const double wrNorm = (oracleWeighted <= 0.0) ? 0.0 : (wr / oracleWeighted);
            const std::vector<std::size_t> ranks = collectPositiveRanks(rows, order);
            const u64 gotRecallK1 = topKPositiveCount(rows, order, recallK1);
            const u64 gotRecallK2 = topKPositiveCount(rows, order, recallK2);
            const u64 gotRecallK3 = topKPositiveCount(rows, order, recallK3);

            const double recall1 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK1) / static_cast<double>(totalPositives);
            const double recall2 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK2) / static_cast<double>(totalPositives);
            const double recall3 = (totalPositives == 0) ? 0.0 : static_cast<double>(gotRecallK3) / static_cast<double>(totalPositives);
            const double meanR = meanRank(ranks);
            const double meanRPct = (n == 0) ? 0.0 : (meanR / static_cast<double>(n));
            const double mrr = meanReciprocalRank(ranks);
            const std::size_t adjGap = adjacencyGapPenalty(ranks, n);

            reports.push_back({
                    static_cast<int>(kHeuristicCount + 1),
                    kLearnedHeuristicName,
                    ranks,
                    r1,
                    r3,
                    wrNorm,
                    recall1,
                    recall2,
                    recall3,
                    adjGap,
                    meanR,
                    meanRPct,
                    mrr,
                    ranks.empty() ? "-" : std::to_string(ranks.front()),
                    ranks.empty() ? "-" : std::to_string(ranks[ranks.size() / 2]),
                    ranks.empty() ? "-" : std::to_string(ranks.back())
            });
        }

        std::sort(reports.begin(), reports.end(), [](const HeuristicReport& a, const HeuristicReport& b) {
            if (a.adjGap != b.adjGap) {
                return a.adjGap < b.adjGap;
            }
            if (a.recall2 != b.recall2) {
                return a.recall2 > b.recall2;
            }
            if (a.meanRPct != b.meanRPct) {
                return a.meanRPct < b.meanRPct;
            }
            if (a.mrr != b.mrr) {
                return a.mrr > b.mrr;
            }
            return a.heuristicIndex < b.heuristicIndex;
        });

        std::cout << "=== Depth " << depth << " report ===\n";
        std::cout << "sorted by: adjGap asc, then R@250 desc, then meanRk% asc, then MRR desc\n\n";

        std::cout << std::left
                  << std::setw(34) << "heuristic"
                  << std::right
                  << std::setw(9) << "top16"
                  << std::setw(9) << "top64"
                  << std::setw(9) << "wRank"
                  << std::setw(9) << "R@100"
                  << std::setw(9) << "R@250"
                  << std::setw(9) << "R@500"
                  << std::setw(9) << "adjGap"
                  << std::setw(9) << "meanRk"
                  << std::setw(9) << "meanRk%"
                  << std::setw(9) << "MRR"
                  << std::setw(9) << "first"
                  << std::setw(9) << "median"
                  << std::setw(9) << "last"
                  << "\n";

        for (const auto& rep : reports) {
            std::cout << std::left << std::setw(34) << rep.label
                      << std::right << std::setw(9) << std::fixed << std::setprecision(3) << rep.r1
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.r3
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.wrNorm
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.recall1
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.recall2
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.recall3
                      << std::setw(9) << rep.adjGap
                      << std::setw(9) << std::fixed << std::setprecision(1) << rep.meanR
                      << std::setw(9) << std::fixed << std::setprecision(3) << rep.meanRPct
                      << std::setw(9) << std::fixed << std::setprecision(4) << rep.mrr
                      << std::setw(9) << rep.firstRank
                      << std::setw(9) << rep.medianRank
                      << std::setw(9) << rep.lastRank
                      << "\n";
        }

        std::cout << "\n=== Depth " << depth << " Positive Rank Positions ===\n";
        std::cout << std::left << std::setw(34) << "oracle"
                  << std::right << ranksToString(oracleRanks) << "\n";
        for (const auto& rep : reports) {
            std::cout << std::left << std::setw(34) << rep.label
                      << std::right << ranksToString(rep.ranks) << "\n";
        }

        std::cout << "\n=== Depth " << depth << " Positive Placement Lines (color reflects compression per bin) ===\n";
        std::cout << "legend: cool=single hit, hot=multiple hits in same bin\n";
        std::cout << "placement width: " << placementWidth << " bins\n";
        std::cout << std::left << std::setw(34) << "oracle"
                  << std::right << buildColoredCompressionLine(oracleRanks, n, placementWidth) << "\n";
        for (const auto& rep : reports) {
            std::cout << std::left << std::setw(34) << rep.label
                      << std::right << buildColoredCompressionLine(rep.ranks, n, placementWidth) << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Depth Comparison vs depth 0 ===\n";
    const auto& baseReports = depthReports[0];
    for (int depth = 1; depth <= kMaxLookaheadDepth; ++depth) {
        const auto& reports = depthReports[static_cast<std::size_t>(depth)];

        int betterAdjGapCount = 0;
        int betterRecall250Count = 0;
        int betterMeanRankPctCount = 0;

        for (int h = 0; h < static_cast<int>(kHeuristicCount); ++h) {
            const auto baseIt = std::find_if(baseReports.begin(), baseReports.end(), [h](const HeuristicReport& r) {
                return r.heuristicIndex == h;
            });
            const auto curIt = std::find_if(reports.begin(), reports.end(), [h](const HeuristicReport& r) {
                return r.heuristicIndex == h;
            });
            if (baseIt == baseReports.end() || curIt == reports.end()) {
                continue;
            }

            if (curIt->adjGap < baseIt->adjGap) {
                ++betterAdjGapCount;
            }
            if (curIt->recall2 > baseIt->recall2) {
                ++betterRecall250Count;
            }
            if (curIt->meanRPct < baseIt->meanRPct) {
                ++betterMeanRankPctCount;
            }
        }

        const auto bestAdjGapBaseIt = std::min_element(baseReports.begin(), baseReports.end(),
                                                       [](const HeuristicReport& a, const HeuristicReport& b) {
                                                           return a.adjGap < b.adjGap;
                                                       });
        const auto bestAdjGapCurIt = std::min_element(reports.begin(), reports.end(),
                                                      [](const HeuristicReport& a, const HeuristicReport& b) {
                                                          return a.adjGap < b.adjGap;
                                                      });
        const auto bestR250BaseIt = std::max_element(baseReports.begin(), baseReports.end(),
                                                     [](const HeuristicReport& a, const HeuristicReport& b) {
                                                         return a.recall2 < b.recall2;
                                                     });
        const auto bestR250CurIt = std::max_element(reports.begin(), reports.end(),
                                                    [](const HeuristicReport& a, const HeuristicReport& b) {
                                                        return a.recall2 < b.recall2;
                                                    });
        const auto bestMeanPctBaseIt = std::min_element(baseReports.begin(), baseReports.end(),
                                                        [](const HeuristicReport& a, const HeuristicReport& b) {
                                                            return a.meanRPct < b.meanRPct;
                                                        });
        const auto bestMeanPctCurIt = std::min_element(reports.begin(), reports.end(),
                                                       [](const HeuristicReport& a, const HeuristicReport& b) {
                                                           return a.meanRPct < b.meanRPct;
                                                       });

        const long long bestAdjGapDelta = static_cast<long long>(bestAdjGapCurIt->adjGap)
                                          - static_cast<long long>(bestAdjGapBaseIt->adjGap);
        const double bestR250Delta = bestR250CurIt->recall2 - bestR250BaseIt->recall2;
        const double bestMeanPctDelta = bestMeanPctCurIt->meanRPct - bestMeanPctBaseIt->meanRPct;

        std::cout << "depth " << depth << ": best heuristic = "
                  << reports.front().label
                  << " | adjGap delta=" << bestAdjGapDelta
                  << " | R@250 delta=" << std::setprecision(3) << bestR250Delta
                  << " | meanRk% delta=" << std::setprecision(3) << bestMeanPctDelta;

        if (betterAdjGapCount > 0 || betterRecall250Count > 0 || betterMeanRankPctCount > 0) {
            std::cout << " | improvements: adjGap " << betterAdjGapCount << ", R@250 " << betterRecall250Count
                      << ", meanRk% " << betterMeanRankPctCount;
        }
        std::cout << "\n";

        const int scoreSignals = (bestAdjGapDelta < 0 ? 1 : 0)
                                 + (bestR250Delta > 0.0 ? 1 : 0)
                                 + (bestMeanPctDelta < 0.0 ? 1 : 0);
        if (scoreSignals >= 2 || betterAdjGapCount >= (static_cast<int>(kHeuristicCount) / 2 + 1)) {
            std::cout << "          conclusion: depth " << depth << " looks better than depth 0 for ranking quality.\n";
        } else if (scoreSignals == 0 && betterAdjGapCount == 0 && betterRecall250Count == 0 && betterMeanRankPctCount == 0) {
            std::cout << "          conclusion: depth " << depth << " does not improve over depth 0 on these signals.\n";
        } else {
            std::cout << "          conclusion: depth " << depth << " is mixed; some heuristics improve, some regress.\n";
        }
    }
    return 0;
}

int main() {
#ifdef _WIN32
    force_utf8_console();
#endif

    std::cout << "run config: scope=" << toString(kRunConfig.scope)
              << ", learned-model=" << toString(kRunConfig.modelMode) << "\n";

    std::vector<BenchmarkTarget> targets;
    const bool needsAllTargetsForBenchmark = (kRunConfig.scope == BenchmarkScope::AllNonFat);
    const bool needsAllTargetsForTraining =
            (kRunConfig.modelMode == LearnedModelMode::TrainAndBenchmark)
            || (kRunConfig.modelMode == LearnedModelMode::TrainOnly);

    if (needsAllTargetsForBenchmark || needsAllTargetsForTraining) {
        targets = collectAllNonFatTargetsFromLevelsFinal();
        if (targets.empty()) {
            std::cerr << "[ERROR] No non-fat targets found in " << kLevelsFinalDir.string() << "\n";
            return 1;
        }
    }

    switch (kRunConfig.modelMode) {
        case LearnedModelMode::Disabled:
            break;
        case LearnedModelMode::LoadFromDisk: {
            LearnedBlendModel loaded;
            if (loadLearnedModel(kLearnedModelFile, loaded)) {
                gLearnedModel = loaded;
                std::cout << "loaded learned model: " << kLearnedModelFile.string() << "\n";
            }
            break;
        }
        case LearnedModelMode::TrainAndBenchmark:
        case LearnedModelMode::TrainOnly: {
            std::size_t skippedTrainingTargets = 0;
            const std::vector<BenchmarkTarget> trainingTargets =
                    filterTrainingTargetsByMinPuzzleLength(targets, kMinTrainingPuzzleLength, skippedTrainingTargets);
            std::cout << "training learned blend model from " << trainingTargets.size()
                      << " non-fat puzzles (min length " << kMinTrainingPuzzleLength
                      << ", skipped " << skippedTrainingTargets << ")...\n";
            if (trainingTargets.empty()) {
                std::cerr << "[ERROR] No eligible training targets remain after min-length filter.\n";
                return 1;
            }

            LearnedBlendModel trained;
            if (!trainLearnedBlendModel(trainingTargets, trained)) {
                std::cerr << "[ERROR] Failed to train learned blend model.\n";
                return 1;
            }
            gLearnedModel = trained;
            if (!saveLearnedModel(gLearnedModel, kLearnedModelFile)) {
                std::cerr << "[WARN] Trained model could not be saved to " << kLearnedModelFile.string() << "\n";
            } else {
                std::cout << "saved learned model: " << kLearnedModelFile.string() << "\n";
            }

            if (kRunConfig.modelMode == LearnedModelMode::TrainOnly) {
                std::cout << "train-only mode done.\n";
                return 0;
            }
            break;
        }
    }

    if (kRunConfig.scope == BenchmarkScope::SinglePuzzle) {
        return runBenchmarkForTarget(kPuzzleName, kSolutionsFile);
    }

    std::size_t failed = 0;
    std::cout << "batch mode: running " << targets.size() << " non-fat targets from "
              << kLevelsFinalDir.string() << "\n\n";
    for (std::size_t i = 0; i < targets.size(); ++i) {
        std::cout << "------------------------------------------------------------\n";
        std::cout << "target " << (i + 1) << "/" << targets.size() << ": "
                  << targets[i].puzzleName << " | " << targets[i].solutionsFile.string() << "\n";
        const int rc = runBenchmarkForTarget(targets[i].puzzleName, targets[i].solutionsFile);
        if (rc != 0) {
            ++failed;
        }
        std::cout << "\n";
    }

    std::cout << "batch finished: " << (targets.size() - failed) << " succeeded, "
              << failed << " failed\n";
    int x;
    std::cin >> x;
    return (failed == 0) ? 0 : 1;
}