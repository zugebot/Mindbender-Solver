#include "code/include.hpp"
#include "code/solver/memory_perm_gen.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

    enum class RunMode {
        SequentialSeedsSearch,
        RandomTrialsSearch,
        SingleSeedVerbose,
    };

    constexpr RunMode RUN_MODE = RunMode::SingleSeedVerbose;

    constexpr u32 TARGET_DEPTH = 7;
    constexpr u32 LEFT_REJECT_DEPTH = 3;
    constexpr u32 RIGHT_REJECT_DEPTH = 3;
    constexpr u32 LEFT_FINAL_DEPTH = 3;
    constexpr u32 RIGHT_FINAL_DEPTH = 4;

    constexpr u32 SMALL_FULL_NONE_LIMIT = 40;

    constexpr u32 TRIAL_COUNT = 1000;
    constexpr u32 MAX_PUZZLE_GENERATION_ATTEMPTS = 200000;
    constexpr u64 RNG_SEED = 0x41bULL;

    constexpr bool PRINT_ACCEPTED_BOARDS_IN_SEARCH = false;
    constexpr bool STOP_ON_ANY_MISMATCH = false;
    constexpr bool STOP_ON_SMALL_MISMATCH = true;

    struct PuzzleInstance {
        Board start;
        Board goal;
        std::vector<u8> witnessScramble;
        u64 attemptSeed = 0;
    };

    struct CanonicalExpansionInfo {
        std::string rawCanonical;
        std::vector<std::string> expanded;
    };

    struct FrontierCache {
        const Board& start;
        const Board& goal;

        std::array<JVec<Memory>, 6> startNone;
        std::array<JVec<Memory>, 6> goalNone;
        std::array<JVec<Memory>, 6> startAsc;
        std::array<JVec<Memory>, 6> goalDesc;

        std::array<bool, 6> builtStartNone{};
        std::array<bool, 6> builtGoalNone{};
        std::array<bool, 6> builtStartAsc{};
        std::array<bool, 6> builtGoalDesc{};

        FrontierCache(const Board& s, const Board& g) : start(s), goal(g) {}

        template<eSequenceDir DIR>
        const JVec<Memory>& getStart(u32 depth) {
            if constexpr (DIR == eSequenceDir::NONE) {
                if (!builtStartNone[depth]) {
                    Timer timer;
                    Perms<Memory>::getDepthFunc<eSequenceDir::NONE>(start, startNone[depth], depth, true);
                    std::sort(startNone[depth].begin(), startNone[depth].end());
                    builtStartNone[depth] = true;
                    tcout << "      built START NONE depth " << depth
                          << " | size=" << startNone[depth].size()
                          << " | time=" << timer.getSeconds() << "s\n";
                }
                return startNone[depth];
            } else if constexpr (DIR == eSequenceDir::ASCENDING) {
                if (!builtStartAsc[depth]) {
                    Timer timer;
                    Perms<Memory>::getDepthFunc<eSequenceDir::ASCENDING>(start, startAsc[depth], depth, true);
                    std::sort(startAsc[depth].begin(), startAsc[depth].end());
                    builtStartAsc[depth] = true;
                    tcout << "      built START ASC depth " << depth
                          << " | size=" << startAsc[depth].size()
                          << " | time=" << timer.getSeconds() << "s\n";
                }
                return startAsc[depth];
            } else {
                static_assert(DIR != DIR, "Unsupported start frontier direction");
            }
        }

        template<eSequenceDir DIR>
        const JVec<Memory>& getGoal(u32 depth) {
            if constexpr (DIR == eSequenceDir::NONE) {
                if (!builtGoalNone[depth]) {
                    Timer timer;
                    Perms<Memory>::getDepthFunc<eSequenceDir::NONE>(goal, goalNone[depth], depth, true);
                    std::sort(goalNone[depth].begin(), goalNone[depth].end());
                    builtGoalNone[depth] = true;
                    tcout << "      built GOAL NONE depth " << depth
                              << " | size=" << goalNone[depth].size()
                              << " | time=" << timer.getSeconds() << "s\n";
                }
                return goalNone[depth];
            } else if constexpr (DIR == eSequenceDir::DESCENDING) {
                if (!builtGoalDesc[depth]) {
                    Timer timer;
                    Perms<Memory>::getDepthFunc<eSequenceDir::DESCENDING>(goal, goalDesc[depth], depth, true);
                    std::sort(goalDesc[depth].begin(), goalDesc[depth].end());
                    builtGoalDesc[depth] = true;
                    tcout << "      built GOAL DESC depth " << depth
                              << " | size=" << goalDesc[depth].size()
                              << " | time=" << timer.getSeconds() << "s\n";
                }
                return goalDesc[depth];
            } else {
                static_assert(DIR != DIR, "Unsupported goal frontier direction");
            }
        }
    };

    const std::vector<u8>& getAllNormalMoves() {
        static const std::vector<u8> moves = [] {
            std::vector<u8> out;
            out.reserve(60);

            for (u32 i = 0; i < NORMAL_ROW_MOVE_COUNT; ++i) {
                out.push_back(static_cast<u8>(i));
            }
            for (u32 i = 0; i < NORMAL_COL_MOVE_COUNT; ++i) {
                out.push_back(static_cast<u8>(NORMAL_ROW_MOVE_COUNT + NORMAL_MOVE_GAP_COUNT + i));
            }

            return out;
        }();
        return moves;
    }

    std::string boardValuesInitializer(const Board& board) {
        std::string out;
        out += "{\n";
        for (u32 y = 0; y < 6; ++y) {
            out += "    ";
            for (u32 x = 0; x < 6; ++x) {
                out += std::to_string(board.getColor(x, y));
                if (!(x == 5 && y == 5)) {
                    out += ", ";
                }
            }
            out += '\n';
        }
        out += "}";
        return out;
    }

    std::string moveVectorToString(const std::vector<u8>& moves) {
        Memory memory;
        for (u8 move : moves) {
            memory.setNextNMove<1>(move);
        }
        return memory.asmStringForwards();
    }

    Board makeRandomTwoColorBoard(std::mt19937_64& rng) {
        std::array<u8, 36> values{};
        for (u32 i = 0; i < 18; ++i) {
            values[i] = 0;
        }
        for (u32 i = 18; i < 36; ++i) {
            values[i] = 1;
        }

        std::shuffle(values.begin(), values.end(), rng);
        return Board(values.data());
    }

    Board applyRandomEffectiveMoves(
            const Board& start,
            std::mt19937_64& rng,
            u32 moveCount,
            std::vector<u8>& outMoves) {
        const std::vector<u8>& allMoves = getAllNormalMoves();
        std::uniform_int_distribution<u32> dist(0, static_cast<u32>(allMoves.size() - 1));

        Board cur = start;
        outMoves.clear();
        outMoves.reserve(moveCount);

        for (u32 step = 0; step < moveCount; ++step) {
            while (true) {
                const u8 move = allMoves[dist(rng)];

                Board next = cur;
                allActStructList[move].action(next);

                if (next == cur) {
                    continue;
                }

                outMoves.push_back(move);
                cur = next;
                break;
            }
        }

        return cur;
    }

    PuzzleInstance makePuzzleFromSeed(u64 seed, u32 targetDepth) {
        std::mt19937_64 attemptRng(seed);

        PuzzleInstance puzzle;
        puzzle.attemptSeed = seed;
        puzzle.start = makeRandomTwoColorBoard(attemptRng);
        puzzle.goal = applyRandomEffectiveMoves(puzzle.start, attemptRng, targetDepth, puzzle.witnessScramble);
        return puzzle;
    }

    template<eSequenceDir LEFT_DIR, eSequenceDir RIGHT_DIR>
    std::size_t countValidMeetMatches(
            FrontierCache& cache,
            u32 leftDepth,
            u32 rightDepth,
            const char* label) {
        tcout << "    [" << label << "] split (" << leftDepth << ", " << rightDepth << ")\n";

        const JVec<Memory>& left = cache.template getStart<LEFT_DIR>(leftDepth);
        const JVec<Memory>& right = cache.template getGoal<RIGHT_DIR>(rightDepth);

        Timer timerInter;
        const auto matches = intersection_all_pairs(left, right);
        const double interTime = timerInter.getSeconds();

        std::size_t valid = 0;
        for (const auto& [fst, snd] : matches) {
            const Board midLeft = makeBoardWithMoves(cache.start, *fst);
            const Board midRight = makeBoardWithMoves(cache.goal, *snd);
            if (midLeft == midRight) {
                ++valid;
            }
        }

        tcout << "      left=" << left.size()
                  << " right=" << right.size()
                  << " rawMatches=" << matches.size()
                  << " validMeets=" << valid
                  << " interTime=" << interTime << "s\n";

        return valid;
    }

    template<eSequenceDir LEFT_DIR, eSequenceDir RIGHT_DIR>
    bool hasAnySolutionBySplit(
            FrontierCache& cache,
            u32 leftDepth,
            u32 rightDepth,
            const char* label) {
        return countValidMeetMatches<LEFT_DIR, RIGHT_DIR>(cache, leftDepth, rightDepth, label) != 0;
    }

    template<eSequenceDir LEFT_DIR, eSequenceDir RIGHT_DIR>
    std::set<std::string> collectSolutionSetBySplit(
            FrontierCache& cache,
            u32 leftDepth,
            u32 rightDepth,
            bool expandCanonical,
            const char* label) {
        std::set<std::string> out;

        tcout << "    [" << label << "] collecting split (" << leftDepth << ", " << rightDepth << ")\n";

        const JVec<Memory>& left = cache.template getStart<LEFT_DIR>(leftDepth);
        const JVec<Memory>& right = cache.template getGoal<RIGHT_DIR>(rightDepth);

        Timer timerInter;
        const auto matches = intersection_all_pairs(left, right);
        const double interTime = timerInter.getSeconds();

        tcout << "      left=" << left.size()
                  << " right=" << right.size()
                  << " rawMatches=" << matches.size()
                  << " interTime=" << interTime << "s\n";

        std::size_t validMeets = 0;

        for (const auto& [fst, snd] : matches) {
            const Board midLeft = makeBoardWithMoves(cache.start, *fst);
            const Board midRight = makeBoardWithMoves(cache.goal, *snd);

            if (!(midLeft == midRight)) {
                continue;
            }

            ++validMeets;

            const std::string raw = fst->asmString(snd);

            if (!expandCanonical) {
                out.insert(raw);
            } else {
                std::vector<u8> parsed = Memory::parseNormMoveString(raw);
                std::vector<std::string> expanded =
                        createMemoryPermutationStringsChecked(cache.start, cache.goal, parsed);
                for (const std::string& s : expanded) {
                    out.insert(s);
                }
            }
        }

        tcout << "      validMeets=" << validMeets
                  << " finalSetSize=" << out.size()
                  << '\n';

        return out;
    }

    std::vector<CanonicalExpansionInfo> collectCanonicalExpansionInfos(
            FrontierCache& cache,
            u32 leftDepth,
            u32 rightDepth,
            std::set<std::string>& expandedUnionOut) {
        std::vector<CanonicalExpansionInfo> infos;
        std::set<std::string> seenRaw;

        tcout << "    [VERBOSE CANONICAL] collecting split (" << leftDepth << ", " << rightDepth << ")\n";

        const JVec<Memory>& left = cache.getStart<eSequenceDir::ASCENDING>(leftDepth);
        const JVec<Memory>& right = cache.getGoal<eSequenceDir::DESCENDING>(rightDepth);

        Timer timerInter;
        const auto matches = intersection_all_pairs(left, right);
        const double interTime = timerInter.getSeconds();

        tcout << "      left=" << left.size()
                  << " right=" << right.size()
                  << " rawMatches=" << matches.size()
                  << " interTime=" << interTime << "s\n";

        std::size_t validMeets = 0;

        for (const auto& [fst, snd] : matches) {
            const Board midLeft = makeBoardWithMoves(cache.start, *fst);
            const Board midRight = makeBoardWithMoves(cache.goal, *snd);

            if (!(midLeft == midRight)) {
                continue;
            }

            ++validMeets;

            const std::string raw = fst->asmString(snd);
            if (!seenRaw.insert(raw).second) {
                continue;
            }

            CanonicalExpansionInfo info;
            info.rawCanonical = raw;

            std::vector<u8> parsed = Memory::parseNormMoveString(info.rawCanonical);
            std::vector<std::string> expanded =
                    createMemoryPermutationStringsChecked(cache.start, cache.goal, parsed);

            std::sort(expanded.begin(), expanded.end());
            expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());

            for (const std::string& s : expanded) {
                info.expanded.push_back(s);
                expandedUnionOut.insert(s);
            }

            infos.push_back(std::move(info));
        }

        tcout << "      validMeets=" << validMeets
                  << " uniqueRawCanonical=" << infos.size()
                  << " expandedUnionSize=" << expandedUnionOut.size()
                  << '\n';

        return infos;
    }

    bool isAcceptablePuzzle(FrontierCache& cache) {
        if (hasAnySolutionBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                    cache, LEFT_REJECT_DEPTH, RIGHT_REJECT_DEPTH, "REJECT CHECK")) {
            tcout << "    rejected: reject split NONE/NONE intersection exists\n";
            return false;
        }

        if (!hasAnySolutionBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                    cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, "ACCEPT CHECK")) {
            tcout << "    rejected: final split NONE/NONE has no intersection\n";
            return false;
        }

        return true;
    }

    bool generateExactDepthPuzzleSequentialSeeds(
            PuzzleInstance& outPuzzle,
            u32 targetDepth,
            u64& nextSeed) {
        while (true) {
            const u64 attemptSeed = nextSeed++;
            PuzzleInstance puzzle = makePuzzleFromSeed(attemptSeed, targetDepth);
            FrontierCache cache(puzzle.start, puzzle.goal);

            tcout << "  attempt seed=0x" << std::hex << attemptSeed << std::dec
                      << " | scramble=" << moveVectorToString(puzzle.witnessScramble)
                      << '\n';

            if (!isAcceptablePuzzle(cache)) {
                continue;
            }

            if (PRINT_ACCEPTED_BOARDS_IN_SEARCH) {
                tcout << "    accepted\n";
                tcout << "    start:\n" << puzzle.start.toBlandString() << '\n';
                tcout << "    goal:\n" << puzzle.goal.toBlandString() << '\n';
            }

            outPuzzle = std::move(puzzle);
            return true;
        }
    }

    bool generateExactDepthPuzzleRandomAttempts(
            std::mt19937_64& rng,
            PuzzleInstance& outPuzzle,
            u32 targetDepth,
            u32 maxAttempts) {
        for (u32 attempt = 0; attempt < maxAttempts; ++attempt) {
            const u64 attemptSeed = rng();
            PuzzleInstance puzzle = makePuzzleFromSeed(attemptSeed, targetDepth);
            FrontierCache cache(puzzle.start, puzzle.goal);

            tcout << "  attempt " << (attempt + 1)
                      << " | seed=0x" << std::hex << attemptSeed << std::dec
                      << " | scramble=" << moveVectorToString(puzzle.witnessScramble)
                      << '\n';

            if (!isAcceptablePuzzle(cache)) {
                continue;
            }

            if (PRINT_ACCEPTED_BOARDS_IN_SEARCH) {
                tcout << "    accepted\n";
                tcout << "    start:\n" << puzzle.start.toBlandString() << '\n';
                tcout << "    goal:\n" << puzzle.goal.toBlandString() << '\n';
            }

            outPuzzle = std::move(puzzle);
            return true;
        }

        return false;
    }

    bool generateExactDepthPuzzle(
            std::mt19937_64& rng,
            PuzzleInstance& outPuzzle,
            u32 targetDepth,
            u32 maxAttempts,
            u64& nextSeed) {
        switch (RUN_MODE) {
            case RunMode::SequentialSeedsSearch:
                return generateExactDepthPuzzleSequentialSeeds(outPuzzle, targetDepth, nextSeed);

            case RunMode::RandomTrialsSearch:
                return generateExactDepthPuzzleRandomAttempts(rng, outPuzzle, targetDepth, maxAttempts);

            case RunMode::SingleSeedVerbose:
                outPuzzle = makePuzzleFromSeed(RNG_SEED, targetDepth);
                return true;
        }

        return false;
    }

    void printSet(const std::set<std::string>& values, const std::string& title) {
        tcout << "\n=== " << title << " ===\n";
        for (const std::string& s : values) {
            tcout << s << '\n';
        }
    }

    void printSetDifference(
            const std::set<std::string>& lhs,
            const std::set<std::string>& rhs,
            const std::string& title) {
        tcout << "\n=== " << title << " ===\n";
        for (const std::string& s : lhs) {
            if (rhs.find(s) == rhs.end()) {
                tcout << s << '\n';
            }
        }
    }

    void printCanonicalInfos(const std::vector<CanonicalExpansionInfo>& infos) {
        tcout << "\n=== RAW CANONICAL PARENTS AND THEIR EXPANSIONS ===\n";
        for (u32 i = 0; i < static_cast<u32>(infos.size()); ++i) {
            tcout << "\n[" << (i + 1) << "] raw canonical: " << infos[i].rawCanonical << '\n';
            for (const std::string& s : infos[i].expanded) {
                tcout << "    -> " << s << '\n';
            }
        }
    }

    bool shouldStopOnMismatch(
            const std::set<std::string>& fullNone,
            const std::set<std::string>& expandedCanonical) {
        if (fullNone == expandedCanonical) {
            return false;
        }

        if (STOP_ON_ANY_MISMATCH) {
            return true;
        }

        if (STOP_ON_SMALL_MISMATCH && fullNone.size() <= SMALL_FULL_NONE_LIMIT) {
            return true;
        }

        return false;
    }

    void printMismatchReport(
            const PuzzleInstance& puzzle,
            const std::set<std::string>& fullNone,
            const std::set<std::string>& expandedCanonical,
            const Timer& totalTimer,
            bool smallMismatchLabel) {
        tcout << "\n" << (smallMismatchLabel ? "SMALL MISMATCH FOUND" : "MISMATCH FOUND") << '\n';
        tcout << "Case seed: 0x" << std::hex << puzzle.attemptSeed << std::dec << '\n';
        tcout << "Witness scramble: " << moveVectorToString(puzzle.witnessScramble) << "\n\n";

        tcout << "Start board:\n" << puzzle.start.toBlandString() << '\n';
        tcout << "Goal board:\n" << puzzle.goal.toBlandString() << '\n';

        tcout << "Start board initializer:\n"
                  << boardValuesInitializer(puzzle.start) << "\n\n";
        tcout << "Goal board initializer:\n"
                  << boardValuesInitializer(puzzle.goal) << "\n\n";

        tcout << "Full NONE count: " << fullNone.size() << '\n';
        tcout << "Expanded canonical count: " << expandedCanonical.size() << '\n';

        printSetDifference(fullNone, expandedCanonical, "In full NONE but missing from expanded canonical");
        printSetDifference(expandedCanonical, fullNone, "In expanded canonical but not in full NONE");

        tcout << "\nTotal time: " << totalTimer.getSeconds() << "s\n";
    }

    int runSingleSeedVerbose(const Timer& totalTimer) {
        tcout << "\n============================================================\n";
        tcout << "SINGLE SEED VERBOSE ANALYSIS\n";
        tcout << "============================================================\n";

        PuzzleInstance puzzle = makePuzzleFromSeed(RNG_SEED, TARGET_DEPTH);

        tcout << "Seed: 0x" << std::hex << puzzle.attemptSeed << std::dec << '\n';
        tcout << "Witness scramble: " << moveVectorToString(puzzle.witnessScramble) << "\n\n";

        tcout << "Start board:\n" << puzzle.start.toBlandString() << '\n';
        tcout << "Goal board:\n" << puzzle.goal.toBlandString() << '\n';

        tcout << "Start board initializer:\n"
                  << boardValuesInitializer(puzzle.start) << "\n\n";
        tcout << "Goal board initializer:\n"
                  << boardValuesInitializer(puzzle.goal) << "\n\n";

        FrontierCache cache(puzzle.start, puzzle.goal);

        const bool rejectHit = hasAnySolutionBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                cache, LEFT_REJECT_DEPTH, RIGHT_REJECT_DEPTH, "REJECT CHECK");
        const bool acceptHit = hasAnySolutionBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, "ACCEPT CHECK");

        tcout << "\nReject check result: " << (rejectHit ? "true" : "false") << '\n';
        tcout << "Accept check result: " << (acceptHit ? "true" : "false") << '\n';

        const std::set<std::string> fullNone =
                collectSolutionSetBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                        cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, false, "FULL NONE");

        std::set<std::string> expandedCanonical;
        const std::vector<CanonicalExpansionInfo> infos =
                collectCanonicalExpansionInfos(cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, expandedCanonical);

        tcout << "\nSummary:\n";
        tcout << "  fullNone=" << fullNone.size() << '\n';
        tcout << "  expandedCanonical=" << expandedCanonical.size() << '\n';
        tcout << "  rawCanonicalParents=" << infos.size() << '\n';

        printSet(fullNone, "FULL NONE SOLUTIONS");
        printCanonicalInfos(infos);
        printSet(expandedCanonical, "EXPANDED CANONICAL SOLUTIONS");
        printSetDifference(fullNone, expandedCanonical, "In full NONE but missing from expanded canonical");
        printSetDifference(expandedCanonical, fullNone, "In expanded canonical but not in full NONE");

        tcout << "\nTotal time: " << totalTimer.getSeconds() << "s\n";
        return fullNone == expandedCanonical ? 0 : -1;
    }

} // namespace

int main() {
    std::mt19937_64 rng(RNG_SEED);
    u64 nextSeed = RNG_SEED;

    tcout << "Target depth: " << TARGET_DEPTH << '\n';
    tcout << "Initial seed: 0x" << std::hex << RNG_SEED << std::dec << '\n';
    tcout << "Run mode: ";
    switch (RUN_MODE) {
        case RunMode::SequentialSeedsSearch:
            tcout << "SequentialSeedsSearch\n";
            break;
        case RunMode::RandomTrialsSearch:
            tcout << "RandomTrialsSearch\n";
            break;
        case RunMode::SingleSeedVerbose:
            tcout << "SingleSeedVerbose\n";
            break;
    }

    if (RUN_MODE != RunMode::SingleSeedVerbose) {
        tcout << "Searching for:\n";
        tcout << "  fullNone <= " << SMALL_FULL_NONE_LIMIT << '\n';
        tcout << "  fullNone != expandedCanonical\n";
    }
    tcout << '\n';

    Timer totalTimer;

    if (RUN_MODE == RunMode::SingleSeedVerbose) {
        return runSingleSeedVerbose(totalTimer);
    }

    u64 trial = 0;
    while (true) {
        ++trial;

        tcout << "\n============================================================\n";
        tcout << "TRIAL " << trial;
        if (RUN_MODE == RunMode::RandomTrialsSearch) {
            tcout << "/" << TRIAL_COUNT;
        }
        tcout << '\n';
        tcout << "============================================================\n";

        PuzzleInstance puzzle;
        const bool found = generateExactDepthPuzzle(
                rng,
                puzzle,
                TARGET_DEPTH,
                MAX_PUZZLE_GENERATION_ATTEMPTS,
                nextSeed);

        if (!found) {
            tcout << "Failed to generate an acceptable puzzle within attempt budget.\n";
            return 2;
        }

        FrontierCache cache(puzzle.start, puzzle.goal);

        const std::set<std::string> fullNone =
                collectSolutionSetBySplit<eSequenceDir::NONE, eSequenceDir::NONE>(
                        cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, false, "FULL NONE");

        const std::set<std::string> expandedCanonical =
                collectSolutionSetBySplit<eSequenceDir::ASCENDING, eSequenceDir::DESCENDING>(
                        cache, LEFT_FINAL_DEPTH, RIGHT_FINAL_DEPTH, true, "EXPANDED CANONICAL");

        tcout << "\n[" << trial << "] "
                  << "fullNone=" << fullNone.size()
                  << " expandedCanonical=" << expandedCanonical.size()
                  << " witness=" << moveVectorToString(puzzle.witnessScramble)
                  << " | seed=0x" << std::hex << puzzle.attemptSeed << std::dec
                  << '\n';

        if (shouldStopOnMismatch(fullNone, expandedCanonical)) {
            const bool smallMismatch = fullNone.size() <= SMALL_FULL_NONE_LIMIT;
            printMismatchReport(puzzle, fullNone, expandedCanonical, totalTimer, smallMismatch);
            return -1;
        }

        if (RUN_MODE == RunMode::RandomTrialsSearch && trial >= TRIAL_COUNT) {
            break;
        }
    }

    tcout << "\nNo mismatches found.\n";
    tcout << "Total time: " << totalTimer.getSeconds() << "s\n";
    return 0;
}