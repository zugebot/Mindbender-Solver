#pragma once
// code/solver_frontier.hpp

#include <atomic>
#include <cstddef>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "frontier_right_index.hpp"
#include "intersection.hpp"
#include "perm_stream.hpp"
#include "solver_base.hpp"

class MU BoardSolverFrontier : public BoardSolverBase {
public:
    enum class SearchDirection : u8 {
        Auto = 0,
        Forward,
        Reverse
    };

private:
    struct RecoveryBoardFrontierCache {
        Board root{};
        u32 depth = 0;
        bool valid = false;

        JVec<Board> states;
        JVec<u64> hashes;

        BoardSorter<Board> sorter;
    };

    RightFrontierIndexB1B2 rightFrontierIndex_;
    RecoveryBoardFrontierCache prefixLeftCache_;
    RecoveryBoardFrontierCache goalRightCache_;

    struct StreamProbeMetrics {
        double permStreamSeconds = 0.0;
        double probeSeconds = 0.0;
        double appendSeconds = 0.0;
        double finalSortSeconds = 0.0;
        double finalDedupeSeconds = 0.0;

        u64 chunkCount = 0;
        u64 streamedStateCount = 0;
        u64 preMergeMatchCount = 0;

        RightFrontierIndexB1B2::ProbeStats probeStats{};
    };

private:
    static constexpr double AUTO_DIRECTION_RATIO_THRESHOLD = 1.20;

private:
    MUND static const char* directionName(const SearchDirection direction) {
        switch (direction) {
            case SearchDirection::Auto:
                return "auto";
            case SearchDirection::Forward:
                return "forward";
            case SearchDirection::Reverse:
                return "reverse";
        }
        return "unknown";
    }

    template<bool REVERSE_SEARCH>
    MUND static const char* directionNameTemplated() {
        if constexpr (REVERSE_SEARCH) {
            return "reverse";
        } else {
            return "forward";
        }
    }

    MUND static SearchDirection smallerSideToDirection(const std::size_t forwardCount,
                                                       const std::size_t reverseCount) {
        if (forwardCount <= reverseCount) {
            return SearchDirection::Forward;
        }
        return SearchDirection::Reverse;
    }

    MUND static bool ratioClearlyDifferent(const std::size_t lhs,
                                           const std::size_t rhs) {
        if (lhs == rhs) {
            return false;
        }

        const double smaller = static_cast<double>(std::min(lhs, rhs));
        const double larger = static_cast<double>(std::max(lhs, rhs));

        if (smaller <= 0.0) {
            return true;
        }

        return (larger / smaller) >= AUTO_DIRECTION_RATIO_THRESHOLD;
    }

    template<bool REVERSE_SEARCH>
    MUND Board& getSearchStartBoard() {
        if constexpr (REVERSE_SEARCH) {
            return board2;
        } else {
            return board1;
        }
    }

    template<bool REVERSE_SEARCH>
    MUND Board& getSearchGoalBoard() {
        if constexpr (REVERSE_SEARCH) {
            return board1;
        } else {
            return board2;
        }
    }

    MU static std::string reverseNormalSolutionString(const std::string& solution) {
        if (solution.empty()) {
            return {};
        }

        std::string copy = solution;
        std::vector<u8> moves = Memory::parseNormMoveString(copy);

        if (moves.empty()) {
            return solution;
        }

        std::string out;
        out.reserve(solution.size());

        for (std::size_t i = moves.size(); i > 0; --i) {
            if (!out.empty()) {
                out += ' ';
            }
            out += Memory::formatMoveString(moves[i - 1], false);
        }

        return out;
    }

    MU static void buildBoardExactNoneFrontier(const Board& root,
                                               const u32 depth,
                                               RecoveryBoardFrontierCache& cache) {
        cache.states.clear();
        cache.hashes.clear();

        Perms<Board>::getDepthFunc<eSequenceDir::NONE>(
                root,
                cache.states,
                cache.hashes,
                depth,
                true
        );

        cache.sorter.sortBoards(cache.states, cache.hashes, depth, root.getColorCount());
    }

    MU void ensureCache(RecoveryBoardFrontierCache& cache,
                        const Board& root,
                        const u32 depth) {
        if (cache.valid && cache.depth == depth && cache.root == root) {
            return;
        }

        cache.root = root;
        cache.depth = depth;
        cache.valid = true;
        buildBoardExactNoneFrontier(cache.root, cache.depth, cache);
    }

    MU static void recoverExactNormalSplit(const Board& leftRoot,
                                           const u32 leftDepth,
                                           const JVec<Board>& leftStates,
                                           const JVec<u64>& leftHashes,
                                           const Board& rightRoot,
                                           const u32 rightDepth,
                                           const JVec<Board>& rightStates,
                                           const JVec<u64>& rightHashes,
                                           std::set<std::string>& outPaths) {
        outPaths.clear();

        const auto matches = (leftDepth != 0 && rightDepth != 0)
                                 ? intersection_threaded(leftStates, leftHashes, rightStates, rightHashes)
                                 : intersection(leftStates, leftHashes, rightStates, rightHashes);

        for (const auto& [fst, snd] : matches) {
            const Board temp1 = makeBoardWithMoves(leftRoot, fst->memory);
            const Board temp2 = makeBoardWithMoves(rightRoot, snd->memory);

            if (temp1 == temp2) {
                outPaths.insert(fst->memory.asmString(&snd->memory));
            }
        }
    }

    template<bool REVERSE_SEARCH>
    MU void recoverSeedPrefixes(const Board& seedBoard,
                                const u32 prefixLeftDepth,
                                const u32 prefixRightDepth,
                                std::set<std::string>& outPrefixes) {
        outPrefixes.clear();

        if (prefixLeftDepth == 0 && prefixRightDepth == 0) {
            outPrefixes.insert("");
            return;
        }

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();

        ensureCache(prefixLeftCache_, searchStartRoot, prefixLeftDepth);

        RecoveryBoardFrontierCache seedPrefixRightCache;
        ensureCache(seedPrefixRightCache, seedBoard, prefixRightDepth);

        recoverExactNormalSplit(
                searchStartRoot,
                prefixLeftDepth,
                prefixLeftCache_.states,
                prefixLeftCache_.hashes,
                seedBoard,
                prefixRightDepth,
                seedPrefixRightCache.states,
                seedPrefixRightCache.hashes,
                outPrefixes
        );
    }

    MU void recoverSeedToMiddle(const Board& seedBoard,
                                const B1B2& middleState,
                                const u32 seedLeftDepth,
                                const u32 middleRightDepth,
                                RecoveryBoardFrontierCache& seedLeftCache,
                                RecoveryBoardFrontierCache& middleRightCache,
                                std::set<std::string>& outPaths) {
        Board middleBoard = makeBoardFromState(middleState);

        ensureCache(seedLeftCache, seedBoard, seedLeftDepth);
        ensureCache(middleRightCache, middleBoard, middleRightDepth);

        recoverExactNormalSplit(
                seedBoard,
                seedLeftDepth,
                seedLeftCache.states,
                seedLeftCache.hashes,
                middleBoard,
                middleRightDepth,
                middleRightCache.states,
                middleRightCache.hashes,
                outPaths
        );
    }

    template<bool REVERSE_SEARCH>
    MU void recoverMiddleToGoal(const B1B2& middleState,
                                const u32 middleLeftDepth,
                                const u32 goalRightDepth,
                                RecoveryBoardFrontierCache& middleLeftCache,
                                std::set<std::string>& outPaths) {
        Board middleBoard = makeBoardFromState(middleState);
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        ensureCache(middleLeftCache, middleBoard, middleLeftDepth);
        ensureCache(goalRightCache_, searchGoalRoot, goalRightDepth);

        recoverExactNormalSplit(
                middleBoard,
                middleLeftDepth,
                middleLeftCache.states,
                middleLeftCache.hashes,
                searchGoalRoot,
                goalRightDepth,
                goalRightCache_.states,
                goalRightCache_.hashes,
                outPaths
        );
    }

    template<bool REVERSE_SEARCH>
    MU static void appendJoinedSolutions(const std::set<std::string>& prefixes,
                                         const std::set<std::string>& middles,
                                         const std::set<std::string>& suffixes,
                                         std::unordered_set<std::string>& outRaw) {
        for (const auto& p : prefixes) {
            for (const auto& m : middles) {
                std::string leftHalf;
                if (p.empty()) {
                    leftHalf = m;
                } else if (m.empty()) {
                    leftHalf = p;
                } else {
                    leftHalf = p + " " + m;
                }

                for (const auto& s : suffixes) {
                    std::string fullSolution;
                    if (leftHalf.empty()) {
                        fullSolution = s;
                    } else if (s.empty()) {
                        fullSolution = leftHalf;
                    } else {
                        fullSolution = leftHalf + " " + s;
                    }

                    if constexpr (REVERSE_SEARCH) {
                        outRaw.insert(reverseNormalSolutionString(fullSolution));
                    } else {
                        outRaw.insert(std::move(fullSolution));
                    }
                }
            }
        }
    }

    MU static void appendStates(JVec<B1B2>& dstStates,
                                JVec<u64>& dstHashes,
                                const JVec<B1B2>& srcStates,
                                const JVec<u64>& srcHashes) {
        if (srcStates.empty()) {
            return;
        }

        const std::size_t oldSize = dstStates.size();
        dstStates.resize(oldSize + srcStates.size());
        dstHashes.resize(oldSize + srcHashes.size());

        for (std::size_t i = 0; i < srcStates.size(); ++i) {
            dstStates[oldSize + i] = srcStates[i];
            dstHashes[oldSize + i] = srcHashes[i];
        }
    }

    template<int LEFT_FRONTIER_DEPTH, int RIGHT_FRONTIER_DEPTH, bool COLLECT_PROBE_STATS>
    MU void collectLeftFrontierMiddleMatchesStreamed(const Board& seedBoard,
                                                     JVec<B1B2>& outMiddleMatches,
                                                     JVec<u64>& outMiddleMatchHashes,
                                                     StreamProbeMetrics& metrics) {
        outMiddleMatches.clear();
        outMiddleMatchHashes.clear();
        metrics = {};

        struct Sink {
            const RightFrontierIndexB1B2& rightIndex;
            JVec<B1B2>& allMatches;
            JVec<u64>& allMatchHashes;
            StreamProbeMetrics& metrics;

            JVec<B1B2> chunkMatches;
            JVec<u64> chunkMatchHashes;

            MU Sink(const RightFrontierIndexB1B2& rightIndexIn,
                    JVec<B1B2>& allMatchesIn,
                    JVec<u64>& allMatchHashesIn,
                    StreamProbeMetrics& metricsIn)
                : rightIndex(rightIndexIn),
                  allMatches(allMatchesIn),
                  allMatchHashes(allMatchHashesIn),
                  metrics(metricsIn),
                  chunkMatches(),
                  chunkMatchHashes() {}

            MU void operator()(JVec<B1B2>& chunkStates,
                               JVec<u64>& chunkHashes,
                               u32 count) {
                if (count == 0) {
                    return;
                }

                chunkStates.resize(count);
                chunkHashes.resize(count);

                if constexpr (COLLECT_PROBE_STATS) {
                    ++metrics.chunkCount;
                    metrics.streamedStateCount += static_cast<u64>(count);

                    RightFrontierIndexB1B2::ProbeStats localProbeStats{};

                    {
                        Timer timerProbe;
                        rightIndex.template collectMatches<RIGHT_FRONTIER_DEPTH, true>(
                                chunkStates,
                                chunkHashes,
                                chunkMatches,
                                chunkMatchHashes,
                                &localProbeStats
                        );
                        metrics.probeSeconds += timerProbe.getSeconds();
                    }

                    metrics.probeStats.leftStateCount += localProbeStats.leftStateCount;
                    metrics.probeStats.hashHits += localProbeStats.hashHits;
                    metrics.probeStats.hashMisses += localProbeStats.hashMisses;
                    metrics.probeStats.bucketsVisited += localProbeStats.bucketsVisited;
                    metrics.probeStats.bucketStatesScanned += localProbeStats.bucketStatesScanned;
                    metrics.probeStats.equalityChecks += localProbeStats.equalityChecks;
                    metrics.probeStats.exactMatches += localProbeStats.exactMatches;
                    metrics.probeStats.highFilterRejects += localProbeStats.highFilterRejects;
                    metrics.probeStats.midFilterRejects += localProbeStats.midFilterRejects;
                    metrics.probeStats.lowFilterRejects += localProbeStats.lowFilterRejects;
                    metrics.probeStats.prefixRejects += localProbeStats.prefixRejects;
                    metrics.probeStats.filterPasses += localProbeStats.filterPasses;

                    metrics.preMergeMatchCount += static_cast<u64>(chunkMatches.size());

                    {
                        Timer timerAppend;
                        BoardSolverFrontier::appendStates(
                                allMatches,
                                allMatchHashes,
                                chunkMatches,
                                chunkMatchHashes
                        );
                        metrics.appendSeconds += timerAppend.getSeconds();
                    }
                } else {
                    rightIndex.template collectMatches<RIGHT_FRONTIER_DEPTH, false>(
                            chunkStates,
                            chunkHashes,
                            chunkMatches,
                            chunkMatchHashes,
                            nullptr
                    );

                    BoardSolverFrontier::appendStates(
                            allMatches,
                            allMatchHashes,
                            chunkMatches,
                            chunkMatchHashes
                    );
                }
            }
        };

        Sink sink(
                rightFrontierIndex_,
                outMiddleMatches,
                outMiddleMatchHashes,
                metrics
        );

        if constexpr (COLLECT_PROBE_STATS) {
            {
                Timer timerPermStream;
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_FRONTIER_DEPTH>(
                        seedBoard,
                        sink,
                        LEFT_STREAM_CHUNK_SIZE
                );
                metrics.permStreamSeconds = timerPermStream.getSeconds();
            }

            if (!outMiddleMatches.empty()) {
                {
                    Timer timerSort;
                    sortStatesByHash(outMiddleMatches, outMiddleMatchHashes);
                    metrics.finalSortSeconds = timerSort.getSeconds();
                }

                {
                    Timer timerDedupe;
                    compactUniqueSortedStatesInPlace(outMiddleMatches, outMiddleMatchHashes);
                    metrics.finalDedupeSeconds = timerDedupe.getSeconds();
                }
            }
        } else {
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_FRONTIER_DEPTH>(
                    seedBoard,
                    sink,
                    LEFT_STREAM_CHUNK_SIZE
            );

            if (!outMiddleMatches.empty()) {
                sortStatesByHash(outMiddleMatches, outMiddleMatchHashes);
                compactUniqueSortedStatesInPlace(outMiddleMatches, outMiddleMatchHashes);
            }
        }
    }

    template<int SEED_DEPTH, int LEFT_FRONTIER_DEPTH, int RIGHT_FRONTIER_DEPTH, bool debug>
    MUND SearchDirection chooseAutoSearchDirection() {
        JVec<B1B2> forwardDepth1States;
        JVec<u64> forwardDepth1Hashes;
        JVec<B1B2> reverseDepth1States;
        JVec<u64> reverseDepth1Hashes;

        buildUniqueNoneDepthFrontierB1B2<1>(board1, forwardDepth1States, forwardDepth1Hashes);
        buildUniqueNoneDepthFrontierB1B2<1>(board2, reverseDepth1States, reverseDepth1Hashes);

        const std::size_t forwardDepth1Count = forwardDepth1States.size();
        const std::size_t reverseDepth1Count = reverseDepth1States.size();

        if constexpr (debug) {
            tcout << "auto direction preview depth 1:\n";
            tcout << "    forward count: " << forwardDepth1Count << '\n';
            tcout << "    reverse count: " << reverseDepth1Count << '\n';
        }

        if (ratioClearlyDifferent(forwardDepth1Count, reverseDepth1Count)) {
            return smallerSideToDirection(forwardDepth1Count, reverseDepth1Count);
        }

        JVec<B1B2> forwardDepth2States;
        JVec<u64> forwardDepth2Hashes;
        JVec<B1B2> reverseDepth2States;
        JVec<u64> reverseDepth2Hashes;

        buildUniqueNoneDepthFrontierB1B2<2>(board1, forwardDepth2States, forwardDepth2Hashes);
        buildUniqueNoneDepthFrontierB1B2<2>(board2, reverseDepth2States, reverseDepth2Hashes);

        const std::size_t forwardDepth2Count = forwardDepth2States.size();
        const std::size_t reverseDepth2Count = reverseDepth2States.size();

        if constexpr (debug) {
            tcout << "auto direction preview depth 2:\n";
            tcout << "    forward count: " << forwardDepth2Count << '\n';
            tcout << "    reverse count: " << reverseDepth2Count << '\n';
        }

        if (forwardDepth2Count != reverseDepth2Count) {
            return smallerSideToDirection(forwardDepth2Count, reverseDepth2Count);
        }

        if (forwardDepth1Count != reverseDepth1Count) {
            return smallerSideToDirection(forwardDepth1Count, reverseDepth1Count);
        }

        return SearchDirection::Forward;
    }

    template<
            int SEED_DEPTH,
            int LEFT_FRONTIER_DEPTH,
            int RIGHT_FRONTIER_DEPTH,
            bool debug,
            bool REVERSE_SEARCH
            >
    MU void findSolutionsFrontierImpl() {
        static constexpr int TOTAL_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH + RIGHT_FRONTIER_DEPTH;

        static constexpr u32 PREFIX_LEFT_DEPTH  = SEED_DEPTH / 2;
        static constexpr u32 PREFIX_RIGHT_DEPTH = SEED_DEPTH - PREFIX_LEFT_DEPTH;

        static constexpr u32 SEED_LEFT_DEPTH    = LEFT_FRONTIER_DEPTH / 2;
        static constexpr u32 SEED_RIGHT_DEPTH   = LEFT_FRONTIER_DEPTH - SEED_LEFT_DEPTH;

        static constexpr u32 GOAL_LEFT_DEPTH    = RIGHT_FRONTIER_DEPTH / 2;
        static constexpr u32 GOAL_RIGHT_DEPTH   = RIGHT_FRONTIER_DEPTH - GOAL_LEFT_DEPTH;

        resultSet.clear();
        expandedResultSet.clear();

        if (hasFat) {
            tcout << "findSolutionsFrontier currently only supports non-fat puzzles.\n";
            return;
        }

        prefixLeftCache_.valid = false;
        goalRightCache_.valid = false;
        rightFrontierIndex_.clear();

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        const Timer totalTime;

        JVec<B1B2> leftSeeds;
        JVec<u64> leftSeedHashes;
        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(searchStartRoot, leftSeeds, leftSeedHashes);
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';

        JVec<B1B2> rightFrontierStates;
        JVec<u64> rightFrontierHashes;
        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(searchGoalRoot, rightFrontierStates, rightFrontierHashes);
        tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH << ") final unique size: "
              << rightFrontierStates.size() << '\n';

        rightFrontierIndex_.template buildFromUniqueStates<debug>(
                std::move(rightFrontierStates),
                std::move(rightFrontierHashes),
                searchGoalRoot
        );
        tcout << "right frontier ranges built for "
              << rightFrontierIndex_.size()
              << " states across "
              << rightFrontierIndex_.rangeCount()
              << " hash buckets\n";

        if constexpr (debug) {
            rightFrontierIndex_.printStats();
        }

        JVec<B1B2> middleMatches;
        JVec<u64> middleMatchHashes;

        RecoveryBoardFrontierCache seedLeftCache;
        RecoveryBoardFrontierCache middleRightCache;
        RecoveryBoardFrontierCache middleLeftCache;

        std::set<std::string> prefixPaths;
        std::set<std::string> seedToMiddlePaths;
        std::set<std::string> middleToGoalPaths;

        for (std::size_t i = 0; i < leftSeeds.size(); ++i) {
            if constexpr (debug) {
                tcout << "[seed " << (i + 1) << "/" << leftSeeds.size()
                      << "] streaming left frontier +" << LEFT_FRONTIER_DEPTH
                      << " from seed(" << SEED_DEPTH << ")\n" << std::flush;
            }

            Board seedBoard = makeBoardFromState(leftSeeds[i]);

            StreamProbeMetrics metrics;
            collectLeftFrontierMiddleMatchesStreamed<LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>(
                    seedBoard,
                    middleMatches,
                    middleMatchHashes,
                    metrics
            );

            if constexpr (debug) {
                double estimatedStreamOnlySeconds =
                        metrics.permStreamSeconds - metrics.probeSeconds - metrics.appendSeconds;
                if (estimatedStreamOnlySeconds < 0.0) {
                    estimatedStreamOnlySeconds = 0.0;
                }

                tcout << "    middle matches: " << middleMatches.size() << '\n';
                tcout << "    permstream total time: " << metrics.permStreamSeconds << '\n';
                tcout << "    estimated stream-only time: " << estimatedStreamOnlySeconds << '\n';
                tcout << "    probe time: " << metrics.probeSeconds << '\n';
                tcout << "    append time: " << metrics.appendSeconds << '\n';
                tcout << "    final sort time: " << metrics.finalSortSeconds << '\n';
                tcout << "    final dedupe time: " << metrics.finalDedupeSeconds << '\n';
                tcout << "    streamed chunks: " << metrics.chunkCount << '\n';
                tcout << "    streamed states: " << metrics.streamedStateCount << '\n';
                tcout << "    pre-merge match count: " << metrics.preMergeMatchCount << '\n';
                tcout << "    probe hash hits: " << metrics.probeStats.hashHits << '\n';
                tcout << "    probe hash misses: " << metrics.probeStats.hashMisses << '\n';
                tcout << "    probe buckets visited: " << metrics.probeStats.bucketsVisited << '\n';
                tcout << "    probe bucket states scanned: " << metrics.probeStats.bucketStatesScanned << '\n';
                tcout << "    probe equality checks: " << metrics.probeStats.equalityChecks << '\n';
                tcout << "    probe exact matches: " << metrics.probeStats.exactMatches << '\n';
            }

            if (middleMatches.empty()) {
                if constexpr (debug) {
                    tcout << "    unique raw solutions so far: " << resultSet.size() << '\n';
                }
                continue;
            }

            prefixPaths.clear();
            if constexpr (SEED_DEPTH == 0) {
                prefixPaths.insert("");
            } else {
                recoverSeedPrefixes<REVERSE_SEARCH>(
                        seedBoard,
                        PREFIX_LEFT_DEPTH,
                        PREFIX_RIGHT_DEPTH,
                        prefixPaths
                );
            }

            ensureCache(seedLeftCache, seedBoard, SEED_LEFT_DEPTH);

            for (std::size_t m = 0; m < middleMatches.size(); ++m) {
                middleRightCache.valid = false;
                middleLeftCache.valid = false;

                recoverSeedToMiddle(
                        seedBoard,
                        middleMatches[m],
                        SEED_LEFT_DEPTH,
                        SEED_RIGHT_DEPTH,
                        seedLeftCache,
                        middleRightCache,
                        seedToMiddlePaths
                );

                if (seedToMiddlePaths.empty()) {
                    continue;
                }

                if constexpr (SEED_RIGHT_DEPTH == GOAL_LEFT_DEPTH) {
                    middleLeftCache = middleRightCache;
                    middleLeftCache.valid = true;
                }

                recoverMiddleToGoal<REVERSE_SEARCH>(
                        middleMatches[m],
                        GOAL_LEFT_DEPTH,
                        GOAL_RIGHT_DEPTH,
                        middleLeftCache,
                        middleToGoalPaths
                );

                if (middleToGoalPaths.empty()) {
                    continue;
                }

                appendJoinedSolutions<REVERSE_SEARCH>(
                        prefixPaths,
                        seedToMiddlePaths,
                        middleToGoalPaths,
                        resultSet
                );
            }

            if constexpr (debug) {
                tcout << "    unique raw solutions so far: " << resultSet.size() << '\n';
            }
        }
        
        tcout << "\nPuzzle: " << pair->getName() << '\n';
        tcout << "Total Depth: " << TOTAL_DEPTH << '\n';
        tcout << "Total Time: " << totalTime.getSeconds() << '\n';
        tcout << "Left Seed depth: " << SEED_DEPTH << '\n';
        tcout << "Left frontier depth: " << LEFT_FRONTIER_DEPTH << '\n';
        tcout << "Right frontier depth: " << RIGHT_FRONTIER_DEPTH << '\n';
        tcout << "Search direction: " << directionNameTemplated<REVERSE_SEARCH>() << '\n';

        if (!resultSet.empty()) {
            expandRawSolutionsIntoFinalSet();
            writeExpandedSolutions(TOTAL_DEPTH);
        } else {
            tcout << "No solutions found...\n";
        }
    }

    template<
            int SEED_DEPTH,
            int LEFT_FRONTIER_DEPTH,
            int RIGHT_FRONTIER_DEPTH,
            bool debug,
            bool REVERSE_SEARCH
            >
    MU void findSolutionsFrontierThreadedImpl(int worker_count) {
        static constexpr int TOTAL_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH + RIGHT_FRONTIER_DEPTH;

        static constexpr u32 PREFIX_LEFT_DEPTH  = SEED_DEPTH / 2;
        static constexpr u32 PREFIX_RIGHT_DEPTH = SEED_DEPTH - PREFIX_LEFT_DEPTH;

        static constexpr u32 SEED_LEFT_DEPTH    = LEFT_FRONTIER_DEPTH / 2;
        static constexpr u32 SEED_RIGHT_DEPTH   = LEFT_FRONTIER_DEPTH - SEED_LEFT_DEPTH;

        static constexpr u32 GOAL_LEFT_DEPTH    = RIGHT_FRONTIER_DEPTH / 2;
        static constexpr u32 GOAL_RIGHT_DEPTH   = RIGHT_FRONTIER_DEPTH - GOAL_LEFT_DEPTH;

        resultSet.clear();
        expandedResultSet.clear();

        if (hasFat) {
            tcout << "findSolutionsFrontierThreaded currently only supports non-fat puzzles.\n";
            return;
        }

        if (worker_count < 1) {
            worker_count = 1;
        }

        prefixLeftCache_.valid = false;
        goalRightCache_.valid = false;
        rightFrontierIndex_.clear();

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        const Timer totalTime;

        JVec<B1B2> leftSeeds;
        JVec<u64> leftSeedHashes;

        if constexpr (debug) {
            tcout << "#####################################\n";
            tcout << "#        Building LEFT_SEED         #\n";
            tcout << "#####################################\n\n";
        }

        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(searchStartRoot, leftSeeds, leftSeedHashes);
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';

        JVec<B1B2> rightFrontierStates;
        JVec<u64> rightFrontierHashes;

        if constexpr (debug) {
            tcout << "\n#####################################\n";
            tcout << "#      Building RIGHT_FRONTIER      #\n";
            tcout << "#####################################\n\n";
        }

        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(searchGoalRoot, rightFrontierStates, rightFrontierHashes);
        tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH << ") final unique size: "
              << rightFrontierStates.size() << '\n';

        rightFrontierIndex_.template buildFromUniqueStates<debug>(
                std::move(rightFrontierStates),
                std::move(rightFrontierHashes),
                searchGoalRoot
        );
        tcout << "right frontier ranges built for "
              << rightFrontierIndex_.size()
              << " states across "
              << rightFrontierIndex_.rangeCount()
              << " hash buckets\n";

        if constexpr (debug) {
            tcout << "\n";
            rightFrontierIndex_.printStats();
            tcout << "\n";
        }

        if constexpr (SEED_DEPTH != 0) {
            ensureCache(prefixLeftCache_, searchStartRoot, PREFIX_LEFT_DEPTH);
        }
        ensureCache(goalRightCache_, searchGoalRoot, GOAL_RIGHT_DEPTH);

        std::atomic<std::size_t> nextIndex = 0;
        std::mutex printMutex;
        std::mutex resultMutex;

        auto worker = [&](const std::size_t workerId) {
            JVec<B1B2> middleMatches;
            JVec<u64> middleMatchHashes;

            RecoveryBoardFrontierCache seedLeftCache;
            RecoveryBoardFrontierCache seedPrefixRightCache;
            RecoveryBoardFrontierCache middleRightCache;
            RecoveryBoardFrontierCache middleLeftCache;

            std::set<std::string> prefixPaths;
            std::set<std::string> seedToMiddlePaths;
            std::set<std::string> middleToGoalPaths;
            std::unordered_set<std::string> localRecovered;

            while (true) {
                const std::size_t i = nextIndex.fetch_add(1);
                if (i >= leftSeeds.size()) {
                    break;
                }

                if constexpr (debug) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "[worker " << workerId
                          << "] [seed " << (i + 1) << "/" << leftSeeds.size()
                          << "] streaming left frontier +" << LEFT_FRONTIER_DEPTH
                          << " from seed(" << SEED_DEPTH << ")\n" << std::flush;
                }

                Board seedBoard = makeBoardFromState(leftSeeds[i]);

                StreamProbeMetrics metrics;
                collectLeftFrontierMiddleMatchesStreamed<LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>(
                        seedBoard,
                        middleMatches,
                        middleMatchHashes,
                        metrics
                );

                if constexpr (debug) {
                    double estimatedStreamOnlySeconds =
                            metrics.permStreamSeconds - metrics.probeSeconds - metrics.appendSeconds;
                    if (estimatedStreamOnlySeconds < 0.0) {
                        estimatedStreamOnlySeconds = 0.0;
                    }

                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "    [worker " << workerId
                          << "] middle matches: " << middleMatches.size() << '\n';
                    tcout << "    [worker " << workerId
                          << "] permstream total time: " << metrics.permStreamSeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] estimated stream-only time: " << estimatedStreamOnlySeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe time: " << metrics.probeSeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] append time: " << metrics.appendSeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] final sort time: " << metrics.finalSortSeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] final dedupe time: " << metrics.finalDedupeSeconds << '\n';
                    tcout << "    [worker " << workerId
                          << "] streamed chunks: " << metrics.chunkCount << '\n';
                    tcout << "    [worker " << workerId
                          << "] streamed states: " << metrics.streamedStateCount << '\n';
                    tcout << "    [worker " << workerId
                          << "] pre-merge match count: " << metrics.preMergeMatchCount << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe hash hits: " << metrics.probeStats.hashHits << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe hash misses: " << metrics.probeStats.hashMisses << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe buckets visited: " << metrics.probeStats.bucketsVisited << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe bucket states scanned: " << metrics.probeStats.bucketStatesScanned << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe equality checks: " << metrics.probeStats.equalityChecks << '\n';
                    tcout << "    [worker " << workerId
                          << "] probe exact matches: " << metrics.probeStats.exactMatches << '\n';
                }

                if (middleMatches.empty()) {
                    continue;
                }

                prefixPaths.clear();
                if constexpr (SEED_DEPTH == 0) {
                    prefixPaths.insert("");
                } else {
                    Board& searchStartRootLocal = getSearchStartBoard<REVERSE_SEARCH>();

                    ensureCache(seedPrefixRightCache, seedBoard, PREFIX_RIGHT_DEPTH);

                    recoverExactNormalSplit(
                            searchStartRootLocal,
                            PREFIX_LEFT_DEPTH,
                            prefixLeftCache_.states,
                            prefixLeftCache_.hashes,
                            seedBoard,
                            PREFIX_RIGHT_DEPTH,
                            seedPrefixRightCache.states,
                            seedPrefixRightCache.hashes,
                            prefixPaths
                    );
                }

                ensureCache(seedLeftCache, seedBoard, SEED_LEFT_DEPTH);

                localRecovered.clear();

                for (std::size_t m = 0; m < middleMatches.size(); ++m) {
                    middleRightCache.valid = false;
                    middleLeftCache.valid = false;

                    recoverSeedToMiddle(
                            seedBoard,
                            middleMatches[m],
                            SEED_LEFT_DEPTH,
                            SEED_RIGHT_DEPTH,
                            seedLeftCache,
                            middleRightCache,
                            seedToMiddlePaths
                    );

                    if (seedToMiddlePaths.empty()) {
                        continue;
                    }

                    if constexpr (SEED_RIGHT_DEPTH == GOAL_LEFT_DEPTH) {
                        middleLeftCache = middleRightCache;
                        middleLeftCache.valid = true;
                    }

                    recoverMiddleToGoal<REVERSE_SEARCH>(
                            middleMatches[m],
                            GOAL_LEFT_DEPTH,
                            GOAL_RIGHT_DEPTH,
                            middleLeftCache,
                            middleToGoalPaths
                    );

                    if (middleToGoalPaths.empty()) {
                        continue;
                    }

                    appendJoinedSolutions<REVERSE_SEARCH>(
                            prefixPaths,
                            seedToMiddlePaths,
                            middleToGoalPaths,
                            localRecovered
                    );
                }

                {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    for (const auto& s : localRecovered) {
                        resultSet.insert(s);
                    }
                }

                if constexpr (debug) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "    [worker " << workerId
                          << "] unique raw solutions so far: " << resultSet.size() << '\n';
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));

        for (std::size_t workerId = 0; workerId < static_cast<std::size_t>(worker_count); ++workerId) {
            workers.emplace_back(worker, workerId);
        }

        for (auto& t : workers) {
            t.join();
        }
        
        tcout << "\nTotal Time: " << totalTime.getSeconds() << '\n';
        tcout << "Puzzle: " << pair->getName() << '\n';
        tcout << "Left Seed depth: " << SEED_DEPTH << '\n';
        tcout << "Left frontier depth: " << LEFT_FRONTIER_DEPTH << '\n';
        tcout << "Right frontier depth: " << RIGHT_FRONTIER_DEPTH << '\n';
        tcout << "Search direction: " << directionNameTemplated<REVERSE_SEARCH>() << '\n';

        if (!resultSet.empty()) {
            expandRawSolutionsIntoFinalSet();
            writeExpandedSolutions(TOTAL_DEPTH);
        } else {
            tcout << "No solutions found...\n";
        }
    }

public:
    using BoardSolverBase::BoardSolverBase;

    template<
            int SEED_DEPTH = 1,
            int LEFT_FRONTIER_DEPTH = 5,
            int RIGHT_FRONTIER_DEPTH = 5,
            bool debug = true
            >
    MU void findSolutionsFrontier(SearchDirection direction = SearchDirection::Auto) {
        SearchDirection resolvedDirection = direction;

        if (resolvedDirection == SearchDirection::Auto) {
            resolvedDirection = chooseAutoSearchDirection<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>();
            tcout << "search direction: auto -> " << directionName(resolvedDirection) << '\n';
        } else {
            tcout << "search direction: forced " << directionName(resolvedDirection) << '\n';
        }

        if (resolvedDirection == SearchDirection::Reverse) {
            findSolutionsFrontierImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    true
                    >();
        } else {
            findSolutionsFrontierImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    false
                    >();
        }
    }

    template<
            int SEED_DEPTH = 1,
            int LEFT_FRONTIER_DEPTH = 5,
            int RIGHT_FRONTIER_DEPTH = 5,
            bool debug = true
            >
    MU void findSolutionsFrontierThreaded(int worker_count = 1,
                                          SearchDirection direction = SearchDirection::Auto) {
        SearchDirection resolvedDirection = direction;

        if (resolvedDirection == SearchDirection::Auto) {
            resolvedDirection = chooseAutoSearchDirection<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>();
            tcout << "search direction: auto -> " << directionName(resolvedDirection) << '\n';
        } else {
            tcout << "search direction: forced " << directionName(resolvedDirection) << '\n';
        }

        if (resolvedDirection == SearchDirection::Reverse) {
            findSolutionsFrontierThreadedImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    true
                    >(worker_count);
        } else {
            findSolutionsFrontierThreadedImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    false
                    >(worker_count);
        }
    }
};