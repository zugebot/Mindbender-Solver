#pragma once
// code/solver/solver_frontier.hpp

#include <atomic>
#include <cstddef>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "code/intersection.hpp"
#include "code/perm_stream.hpp"
#include "frontier_right_index2.hpp"
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
    RightFrontierIndexB1B2 rightFrontierCache_;
    Board rightFrontierCacheRoot_{};
    int rightFrontierCacheDepth_ = -1;
    bool rightFrontierCacheValid_ = false;
    bool enableRightFrontierCache_ = false;
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

    template<bool REVERSE_OUTPUT>
    MUND static std::string buildRecoveredPathString(const Board& leftRoot,
                                                     const Memory& leftMemory,
                                                     const Board& rightRoot,
                                                     const Memory& rightMemory) {
        if constexpr (REVERSE_OUTPUT) {
            if (leftRoot.getFatBool() || rightRoot.getFatBool()) {
                return rightMemory.asmFatString(
                        rightRoot.getFatXY(),
                        &leftMemory,
                        leftRoot.getFatXY()
                );
            }

            return rightMemory.asmString(&leftMemory);
        } else {
            if (leftRoot.getFatBool() || rightRoot.getFatBool()) {
                return leftMemory.asmFatString(
                        leftRoot.getFatXY(),
                        &rightMemory,
                        rightRoot.getFatXY()
                );
            }

            return leftMemory.asmString(&rightMemory);
        }
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

    template<bool REVERSE_OUTPUT>
    MU static void recoverExactSplit(const Board& leftRoot,
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

        const bool leftIsFat = leftRoot.getFatBool();
        const bool rightIsFat = rightRoot.getFatBool();

        for (const auto& [fst, snd] : matches) {
            const Board temp1 = leftIsFat
                                        ? makeBoardWithFatMoves(leftRoot, fst->memory)
                                        : makeBoardWithMoves(leftRoot, fst->memory);

            const Board temp2 = rightIsFat
                                        ? makeBoardWithFatMoves(rightRoot, snd->memory)
                                        : makeBoardWithMoves(rightRoot, snd->memory);

            if (temp1 == temp2) {
                outPaths.insert(buildRecoveredPathString<REVERSE_OUTPUT>(
                        leftRoot,
                        fst->memory,
                        rightRoot,
                        snd->memory
                        ));
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

        recoverExactSplit<REVERSE_SEARCH>(
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

    template<bool REVERSE_OUTPUT>
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

        recoverExactSplit<REVERSE_OUTPUT>(
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

        recoverExactSplit<REVERSE_SEARCH>(
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

    MU static void appendJoinedSolutionsForward(const std::set<std::string>& prefixes,
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
                    if (leftHalf.empty()) {
                        outRaw.insert(s);
                    } else if (s.empty()) {
                        outRaw.insert(leftHalf);
                    } else {
                        outRaw.insert(leftHalf + " " + s);
                    }
                }
            }
        }
    }

    MU static void appendJoinedSolutionsReverse(const std::set<std::string>& seedToGoalPaths,
                                                const std::set<std::string>& middleToSeedPaths,
                                                const std::set<std::string>& startToMiddlePaths,
                                                std::unordered_set<std::string>& outRaw) {
        for (const auto& startToMiddle : startToMiddlePaths) {
            for (const auto& middleToSeed : middleToSeedPaths) {
                std::string leftHalf;
                if (startToMiddle.empty()) {
                    leftHalf = middleToSeed;
                } else if (middleToSeed.empty()) {
                    leftHalf = startToMiddle;
                } else {
                    leftHalf = startToMiddle + " " + middleToSeed;
                }

                for (const auto& seedToGoal : seedToGoalPaths) {
                    if (leftHalf.empty()) {
                        outRaw.insert(seedToGoal);
                    } else if (seedToGoal.empty()) {
                        outRaw.insert(leftHalf);
                    } else {
                        outRaw.insert(leftHalf + " " + seedToGoal);
                    }
                }
            }
        }
    }

    template<bool REVERSE_SEARCH>
    MU static void appendJoinedSolutions(const std::set<std::string>& prefixes,
                                         const std::set<std::string>& middles,
                                         const std::set<std::string>& suffixes,
                                         std::unordered_set<std::string>& outRaw) {
        if constexpr (REVERSE_SEARCH) {
            appendJoinedSolutionsReverse(
                    prefixes,
                    middles,
                    suffixes,
                    outRaw
            );
        } else {
            appendJoinedSolutionsForward(
                    prefixes,
                    middles,
                    suffixes,
                    outRaw
            );
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

        buildUniqueNoneDepthFrontierB1B2<1>(board1, forwardDepth1States, forwardDepth1Hashes, debug);
        buildUniqueNoneDepthFrontierB1B2<1>(board2, reverseDepth1States, reverseDepth1Hashes, debug);

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

        buildUniqueNoneDepthFrontierB1B2<2>(board1, forwardDepth2States, forwardDepth2Hashes, debug);
        buildUniqueNoneDepthFrontierB1B2<2>(board2, reverseDepth2States, reverseDepth2Hashes, debug);

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

    MUND static bool hasExactDepthSolutionShallow(const Board& start,
                                                  const Board& goal,
                                                  const int depth) {
        if (depth == 0) {
            return start == goal;
        }

        FrontierBuilderB1B2 builder(start, false);
        JVec<B1B2> frontier;
        JVec<u64> hashes;
        builder.buildExactNoneDepth(depth, frontier, hashes);

        const B1B2 goalState = goal.asB1B2();
        for (const auto& s : frontier) {
            if (s == goalState) {
                return true;
            }
        }

        return false;
    }

    template<int RIGHT_FRONTIER_DEPTH, bool debug>
    MU void prepareRightFrontierIndex(const Board& searchGoalRoot) {
        bool usedCache = false;

        if (enableRightFrontierCache_
            && rightFrontierCacheValid_
            && rightFrontierCacheDepth_ == RIGHT_FRONTIER_DEPTH
            && rightFrontierCacheRoot_ == searchGoalRoot) {
            rightFrontierIndex_ = rightFrontierCache_;
            usedCache = true;
            tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH << ") reused from cache\n";
        }

        if (usedCache) {
            tcout << "right frontier ranges built for "
                  << rightFrontierIndex_.size()
                  << " states across "
                  << rightFrontierIndex_.rangeCount()
                  << " hash buckets\n";
            return;
        }

        JVec<B1B2> rightFrontierStates;
        JVec<u64> rightFrontierHashes;
        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(searchGoalRoot, rightFrontierStates, rightFrontierHashes, debug);
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

        if (enableRightFrontierCache_) {
            rightFrontierCache_ = rightFrontierIndex_;
            rightFrontierCacheRoot_ = searchGoalRoot;
            rightFrontierCacheDepth_ = RIGHT_FRONTIER_DEPTH;
            rightFrontierCacheValid_ = true;
        }
    }

    template<
            int SEED_DEPTH,
            int LEFT_FRONTIER_DEPTH,
            int RIGHT_FRONTIER_DEPTH,
            bool debug
            >
    MU int findSolutionsFrontierThreadedPreset(int worker_count,
                                               const SearchDirection direction) {
        SearchDirection resolvedDirection = direction;

        if (resolvedDirection == SearchDirection::Auto) {
            resolvedDirection = chooseAutoSearchDirection<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>();
            tcout << "search direction: auto -> " << directionName(resolvedDirection) << '\n';
        } else {
            tcout << "search direction: forced " << directionName(resolvedDirection) << '\n';
        }

        if (resolvedDirection == SearchDirection::Reverse) {
            return findSolutionsFrontierThreadedImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    true
                    >(worker_count);
        }

        return findSolutionsFrontierThreadedImpl<
                SEED_DEPTH,
                LEFT_FRONTIER_DEPTH,
                RIGHT_FRONTIER_DEPTH,
                debug,
                false
                >(worker_count);
    }

    template<bool debug>
    MU int runThreadedForTotalDepth(const int totalDepth,
                                    int worker_count,
                                    const SearchDirection direction) {
        switch (totalDepth) {
            case 3:
                return findSolutionsFrontierThreadedPreset<1, 1, 1, debug>(worker_count, direction);
            case 4:
                return findSolutionsFrontierThreadedPreset<1, 1, 2, debug>(worker_count, direction);
            case 5:
                return findSolutionsFrontierThreadedPreset<1, 1, 3, debug>(worker_count, direction);
            case 6:
                return findSolutionsFrontierThreadedPreset<1, 1, 4, debug>(worker_count, direction);
            case 7:
                return findSolutionsFrontierThreadedPreset<1, 2, 4, debug>(worker_count, direction);
            case 8:
                return findSolutionsFrontierThreadedPreset<1, 3, 4, debug>(worker_count, direction);
            case 9:
                return findSolutionsFrontierThreadedPreset<1, 4, 4, debug>(worker_count, direction);
            case 10:
                return findSolutionsFrontierThreadedPreset<1, 4, 5, debug>(worker_count, direction);
            case 11:
                return findSolutionsFrontierThreadedPreset<1, 5, 5, debug>(worker_count, direction);
            case 12:
                return findSolutionsFrontierThreadedPreset<2, 5, 5, debug>(worker_count, direction);
            case 13:
                return findSolutionsFrontierThreadedPreset<3, 5, 5, debug>(worker_count, direction);
            default:
                return -1;
        }
    }

    template<
            int SEED_DEPTH,
            int LEFT_FRONTIER_DEPTH,
            int RIGHT_FRONTIER_DEPTH,
            bool debug,
            bool REVERSE_SEARCH
            >
    MU int findSolutionsFrontierImpl() {
        static constexpr int TOTAL_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH + RIGHT_FRONTIER_DEPTH;

        static constexpr u32 PREFIX_LEFT_DEPTH  = SEED_DEPTH / 2;
        static constexpr u32 PREFIX_RIGHT_DEPTH = SEED_DEPTH - PREFIX_LEFT_DEPTH;

        static constexpr u32 SEED_LEFT_DEPTH    = LEFT_FRONTIER_DEPTH / 2;
        static constexpr u32 SEED_RIGHT_DEPTH   = LEFT_FRONTIER_DEPTH - SEED_LEFT_DEPTH;

        static constexpr u32 GOAL_LEFT_DEPTH    = RIGHT_FRONTIER_DEPTH / 2;
        static constexpr u32 GOAL_RIGHT_DEPTH   = RIGHT_FRONTIER_DEPTH - GOAL_LEFT_DEPTH;

        resultSet.clear();
        expandedResultSet.clear();

        prefixLeftCache_.valid = false;
        goalRightCache_.valid = false;
        rightFrontierIndex_.clear();

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        const Timer totalTime;

        JVec<B1B2> leftSeeds;
        JVec<u64> leftSeedHashes;
        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(searchStartRoot, leftSeeds, leftSeedHashes, debug);
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';

        prepareRightFrontierIndex<RIGHT_FRONTIER_DEPTH, debug>(searchGoalRoot);

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
                      << "] streaming left seed(" << SEED_DEPTH << ") +" 
                      << LEFT_FRONTIER_DEPTH << "\n" << std::flush;
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
                tcout << "    middle matches: " << middleMatches.size()
                      << "\n    streamed states: " << metrics.streamedStateCount
                      << "\n    permstream time: " << metrics.permStreamSeconds
                      << "\n    probe time: " << metrics.probeSeconds
                      << '\n';
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

                recoverSeedToMiddle<REVERSE_SEARCH>(
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
            if (shouldWriteSolutionsToFile()) {
                writeExpandedSolutions(TOTAL_DEPTH);
            }
        } else {
            tcout << "No solutions found...\n";
        }

        return getExpandedSolutionCount();
    }

    template<
            int SEED_DEPTH,
            int LEFT_FRONTIER_DEPTH,
            int RIGHT_FRONTIER_DEPTH,
            bool debug,
            bool REVERSE_SEARCH
            >
    MU int findSolutionsFrontierThreadedImpl(int worker_count) {
        static constexpr int TOTAL_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH + RIGHT_FRONTIER_DEPTH;

        static constexpr u32 PREFIX_LEFT_DEPTH  = SEED_DEPTH / 2;
        static constexpr u32 PREFIX_RIGHT_DEPTH = SEED_DEPTH - PREFIX_LEFT_DEPTH;

        static constexpr u32 SEED_LEFT_DEPTH    = LEFT_FRONTIER_DEPTH / 2;
        static constexpr u32 SEED_RIGHT_DEPTH   = LEFT_FRONTIER_DEPTH - SEED_LEFT_DEPTH;

        static constexpr u32 GOAL_LEFT_DEPTH    = RIGHT_FRONTIER_DEPTH / 2;
        static constexpr u32 GOAL_RIGHT_DEPTH   = RIGHT_FRONTIER_DEPTH - GOAL_LEFT_DEPTH;

        resultSet.clear();
        expandedResultSet.clear();

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

        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(searchStartRoot, leftSeeds, leftSeedHashes, debug);
        
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';
        
        {
            Timer sortTimer;
            struct SeedAndHash {
                B1B2 seed;
                u64 hash;
            };
    
            std::vector<SeedAndHash> scoredSeeds;
            scoredSeeds.reserve(leftSeeds.size());
    
            for (std::size_t i = 0; i < leftSeeds.size(); ++i) {
                scoredSeeds.push_back({leftSeeds[i], leftSeedHashes[i]});
            }
    
            const B1B2 goalState = searchGoalRoot.asB1B2();
    
            std::sort(scoredSeeds.begin(), scoredSeeds.end(), 
                      [&](const SeedAndHash& a, const SeedAndHash& b) {
                const u64 sa = a.seed.getScore1(goalState);
                const u64 sb = b.seed.getScore1(goalState);
    
                if (sa != sb) {
                    return sa < sb;
                }
    
                return a.hash < b.hash;
            });
    
            for (std::size_t i = 0; i < scoredSeeds.size(); ++i) {
                leftSeeds[i] = scoredSeeds[i].seed;
                leftSeedHashes[i] = scoredSeeds[i].hash;
            }
            
            tcout << "sort time: " << sortTimer.getSeconds() << '\n';
        }

        if constexpr (debug) {
            tcout << "\n#####################################\n";
            tcout << "#      Building RIGHT_FRONTIER      #\n";
            tcout << "#####################################\n\n";
        }

        prepareRightFrontierIndex<RIGHT_FRONTIER_DEPTH, debug>(searchGoalRoot);

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
                    tcout << "[seed " << (i + 1) << "/" << leftSeeds.size()
                          << "] streaming left seed(" << SEED_DEPTH << ") +" 
                          << LEFT_FRONTIER_DEPTH << "\n" << std::flush;
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
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "[worker " << workerId << "]"
                             "\n  middle  matches: " << middleMatches.size()
                          << "\n  streamed states: " << metrics.streamedStateCount
                          << "\n  permstream time: " << metrics.permStreamSeconds
                          << "\n  probe      time: " << metrics.probeSeconds
                          << '\n';
                }

                if (middleMatches.empty()) {
                    continue;
                }

                prefixPaths.clear();
                if constexpr (SEED_DEPTH == 0) {
                    prefixPaths.insert("");
                } else {
                    ensureCache(seedPrefixRightCache, seedBoard, PREFIX_RIGHT_DEPTH);

                    Board& searchStartRootLocal = getSearchStartBoard<REVERSE_SEARCH>();

                    recoverExactSplit<REVERSE_SEARCH>(
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

                    recoverSeedToMiddle<REVERSE_SEARCH>(
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
            if (shouldWriteSolutionsToFile()) {
                writeExpandedSolutions(TOTAL_DEPTH);
            }
        } else {
            tcout << "No solutions found...\n";
        }

        return getExpandedSolutionCount();
    }

public:
    using BoardSolverBase::BoardSolverBase;

    template<
            int SEED_DEPTH = 1,
            int LEFT_FRONTIER_DEPTH = 5,
            int RIGHT_FRONTIER_DEPTH = 5,
            bool debug = true
            >
    MU int findSolutionsFrontier(SearchDirection direction = SearchDirection::Auto) {
        SearchDirection resolvedDirection = direction;

        if (resolvedDirection == SearchDirection::Auto) {
            resolvedDirection = chooseAutoSearchDirection<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>();
            tcout << "search direction: auto -> " << directionName(resolvedDirection) << '\n';
        } else {
            tcout << "search direction: forced " << directionName(resolvedDirection) << '\n';
        }

        if (resolvedDirection == SearchDirection::Reverse) {
            return findSolutionsFrontierImpl<
                    SEED_DEPTH,
                    LEFT_FRONTIER_DEPTH,
                    RIGHT_FRONTIER_DEPTH,
                    debug,
                    true
                    >();
        } else {
            return findSolutionsFrontierImpl<
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
    MU int findSolutionsFrontierThreaded(int worker_count = 1,
                                         SearchDirection direction = SearchDirection::Auto,
                                         const bool ensureNoLowerSolutions = false,
                                         const bool enableDepth5RightCache = false) {
        static constexpr int TARGET_TOTAL_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH + RIGHT_FRONTIER_DEPTH;

        enableRightFrontierCache_ = enableDepth5RightCache;

        SearchDirection rampDirection = direction;
        if (ensureNoLowerSolutions && rampDirection == SearchDirection::Auto) {
            if constexpr (debug) {
                rampDirection = chooseAutoSearchDirection<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>();
                tcout << "search direction: auto (locked for ramp-up) -> " << directionName(rampDirection) << '\n';
            } else {
                rampDirection = SearchDirection::Forward;
                tcout << "search direction: auto (locked for ramp-up) -> forward (preview disabled)\n";
            }
        }

        if (!ensureNoLowerSolutions) {
            return findSolutionsFrontierThreadedPreset<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>(
                    worker_count,
                    direction
            );
        }

        for (int d = 0; d < TARGET_TOTAL_DEPTH; ++d) {
            if (d <= 2) {
                if (hasExactDepthSolutionShallow(board1, board2, d)) {
                    resultSet.clear();
                    expandedResultSet.clear();
                    tcout << "lower-depth solution found at depth " << d << ", skipping target depth "
                          << TARGET_TOTAL_DEPTH << '\n';
                    return 0;
                }
                continue;
            }

            const int lowerCount = runThreadedForTotalDepth<debug>(d, worker_count, rampDirection);
            if (lowerCount < 0) {
                tcout << "unsupported ramp-up depth: " << d << '\n';
                return -1;
            }

            if (lowerCount > 0) {
                resultSet.clear();
                expandedResultSet.clear();
                tcout << "lower-depth solution found at depth " << d << ", skipping target depth "
                      << TARGET_TOTAL_DEPTH << '\n';
                return 0;
            }
        }

        return findSolutionsFrontierThreadedPreset<SEED_DEPTH, LEFT_FRONTIER_DEPTH, RIGHT_FRONTIER_DEPTH, debug>(
                worker_count,
                rampDirection
        );
    }
};