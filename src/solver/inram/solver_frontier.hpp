#pragma once
// src/solver/inram/solver_frontier.hpp

#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "src/chuzzles/intersection.hpp"
#include "src/chuzzles/perm_stream.hpp"
#include "src/solver/inram/right_frontier_table.hpp"
#include "src/solver/inram/solver_base.hpp"

class MU BoardSolverFrontier : public BoardSolverBase {
public:
    // Recovery: per-middle halves of at least this many moves are streamed (every path
    // generated and looked up in the shared half on the fly) instead of stored. An all-paths
    // depth-5 frontier is 60^5 = 7.8e8 boards (~30 GB); left depth 10 (16-move <5,5,6>) needs
    // one per middle on top of the shared depth-5 one, which ran 20-2's recovery out of memory.
    inline static u32 recoveryStreamMinDepth = 5;

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

    RightFrontierTableB1B2 rightFrontierIndex_;
    RecoveryBoardFrontierCache goalRightCache_;

    struct StreamProbeMetrics {
        double streamWallSeconds = 0.0;
        double permStreamSeconds = 0.0; // exclusive of synchronous probe/append work
        double probeSeconds = 0.0;
        double appendSeconds = 0.0;
        double finalSortSeconds = 0.0;
        double finalDedupeSeconds = 0.0;
        double taskWallSeconds = 0.0;

        u64 chunkCount = 0;
        u64 streamedStateCount = 0;
        u64 preMergeMatchCount = 0;

        RightFrontierTableB1B2::ProbeStats probeStats{};
    };

    struct GlobalRecoveryMetrics {
        double middleSortSeconds = 0.0;
        double middleDedupeSeconds = 0.0;
        double pathRecoverySeconds = 0.0;
        std::size_t uniqueMiddleCount = 0;
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

    MUND static std::size_t recoveryPathUpperBound(const bool fat,
                                                   const u32 depth) {
        const std::size_t branch = fat ? 48ULL : 60ULL;
        std::size_t result = 1;
        for (u32 i = 0; i < depth; ++i) {
            if (result > std::numeric_limits<std::size_t>::max() / branch) {
                return std::numeric_limits<std::size_t>::max();
            }
            result *= branch;
        }
        return result;
    }

    template<u32 DEPTH, u32 CUR_DEPTH = 0>
    MU static void enumerateAllRecoveryPaths(const Board& board,
                                             const u64 packedMoves,
                                             JVec<Board>& states,
                                             JVec<u64>& hashes,
                                             std::size_t& count) {
        if constexpr (CUR_DEPTH == DEPTH) {
            Board stored = board;
            if constexpr (DEPTH > 0) {
                stored.memory.template setNextNMove<DEPTH>(packedMoves);
            }
            states[count] = stored;
            hashes[count] = StateHash::computeHash(stored);
            ++count;
        } else {
            if (board.getFatBool()) {
                const u8* actionIndexes = fatActionsIndexes[board.getFatXY()];
                for (u64 relativeIndex = 0; relativeIndex < 48ULL; ++relativeIndex) {
                    Board next = board;
                    allActStructList[actionIndexes[relativeIndex]].action(next);
                    if (next == board) {
                        continue;
                    }

                    const u64 nextPacked = packedMoves |
                            (relativeIndex << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));
                    enumerateAllRecoveryPaths<DEPTH, CUR_DEPTH + 1>(
                            next, nextPacked, states, hashes, count);
                }
            } else {
                for (u64 direction = 0; direction < 2ULL; ++direction) {
                    for (u64 section = 0; section < 6ULL; ++section) {
                        const u64 base = (direction << 5ULL) + section * 5ULL;
                        for (u64 shift = 0; shift < 5ULL; ++shift) {
                            const u64 actionIndex = base + shift;
                            Board next = board;
                            allActStructList[actionIndex].action(next);
                            if (next == board) {
                                continue;
                            }

                            const u64 nextPacked = packedMoves |
                                    (actionIndex << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));
                            enumerateAllRecoveryPaths<DEPTH, CUR_DEPTH + 1>(
                                    next, nextPacked, states, hashes, count);
                        }
                    }
                }
            }
        }
    }

    // fn(board) for every exact-length path from `board` (its moves in board.memory), the same
    // paths in the same order as enumerateAllRecoveryPaths, without storing them.
    template<u32 DEPTH, u32 CUR_DEPTH = 0, typename Fn>
    MU static void forEachRecoveryPath(const Board& board, const u64 packedMoves, Fn& fn) {
        if constexpr (CUR_DEPTH == DEPTH) {
            Board stored = board;
            if constexpr (DEPTH > 0) {
                stored.memory.template setNextNMove<DEPTH>(packedMoves);
            }
            fn(stored);
        } else {
            if (board.getFatBool()) {
                const u8* actionIndexes = fatActionsIndexes[board.getFatXY()];
                for (u64 relativeIndex = 0; relativeIndex < 48ULL; ++relativeIndex) {
                    Board next = board;
                    allActStructList[actionIndexes[relativeIndex]].action(next);
                    if (next == board) {
                        continue;
                    }
                    const u64 nextPacked = packedMoves |
                            (relativeIndex << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));
                    forEachRecoveryPath<DEPTH, CUR_DEPTH + 1>(next, nextPacked, fn);
                }
            } else {
                for (u64 direction = 0; direction < 2ULL; ++direction) {
                    for (u64 section = 0; section < 6ULL; ++section) {
                        const u64 base = (direction << 5ULL) + section * 5ULL;
                        for (u64 shift = 0; shift < 5ULL; ++shift) {
                            const u64 actionIndex = base + shift;
                            Board next = board;
                            allActStructList[actionIndex].action(next);
                            if (next == board) {
                                continue;
                            }
                            const u64 nextPacked = packedMoves |
                                    (actionIndex << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));
                            forEachRecoveryPath<DEPTH, CUR_DEPTH + 1>(next, nextPacked, fn);
                        }
                    }
                }
            }
        }
    }

    template<u32 DEPTH>
    MU static void buildBoardAllPathFrontier(const Board& root,
                                             RecoveryBoardFrontierCache& cache) {
        cache.states.clear();
        cache.hashes.clear();

        const std::size_t upper = recoveryPathUpperBound(root.getFatBool(), DEPTH);
        if (upper == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("exact recovery frontier size overflow");
        }

        cache.states.resize(upper);
        cache.hashes.resize(upper);
        std::size_t count = 0;
        enumerateAllRecoveryPaths<DEPTH>(
                root, 0ULL, cache.states, cache.hashes, count);
        cache.states.resize(count);
        cache.hashes.resize(count);
        cache.sorter.sortBoards(cache.states, cache.hashes, DEPTH, root.getColorCount());
    }

    MU static void buildBoardExactNoneFrontier(const Board& root,
                                               const u32 depth,
                                               RecoveryBoardFrontierCache& cache) {
        switch (depth) {
            case 0: buildBoardAllPathFrontier<0>(root, cache); return;
            case 1: buildBoardAllPathFrontier<1>(root, cache); return;
            case 2: buildBoardAllPathFrontier<2>(root, cache); return;
            case 3: buildBoardAllPathFrontier<3>(root, cache); return;
            case 4: buildBoardAllPathFrontier<4>(root, cache); return;
            case 5: buildBoardAllPathFrontier<5>(root, cache); return;
            default:
                throw std::runtime_error("exact recovery half-depth exceeds supported range 0..5");
        }
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

    // All exact-length path pairs meeting at a middle, where one side is a large
    // frontier shared by every middle and the other a small per-middle one:
    // each small entry binary-searches its hash run in the shared side, so the
    // cost is O(small * log(shared)) instead of a full merge over both lanes.
    template<bool REVERSE_OUTPUT, bool SMALL_IS_LEFT>
    MU static void recoverSplitBySharedLookup(const Board& leftRoot,
                                              const Board& rightRoot,
                                              const RecoveryBoardFrontierCache& shared,
                                              const RecoveryBoardFrontierCache& small,
                                              std::set<std::string>& outPaths) {
        outPaths.clear();

        const bool leftIsFat = leftRoot.getFatBool();
        const bool rightIsFat = rightRoot.getFatBool();
        const u64* sharedHashBegin = shared.hashes.data();
        const u64* sharedHashEnd = sharedHashBegin + shared.hashes.size();

        for (std::size_t i = 0; i < small.states.size(); ++i) {
            const Board& smallBoard = small.states[i];
            const auto [lo, hi] = std::equal_range(sharedHashBegin, sharedHashEnd, small.hashes[i]);

            for (const u64* it = lo; it != hi; ++it) {
                const Board& sharedBoard = shared.states[static_cast<std::size_t>(it - sharedHashBegin)];
                if (!(sharedBoard == smallBoard)) {
                    continue;
                }

                const Board& leftBoard = SMALL_IS_LEFT ? smallBoard : sharedBoard;
                const Board& rightBoard = SMALL_IS_LEFT ? sharedBoard : smallBoard;

                const Board temp1 = leftIsFat
                                            ? makeBoardWithFatMoves(leftRoot, leftBoard.memory)
                                            : makeBoardWithMoves(leftRoot, leftBoard.memory);
                const Board temp2 = rightIsFat
                                            ? makeBoardWithFatMoves(rightRoot, rightBoard.memory)
                                            : makeBoardWithMoves(rightRoot, rightBoard.memory);

                if (temp1 == temp2) {
                    outPaths.insert(buildRecoveredPathString<REVERSE_OUTPUT>(
                            leftRoot,
                            leftBoard.memory,
                            rightRoot,
                            rightBoard.memory
                            ));
                }
            }
        }
    }

    // recoverSplitBySharedLookup with the small side streamed from `smallRoot` (DEPTH moves)
    // instead of read from a stored frontier: same matches, no memory for the small side.
    template<bool REVERSE_OUTPUT, bool SMALL_IS_LEFT, u32 DEPTH>
    MU static void recoverSplitBySharedLookupStreamed(const Board& leftRoot,
                                                      const Board& rightRoot,
                                                      const RecoveryBoardFrontierCache& shared,
                                                      const Board& smallRoot,
                                                      std::set<std::string>& outPaths) {
        outPaths.clear();

        const bool leftIsFat = leftRoot.getFatBool();
        const bool rightIsFat = rightRoot.getFatBool();
        const u64* sharedHashBegin = shared.hashes.data();
        const u64* sharedHashEnd = sharedHashBegin + shared.hashes.size();

        auto visit = [&](const Board& smallBoard) {
            const auto [lo, hi] = std::equal_range(sharedHashBegin, sharedHashEnd, StateHash::computeHash(smallBoard));
            for (const u64* it = lo; it != hi; ++it) {
                const Board& sharedBoard = shared.states[static_cast<std::size_t>(it - sharedHashBegin)];
                if (!(sharedBoard == smallBoard)) {
                    continue;
                }

                const Board& leftBoard = SMALL_IS_LEFT ? smallBoard : sharedBoard;
                const Board& rightBoard = SMALL_IS_LEFT ? sharedBoard : smallBoard;

                const Board temp1 = leftIsFat
                                            ? makeBoardWithFatMoves(leftRoot, leftBoard.memory)
                                            : makeBoardWithMoves(leftRoot, leftBoard.memory);
                const Board temp2 = rightIsFat
                                            ? makeBoardWithFatMoves(rightRoot, rightBoard.memory)
                                            : makeBoardWithMoves(rightRoot, rightBoard.memory);

                if (temp1 == temp2) {
                    outPaths.insert(buildRecoveredPathString<REVERSE_OUTPUT>(
                            leftRoot,
                            leftBoard.memory,
                            rightRoot,
                            rightBoard.memory
                            ));
                }
            }
        };
        forEachRecoveryPath<DEPTH>(smallRoot, 0ULL, visit);
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
            const RightFrontierTableB1B2& rightIndex;
            JVec<B1B2>& allMatches;
            JVec<u64>& allMatchHashes;
            StreamProbeMetrics& metrics;

            JVec<B1B2> chunkMatches;
            JVec<u64> chunkMatchHashes;

            MU Sink(const RightFrontierTableB1B2& rightIndexIn,
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

                    RightFrontierTableB1B2::ProbeStats localProbeStats{};

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
                    metrics.probeStats.extraFilterRejects += localProbeStats.extraFilterRejects;
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
                Timer timerStream;
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_FRONTIER_DEPTH>(
                        seedBoard,
                        sink,
                        LEFT_STREAM_CHUNK_SIZE
                );
                metrics.streamWallSeconds = timerStream.getSeconds();
                metrics.permStreamSeconds = std::max(
                        0.0,
                        metrics.streamWallSeconds - metrics.probeSeconds - metrics.appendSeconds
                );
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

    template<int TOTAL_LEFT_DEPTH, int RIGHT_FRONTIER_DEPTH, bool REVERSE_SEARCH, bool debug>
    MU void recoverMatchedMiddlesGlobally(
            JVec<B1B2>& middleMatches,
            JVec<u64>& middleMatchHashes,
            GlobalRecoveryMetrics& metrics,
            const int worker_count) {
        static_assert(TOTAL_LEFT_DEPTH >= 0 && TOTAL_LEFT_DEPTH <= 10);
        // Recovery splits each side in two halves of at most 5 moves.
        static_assert(RIGHT_FRONTIER_DEPTH >= 0 && RIGHT_FRONTIER_DEPTH <= 10);

        metrics = {};
        if (middleMatches.empty()) {
            return;
        }

        {
            Timer timer;
            sortStatesByHash(middleMatches, middleMatchHashes);
            metrics.middleSortSeconds = timer.getSeconds();
        }
        {
            Timer timer;
            compactUniqueSortedStatesInPlace(middleMatches, middleMatchHashes);
            metrics.middleDedupeSeconds = timer.getSeconds();
        }
        metrics.uniqueMiddleCount = middleMatches.size();

        Timer pathRecoveryTimer;

        // The larger half of each split is built once and shared read-only by
        // every middle; each middle only builds the smaller half.
        static constexpr u32 START_SHARED_DEPTH = (TOTAL_LEFT_DEPTH + 1) / 2;
        static constexpr u32 START_PER_MIDDLE_DEPTH = TOTAL_LEFT_DEPTH - START_SHARED_DEPTH;
        static constexpr u32 GOAL_PER_MIDDLE_DEPTH = RIGHT_FRONTIER_DEPTH / 2;
        static constexpr u32 GOAL_SHARED_DEPTH = RIGHT_FRONTIER_DEPTH - GOAL_PER_MIDDLE_DEPTH;

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        RecoveryBoardFrontierCache startSharedCache;
        ensureCache(startSharedCache, searchStartRoot, START_SHARED_DEPTH);
        ensureCache(goalRightCache_, searchGoalRoot, GOAL_SHARED_DEPTH);
        const RecoveryBoardFrontierCache& goalSharedCache = goalRightCache_;

        // Each worker holds two per-middle frontiers (states + hashes + sort
        // scratch); keep their combined worst case under ~6 GB.
        const bool fat = searchStartRoot.getFatBool();
        const bool streamStart = START_PER_MIDDLE_DEPTH >= recoveryStreamMinDepth;
        const bool streamGoal = GOAL_PER_MIDDLE_DEPTH >= recoveryStreamMinDepth;
        const double perWorkerBytes =
                2.0 * static_cast<double>(sizeof(Board) + sizeof(u64))
                * static_cast<double>((streamStart ? 0 : recoveryPathUpperBound(fat, START_PER_MIDDLE_DEPTH))
                                      + (streamGoal ? 0 : recoveryPathUpperBound(fat, GOAL_PER_MIDDLE_DEPTH)));
        const std::size_t memoryWorkers = std::max<std::size_t>(
                1, static_cast<std::size_t>(6.0e9 / std::max(1.0, perWorkerBytes)));
        const std::size_t workerCount = std::min<std::size_t>(
                {static_cast<std::size_t>(std::max(1, worker_count)), memoryWorkers, middleMatches.size()});

        if constexpr (debug) {
            tcout << "global exact recovery: " << middleMatches.size()
                  << " unique middle states, " << workerCount << " workers\n";
        }

        std::atomic<std::size_t> nextMiddle = 0;
        std::vector<std::vector<std::string>> workerSolutions(workerCount);

        auto recoverWorker = [&](const std::size_t workerId) {
            RecoveryBoardFrontierCache middleFromStartCache;
            RecoveryBoardFrontierCache middleToGoalCache;
            std::set<std::string> rootToMiddlePaths;
            std::set<std::string> middleToGoalPaths;
            std::vector<std::string>& solutions = workerSolutions[workerId];

            for (std::size_t i; (i = nextMiddle.fetch_add(1)) < middleMatches.size();) {
                const Board middleBoard = makeBoardFromState(middleMatches[i]);

                if (streamStart) {
                    recoverSplitBySharedLookupStreamed<REVERSE_SEARCH, false, START_PER_MIDDLE_DEPTH>(
                            searchStartRoot, middleBoard, startSharedCache, middleBoard, rootToMiddlePaths);
                } else {
                    middleFromStartCache.valid = false;
                    ensureCache(middleFromStartCache, middleBoard, START_PER_MIDDLE_DEPTH);
                    recoverSplitBySharedLookup<REVERSE_SEARCH, false>(
                            searchStartRoot, middleBoard,
                            startSharedCache, middleFromStartCache,
                            rootToMiddlePaths);
                }
                if (rootToMiddlePaths.empty()) {
                    continue;
                }

                if (streamGoal) {
                    recoverSplitBySharedLookupStreamed<REVERSE_SEARCH, true, GOAL_PER_MIDDLE_DEPTH>(
                            middleBoard, searchGoalRoot, goalSharedCache, middleBoard, middleToGoalPaths);
                } else {
                    middleToGoalCache.valid = false;
                    ensureCache(middleToGoalCache, middleBoard, GOAL_PER_MIDDLE_DEPTH);
                    recoverSplitBySharedLookup<REVERSE_SEARCH, true>(
                            middleBoard, searchGoalRoot,
                            goalSharedCache, middleToGoalCache,
                            middleToGoalPaths);
                }
                if (middleToGoalPaths.empty()) {
                    continue;
                }

                // In reverse search the "start" side is the true goal, so the
                // middle->goal half comes first in the forward move string.
                const auto& firstHalf = REVERSE_SEARCH ? middleToGoalPaths : rootToMiddlePaths;
                const auto& secondHalf = REVERSE_SEARCH ? rootToMiddlePaths : middleToGoalPaths;
                for (const auto& first : firstHalf) {
                    for (const auto& second : secondHalf) {
                        if (first.empty()) {
                            solutions.push_back(second);
                        } else if (second.empty()) {
                            solutions.push_back(first);
                        } else {
                            solutions.push_back(first + " " + second);
                        }
                    }
                }
            }
        };

        std::vector<std::thread> recoveryThreads;
        recoveryThreads.reserve(workerCount);
        for (std::size_t w = 0; w < workerCount; ++w) {
            recoveryThreads.emplace_back(recoverWorker, w);
        }
        for (auto& thread : recoveryThreads) {
            thread.join();
        }
        for (auto& solutions : workerSolutions) {
            for (auto& solution : solutions) {
                resultSet.insert(std::move(solution));
            }
        }

        metrics.pathRecoverySeconds = pathRecoveryTimer.getSeconds();

        if constexpr (debug) {
            tcout << "global exact recovery raw solutions: "
                  << resultSet.size() << '\n';
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
        JVec<B1B2> rightFrontierStates;
        JVec<u64> rightFrontierHashes;

        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(
                searchGoalRoot,
                rightFrontierStates,
                rightFrontierHashes,
                debug
        );

        tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH
              << ") exact unique size: " << rightFrontierStates.size() << '\n';

        rightFrontierIndex_.template buildFromUniqueStates<debug>(
                std::move(rightFrontierStates),
                std::move(rightFrontierHashes),
                searchGoalRoot
        );

        tcout << "right frontier exact index built for "
              << rightFrontierIndex_.size()
              << " states across "
              << rightFrontierIndex_.rangeCount()
              << " hash buckets\n";
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
    MU int findSolutionsFrontierThreadedImpl(int worker_count) {
        static constexpr int TOTAL_LEFT_DEPTH = SEED_DEPTH + LEFT_FRONTIER_DEPTH;
        static constexpr int TOTAL_DEPTH = TOTAL_LEFT_DEPTH + RIGHT_FRONTIER_DEPTH;

        resultSet.clear();
        expandedResultSet.clear();

        if (worker_count < 1) {
            worker_count = 1;
        }

        goalRightCache_.valid = false;
        rightFrontierIndex_.clear();

        Board& searchStartRoot = getSearchStartBoard<REVERSE_SEARCH>();
        Board& searchGoalRoot = getSearchGoalBoard<REVERSE_SEARCH>();

        const Timer totalTime;

        JVec<B1B2> leftSeeds;
        JVec<u64> leftSeedHashes;
        double leftFrontierGenSeconds = 0.0;

        if constexpr (debug) {
            tcout << "#####################################\n";
            tcout << "#        Building LEFT_SEED         #\n";
            tcout << "#####################################\n\n";
        }

        {
            Timer timer;
            buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(
                    searchStartRoot,
                    leftSeeds,
                    leftSeedHashes,
                    debug
            );

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
                if (a.hash != b.hash) {
                    return a.hash < b.hash;
                }
                return a.seed < b.seed;
            });

            for (std::size_t i = 0; i < scoredSeeds.size(); ++i) {
                leftSeeds[i] = scoredSeeds[i].seed;
                leftSeedHashes[i] = scoredSeeds[i].hash;
            }

            leftFrontierGenSeconds = timer.getSeconds();
        }

        const std::size_t seedTaskCount = leftSeeds.size();
        tcout << "seed(" << SEED_DEPTH << ") exact unique task count: "
              << seedTaskCount << '\n';

        if constexpr (debug) {
            tcout << "\n#####################################\n";
            tcout << "#      Building RIGHT_FRONTIER      #\n";
            tcout << "#####################################\n\n";
        }

        double rightFrontierGenSeconds = 0.0;
        {
            Timer timer;
            prepareRightFrontierIndex<RIGHT_FRONTIER_DEPTH, debug>(searchGoalRoot);
            rightFrontierGenSeconds = timer.getSeconds();
        }

        if constexpr (debug) {
            tcout << '\n';
            rightFrontierIndex_.printStats();
            tcout << '\n';
        }

        std::atomic<std::size_t> nextIndex = 0;
        std::mutex printMutex;
        std::mutex middleMutex;

        JVec<B1B2> allMiddleMatches;
        JVec<u64> allMiddleMatchHashes;
        std::vector<StreamProbeMetrics> workerTotals(
                static_cast<std::size_t>(worker_count));

        auto addMetrics = [](StreamProbeMetrics& dst, const StreamProbeMetrics& src) {
            dst.streamWallSeconds += src.streamWallSeconds;
            dst.permStreamSeconds += src.permStreamSeconds;
            dst.probeSeconds += src.probeSeconds;
            dst.appendSeconds += src.appendSeconds;
            dst.finalSortSeconds += src.finalSortSeconds;
            dst.finalDedupeSeconds += src.finalDedupeSeconds;
            dst.taskWallSeconds += src.taskWallSeconds;
            dst.chunkCount += src.chunkCount;
            dst.streamedStateCount += src.streamedStateCount;
            dst.preMergeMatchCount += src.preMergeMatchCount;

            dst.probeStats.leftStateCount += src.probeStats.leftStateCount;
            dst.probeStats.hashHits += src.probeStats.hashHits;
            dst.probeStats.hashMisses += src.probeStats.hashMisses;
            dst.probeStats.bucketsVisited += src.probeStats.bucketsVisited;
            dst.probeStats.bucketStatesScanned += src.probeStats.bucketStatesScanned;
            dst.probeStats.equalityChecks += src.probeStats.equalityChecks;
            dst.probeStats.exactMatches += src.probeStats.exactMatches;
            dst.probeStats.highFilterRejects += src.probeStats.highFilterRejects;
            dst.probeStats.midFilterRejects += src.probeStats.midFilterRejects;
            dst.probeStats.lowFilterRejects += src.probeStats.lowFilterRejects;
            dst.probeStats.extraFilterRejects += src.probeStats.extraFilterRejects;
            dst.probeStats.prefixRejects += src.probeStats.prefixRejects;
            dst.probeStats.filterPasses += src.probeStats.filterPasses;
        };

        auto worker = [&](const std::size_t workerId) {
            JVec<B1B2> middleMatches;
            JVec<u64> middleMatchHashes;

            while (true) {
                const std::size_t i = nextIndex.fetch_add(1);
                if (i >= leftSeeds.size()) {
                    break;
                }

                if constexpr (debug) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "[seed " << (i + 1) << "/" << leftSeeds.size()
                          << "] streaming exact-state seed(" << SEED_DEPTH << ") +"
                          << LEFT_FRONTIER_DEPTH << "\n" << std::flush;
                }

                Board seedBoard = makeBoardFromState(leftSeeds[i]);
                StreamProbeMetrics metrics;

                {
                    Timer taskTimer;
                    collectLeftFrontierMiddleMatchesStreamed<
                            LEFT_FRONTIER_DEPTH,
                            RIGHT_FRONTIER_DEPTH,
                            true>(
                            seedBoard,
                            middleMatches,
                            middleMatchHashes,
                            metrics
                    );
                    metrics.taskWallSeconds = taskTimer.getSeconds();
                }

                if (!middleMatches.empty()) {
                    std::lock_guard<std::mutex> lock(middleMutex);
                    appendStates(
                            allMiddleMatches,
                            allMiddleMatchHashes,
                            middleMatches,
                            middleMatchHashes
                    );
                }

                addMetrics(workerTotals[workerId], metrics);

                if constexpr (debug) {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "[worker " << workerId << "]"
                          << "\n  exact middle matches: " << middleMatches.size()
                          << "\n  streamed states: " << metrics.streamedStateCount
                          << "\n  high filter rejects: " << metrics.probeStats.highFilterRejects
                          << "\n  mid filter rejects: " << metrics.probeStats.midFilterRejects
                          << "\n  low filter rejects: " << metrics.probeStats.lowFilterRejects
                          << "\n  extra filter rejects: " << metrics.probeStats.extraFilterRejects
                          << "\n  filter passes: " << metrics.probeStats.filterPasses
                          << "\n  prefix rejects: " << metrics.probeStats.prefixRejects
                          << "\n  primary hash hits: " << metrics.probeStats.hashHits
                          << "\n  full-state equality checks: " << metrics.probeStats.equalityChecks
                          << "\n  exact matches: " << metrics.probeStats.exactMatches
                          << "\n  permstream time (exclusive): " << metrics.permStreamSeconds
                          << "\n  probe time: " << metrics.probeSeconds
                          << "\n  append/post-probe time: "
                          << (metrics.appendSeconds + metrics.finalSortSeconds + metrics.finalDedupeSeconds)
                          << '\n';
                }
            }
        };

        Timer workerWallTimer;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(worker_count));
        for (std::size_t workerId = 0;
             workerId < static_cast<std::size_t>(worker_count);
             ++workerId) {
            workers.emplace_back(worker, workerId);
        }
        for (auto& thread : workers) {
            thread.join();
        }
        const double workerWallSeconds = workerWallTimer.getSeconds();

        StreamProbeMetrics aggregate{};
        for (const auto& workerMetrics : workerTotals) {
            addMetrics(aggregate, workerMetrics);
        }

        // Probing is complete. Release the multi-gigabyte exact index before
        // constructing recovery frontiers.
        rightFrontierIndex_.clear();
        JVec<B1B2>{}.swap(leftSeeds);
        JVec<u64>{}.swap(leftSeedHashes);

        GlobalRecoveryMetrics recoveryMetrics{};
        Timer recoveryTimer;
        recoverMatchedMiddlesGlobally<
                TOTAL_LEFT_DEPTH,
                RIGHT_FRONTIER_DEPTH,
                REVERSE_SEARCH,
                debug>(
                allMiddleMatches,
                allMiddleMatchHashes,
                recoveryMetrics,
                worker_count
        );
        const double globalRecoverySeconds = recoveryTimer.getSeconds();

        double finalDedupeSeconds = 0.0;
        if (!resultSet.empty()) {
            Timer timer;
            finalizeExactSolutionsIntoFinalSet();
            finalDedupeSeconds = timer.getSeconds();
            if (shouldWriteSolutionsToFile()) {
                writeExpandedSolutions(TOTAL_DEPTH);
            }
        } else {
            tcout << "No solutions found...\n";
        }

        const double leftSeedTotalSeconds = workerWallSeconds + globalRecoverySeconds;
        const double postProbeSeconds = aggregate.appendSeconds
                                      + aggregate.finalSortSeconds
                                      + aggregate.finalDedupeSeconds;

        tcout << "\nTotal Time: " << totalTime.getSeconds() << '\n';
        tcout << "Puzzle: " << pair->getName() << '\n';
        tcout << "Total depth: " << TOTAL_DEPTH << '\n';
        tcout << "Left seed depth: " << SEED_DEPTH << '\n';
        tcout << "Left stream depth: " << LEFT_FRONTIER_DEPTH << '\n';
        tcout << "Right exact frontier depth: " << RIGHT_FRONTIER_DEPTH << '\n';
        tcout << "Search direction: " << directionNameTemplated<REVERSE_SEARCH>() << '\n';

        tcout << "\n========== TIMING SUMMARY (COPY/PASTE) ==========\n";
        tcout << "right frontier gen time: " << rightFrontierGenSeconds << "s\n";
        tcout << "left frontier gen time: " << leftFrontierGenSeconds << "s\n";
        tcout << "left seed permstream time: " << aggregate.permStreamSeconds
              << "s (aggregate CPU, exclusive of probe/append)\n";
        tcout << "left seed probe time: " << aggregate.probeSeconds
              << "s (aggregate CPU, exact full-state lookup)\n";
        tcout << "left seed total time: " << leftSeedTotalSeconds
              << "s (worker wall time + global exact recovery)\n";
        tcout << "left seed worker time: " << workerWallSeconds
              << "s (thread launch through final worker join)\n";
        tcout << "de-dupe at end time: " << finalDedupeSeconds
              << "s (validation and final exact set ordering)\n";
        tcout << "left seed post-probe time: " << postProbeSeconds
              << "s (aggregate CPU; append + per-task middle sort/de-dupe)\n";
        tcout << "left seed aggregate task time: " << aggregate.taskWallSeconds
              << "s (aggregate CPU across all seed tasks)\n";
        tcout << "global recovery time: " << globalRecoverySeconds << "s\n";
        tcout << "global middle sort time: " << recoveryMetrics.middleSortSeconds << "s\n";
        tcout << "global middle de-dupe time: " << recoveryMetrics.middleDedupeSeconds << "s\n";
        tcout << "global path recovery time: " << recoveryMetrics.pathRecoverySeconds << "s\n";
        tcout << "left seed tasks: " << seedTaskCount << '\n';
        tcout << "left streamed states: " << aggregate.streamedStateCount << '\n';
        tcout << "left primary hash hits: " << aggregate.probeStats.hashHits << '\n';
        tcout << "left full-state equality checks: " << aggregate.probeStats.equalityChecks << '\n';
        tcout << "left exact candidate matches: " << aggregate.probeStats.exactMatches << '\n';
        tcout << "unique exact middle states: " << recoveryMetrics.uniqueMiddleCount << '\n';
        tcout << "=================================================\n";

        return getExpandedSolutionCount();
    }

public:
    using BoardSolverBase::BoardSolverBase;

    // Turns middle states found by an external search (exactly TOTAL_LEFT_DEPTH
    // moves from the search start, RIGHT_DEPTH moves from the search goal) into
    // the full validated solution set, written like findSolutions' own output.
    template<int TOTAL_LEFT_DEPTH, int RIGHT_DEPTH, bool REVERSE_SEARCH>
    MU int recoverSolutionsFromMiddles(const std::vector<B1B2>& middles, const int worker_count) {
        resultSet.clear();
        expandedResultSet.clear();
        goalRightCache_.valid = false;

        JVec<B1B2> middleStates;
        JVec<u64> middleHashes;
        middleStates.resize(middles.size());
        middleHashes.resize(middles.size());
        for (std::size_t i = 0; i < middles.size(); ++i) {
            middleStates[i] = middles[i];
            middleHashes[i] = StateHash::computeHash(middles[i]);
        }

        GlobalRecoveryMetrics metrics{};
        recoverMatchedMiddlesGlobally<TOTAL_LEFT_DEPTH, RIGHT_DEPTH, REVERSE_SEARCH, false>(
                middleStates, middleHashes, metrics, worker_count);

        if (resultSet.empty()) {
            tcout << "No solutions found...\n";
            return 0;
        }
        finalizeExactSolutionsIntoFinalSet();
        if (shouldWriteSolutionsToFile()) {
            writeExpandedSolutions(TOTAL_LEFT_DEPTH + RIGHT_DEPTH);
        }
        return getExpandedSolutionCount();
    }

    template<
            int SEED_DEPTH = 1,
            int LEFT_FRONTIER_DEPTH = 5,
            int RIGHT_FRONTIER_DEPTH = 5,
            bool debug = true
            >
    MU int findSolutionsFrontier(SearchDirection direction = SearchDirection::Auto) {
        return findSolutionsFrontierThreaded<
                SEED_DEPTH,
                LEFT_FRONTIER_DEPTH,
                RIGHT_FRONTIER_DEPTH,
                debug>(1, direction);
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

        // Kept for source compatibility with the training-data driver.  The
        // former copied-index cache was removed because copying a multi-GB
        // exact index is both wasteful and unsafe under memory pressure.
        (void) enableDepth5RightCache;

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