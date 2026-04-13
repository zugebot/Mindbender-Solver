#pragma once
// code/solver/frontier_builder.hpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "utils/format_bytes.hpp"
#include "utils/timestamped_cout.hpp"
#include "utils/timer.hpp"

#include "code/board.hpp"
#include "code/perms.hpp"
#include "code/solver/sorter.hpp"

MU static Board makeBoardFromState(const B1B2& state) {
    Board out;
    out.b1 = state.b1;
    out.b2 = state.b2;
    return out;
}

namespace frontier_recovery_detail {

    static constexpr std::size_t NORMAL_NONE_MOVE_COUNT =
            static_cast<std::size_t>(NORMAL_ROW_MOVE_COUNT + NORMAL_COL_MOVE_COUNT);

    static constexpr std::size_t FAT_NONE_MOVE_COUNT = 48;

    struct FrontierChunkResult {
        JVec<B1B2> states;
        JVec<u64> hashes;
    };

    MUND FORCEINLINE std::size_t chooseExpandThreadCount(const std::size_t frontierSize) {
        if (frontierSize < 4096) {
            return 1;
        }

        std::size_t hw = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (hw == 0) {
            hw = 1;
        }

        std::size_t threadCount = std::max<std::size_t>(1, hw / 2);
        if (threadCount > frontierSize) {
            threadCount = frontierSize;
        }
        if (threadCount == 0) {
            threadCount = 1;
        }

        return threadCount;
    }

    MU static void reserveStateHashLanes(JVec<B1B2>& states,
                                         JVec<u64>& hashes,
                                         const std::size_t capacity) {
        if (capacity == 0) {
            return;
        }

        if (states.capacity() < capacity) {
            states.reserve(capacity);
        }
        if (hashes.capacity() < capacity) {
            hashes.reserve(capacity);
        }
    }

    MU static void ensureWritableTail(JVec<B1B2>& states,
                                      JVec<u64>& hashes,
                                      const std::size_t requiredSize) {
        if (requiredSize <= states.size()) {
            return;
        }

        std::size_t newSize = states.size();
        if (newSize == 0) {
            newSize = 64;
        }

        while (newSize < requiredSize) {
            if (newSize > (std::numeric_limits<std::size_t>::max() / 2)) {
                newSize = requiredSize;
                break;
            }
            newSize *= 2;
        }

        reserveStateHashLanes(states, hashes, newSize);
        states.resize(newSize);
        hashes.resize(newSize);
    }

    MUND FORCEINLINE std::size_t emitNormalNoneChildren(
            const B1B2& parent,
            B1B2* dstStates,
            u64* dstHashes) {
        std::size_t produced = 0;

        for (u64 act = 0; act < NORMAL_ROW_MOVE_COUNT; ++act) {
            B1B2 child = parent;
            allActStructList[act].action(child);

            if (child == parent) {
                continue;
            }

            dstStates[produced] = child;
            dstHashes[produced] = StateHash::computeHash(child);
            ++produced;
        }

        for (u64 act = NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT;
             act < NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT + NORMAL_COL_MOVE_COUNT;
             ++act) {
            B1B2 child = parent;
            allActStructList[act].action(child);

            if (child == parent) {
                continue;
            }

            dstStates[produced] = child;
            dstHashes[produced] = StateHash::computeHash(child);
            ++produced;
        }

        return produced;
    }

    MUND FORCEINLINE std::size_t emitFatNoneChildren(
            const B1B2& parent,
            B1B2* dstStates,
            u64* dstHashes) {
        std::size_t produced = 0;

        const u8* funcIndexes = fatActionsIndexes[parent.getFatXY()];

        for (u8 actn_i = 0; actn_i < FAT_NONE_MOVE_COUNT; ++actn_i) {
            B1B2 child = parent;
            allActStructList[funcIndexes[actn_i]].action(child);

            if (child == parent) {
                continue;
            }

            dstStates[produced] = child;
            dstHashes[produced] = StateHash::computeHash(child);
            ++produced;
        }

        return produced;
    }

    MUND FORCEINLINE std::size_t emitNoneChildren(
            const B1B2& parent,
            const bool isFatPuzzle,
            B1B2* dstStates,
            u64* dstHashes) {
        if (isFatPuzzle) {
            return emitFatNoneChildren(parent, dstStates, dstHashes);
        }
        return emitNormalNoneChildren(parent, dstStates, dstHashes);
    }

    MU static void expandNoneFrontierRangeByOne(
            const JVec<B1B2>& frontierStates,
            const std::size_t beginIndex,
            const std::size_t endIndex,
            const bool isFatPuzzle,
            JVec<B1B2>& outStates,
            JVec<u64>& outHashes,
            const u64 reserveGuessPerThread) {
        outStates.clear();
        outHashes.clear();

        if (beginIndex >= endIndex) {
            return;
        }

        const std::size_t branchCap = isFatPuzzle ? FAT_NONE_MOVE_COUNT : NORMAL_NONE_MOVE_COUNT;
        const std::size_t parentCount = endIndex - beginIndex;
        const std::size_t hardUpper = parentCount * branchCap;

        if (reserveGuessPerThread != 0) {
            reserveStateHashLanes(
                    outStates,
                    outHashes,
                    std::max<std::size_t>(
                            static_cast<std::size_t>(reserveGuessPerThread),
                            hardUpper
                            )
            );
        } else {
            reserveStateHashLanes(outStates, outHashes, hardUpper);
        }

        outStates.resize(hardUpper);
        outHashes.resize(hardUpper);

        std::size_t producedTotal = 0;
        for (std::size_t i = beginIndex; i < endIndex; ++i) {
            producedTotal += emitNoneChildren(
                    frontierStates[i],
                    isFatPuzzle,
                    &outStates[producedTotal],
                    &outHashes[producedTotal]
            );
        }

        outStates.resize(producedTotal);
        outHashes.resize(producedTotal);
    }

    MU static void copyLaneRangeIntoOffset(
            JVec<B1B2>& dstStates,
            JVec<u64>& dstHashes,
            const std::size_t dstOffset,
            const JVec<B1B2>& srcStates,
            const JVec<u64>& srcHashes) {
        if (srcStates.empty()) {
            return;
        }

        for (std::size_t i = 0; i < srcStates.size(); ++i) {
            dstStates[dstOffset + i] = srcStates[i];
            dstHashes[dstOffset + i] = srcHashes[i];
        }
    }

    MU static void expandNoneFrontierByOne(
            const JVec<B1B2>& frontierStates,
            const bool isFatPuzzle,
            JVec<B1B2>& nextStates,
            JVec<u64>& nextHashes,
            const u64 reserveGuess) {
        nextStates.clear();
        nextHashes.clear();

        if (frontierStates.empty()) {
            return;
        }

        const std::size_t threadCount = chooseExpandThreadCount(frontierStates.size());

        if (threadCount <= 1) {
            expandNoneFrontierRangeByOne(
                    frontierStates,
                    0,
                    frontierStates.size(),
                    isFatPuzzle,
                    nextStates,
                    nextHashes,
                    reserveGuess
            );
            return;
        }

        std::vector<FrontierChunkResult> partials(threadCount);
        std::vector<std::thread> workers;
        workers.reserve(threadCount);

        const std::size_t baseChunk = frontierStates.size() / threadCount;
        const std::size_t remainder = frontierStates.size() % threadCount;

        u64 reserveGuessPerThread = reserveGuess == 0
                                            ? 0
                                            : (reserveGuess / static_cast<u64>(threadCount));

        const u64 minReservePerThread = isFatPuzzle
                                                ? static_cast<u64>(FAT_NONE_MOVE_COUNT)
                                                : static_cast<u64>(NORMAL_NONE_MOVE_COUNT);

        if (reserveGuessPerThread < minReservePerThread) {
            reserveGuessPerThread = minReservePerThread;
        }

        std::size_t begin = 0;
        for (std::size_t t = 0; t < threadCount; ++t) {
            const std::size_t chunkLen = baseChunk + (t < remainder ? 1 : 0);
            const std::size_t end = begin + chunkLen;

            workers.emplace_back([&, t, begin, end, isFatPuzzle, reserveGuessPerThread]() {
                expandNoneFrontierRangeByOne(
                        frontierStates,
                        begin,
                        end,
                        isFatPuzzle,
                        partials[t].states,
                        partials[t].hashes,
                        reserveGuessPerThread
                );
            });

            begin = end;
        }

        for (auto& worker : workers) {
            worker.join();
        }

        std::size_t totalSize = 0;
        for (const auto& partial : partials) {
            totalSize += partial.states.size();
        }

        reserveStateHashLanes(
                nextStates,
                nextHashes,
                std::max<std::size_t>(
                        static_cast<std::size_t>(reserveGuess),
                        totalSize
                        )
        );

        nextStates.resize(totalSize);
        nextHashes.resize(totalSize);

        std::size_t writeOffset = 0;
        for (const auto& partial : partials) {
            copyLaneRangeIntoOffset(
                    nextStates,
                    nextHashes,
                    writeOffset,
                    partial.states,
                    partial.hashes
            );
            writeOffset += partial.states.size();
        }
    }

    template<typename T>
    MUND FORCEINLINE bool lessByHashThenState(const T& lhsState,
                                              const u64 lhsHash,
                                              const T& rhsState,
                                              const u64 rhsHash) {
        if (lhsHash < rhsHash) {
            return true;
        }
        if (rhsHash < lhsHash) {
            return false;
        }
        return lhsState < rhsState;
    }

    template<typename T>
    MUND FORCEINLINE bool equalByHashAndState(const T& lhsState,
                                              const u64 lhsHash,
                                              const T& rhsState,
                                              const u64 rhsHash) {
        return lhsHash == rhsHash && lhsState == rhsState;
    }

    template<typename T>
    MU static void normalizeBucketsByState(JVec<T>& states,
                                           const JVec<u64>& hashes) {
        if (states.size() <= 1) {
            return;
        }

        std::size_t begin = 0;
        while (begin < states.size()) {
            std::size_t end = begin + 1;
            const u64 hash = hashes[begin];

            while (end < states.size() && hashes[end] == hash) {
                ++end;
            }

            if (end - begin > 1) {
                std::sort(states.begin() + begin, states.begin() + end);
            }

            begin = end;
        }
    }
}

template<typename T>
MU static void sortStatesByHash(JVec<T>& states,
                                JVec<u64>& hashes) {
    if (states.empty()) {
        hashes.clear();
        return;
    }

    std::sort(states.begin(), states.end(), [&](const T& a, const T& b) {
        const u64 ha = StateHash::computeHash(a);
        const u64 hb = StateHash::computeHash(b);

        if (ha < hb) {
            return true;
        }
        if (hb < ha) {
            return false;
        }
        return a < b;
    });

    hashes.resize(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        hashes[i] = StateHash::computeHash(states[i]);
    }

    frontier_recovery_detail::normalizeBucketsByState(states, hashes);
}

template<typename T>
MU static void compactUniqueSortedStatesInPlace(JVec<T>& states,
                                                JVec<u64>& hashes) {
    if (states.empty()) {
        hashes.clear();
        return;
    }

    std::size_t writeIndex = 1;

    for (std::size_t readIndex = 1; readIndex < states.size(); ++readIndex) {
        if (!frontier_recovery_detail::equalByHashAndState(
                    states[readIndex], hashes[readIndex],
                    states[writeIndex - 1], hashes[writeIndex - 1])) {
            states[writeIndex] = states[readIndex];
            hashes[writeIndex] = hashes[readIndex];
            ++writeIndex;
        }
    }

    states.resize(writeIndex);
    hashes.resize(writeIndex);
}

template<typename T>
MU static void removeStatesPresentInSortedSetLinear(JVec<T>& states,
                                                    JVec<u64>& hashes,
                                                    const JVec<T>& sortedSeenStates,
                                                    const JVec<u64>& sortedSeenHashes) {
    if (states.empty() || sortedSeenStates.empty()) {
        return;
    }

    std::size_t writeIndex = 0;
    std::size_t seenIndex = 0;

    for (std::size_t i = 0; i < states.size(); ++i) {
        const u64 curHash = hashes[i];

        while (seenIndex < sortedSeenStates.size() && sortedSeenHashes[seenIndex] < curHash) {
            ++seenIndex;
        }

        bool found = false;
        std::size_t probe = seenIndex;

        while (probe < sortedSeenStates.size() && sortedSeenHashes[probe] == curHash) {
            if (sortedSeenStates[probe] == states[i]) {
                found = true;
                break;
            }
            ++probe;
        }

        if (!found) {
            states[writeIndex] = states[i];
            hashes[writeIndex] = hashes[i];
            ++writeIndex;
        }
    }

    states.resize(writeIndex);
    hashes.resize(writeIndex);
}

template<typename T>
MU static void mergeSortedUniqueStatesIntoSeen(JVec<T>& seenStates,
                                               JVec<u64>& seenHashes,
                                               const JVec<T>& newStates,
                                               const JVec<u64>& newHashes,
                                               JVec<T>& scratchStates,
                                               JVec<u64>& scratchHashes) {
    if (newStates.empty()) {
        return;
    }

    scratchStates.resize(seenStates.size() + newStates.size());
    scratchHashes.resize(seenHashes.size() + newHashes.size());

    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t out = 0;

    while (i < seenStates.size() && j < newStates.size()) {
        if (frontier_recovery_detail::lessByHashThenState(
                    seenStates[i], seenHashes[i],
                    newStates[j], newHashes[j])) {
            scratchStates[out] = seenStates[i];
            scratchHashes[out] = seenHashes[i];
            ++i;
            ++out;
            continue;
        }

        if (frontier_recovery_detail::lessByHashThenState(
                    newStates[j], newHashes[j],
                    seenStates[i], seenHashes[i])) {
            scratchStates[out] = newStates[j];
            scratchHashes[out] = newHashes[j];
            ++j;
            ++out;
            continue;
        }

        scratchStates[out] = seenStates[i];
        scratchHashes[out] = seenHashes[i];
        ++i;
        ++j;
        ++out;
    }

    while (i < seenStates.size()) {
        scratchStates[out] = seenStates[i];
        scratchHashes[out] = seenHashes[i];
        ++i;
        ++out;
    }

    while (j < newStates.size()) {
        scratchStates[out] = newStates[j];
        scratchHashes[out] = newHashes[j];
        ++j;
        ++out;
    }

    scratchStates.resize(out);
    scratchHashes.resize(out);

    seenStates.swap(scratchStates);
    seenHashes.swap(scratchHashes);

    scratchStates.clear();
    scratchHashes.clear();
}

class FrontierBuilderB1B2 {
    static constexpr u64 NORMAL_BRANCH_CAP = 60ULL;
    static constexpr u64 FAT_BRANCH_CAP = 48ULL;
    static constexpr u64 STATE_PAIR_BYTES = sizeof(B1B2) + sizeof(u64);

    struct LayerStats {
        u64 frontierSize = 0;
        u64 rawGenerated = 0;
        u64 afterSelfDedupe = 0;
        u64 afterSeenSubtract = 0;
        std::size_t expandThreads = 1;
    };

    Board root_{};
    BoardSorter<B1B2> sorter_{};

    JVec<B1B2> seen_{};
    JVec<u64> seenHashes_{};

    JVec<B1B2> frontier_{};
    JVec<u64> frontierHashes_{};

    JVec<B1B2> next_{};
    JVec<u64> nextHashes_{};

    JVec<B1B2> mergeScratch_{};
    JVec<u64> mergeScratchHashes_{};

    u32 colorCount_ = 0;
    bool verbose_ = true;
    LayerStats lastStats_{};
    u64 peakWorkspaceCapacityBytes_ = 0;

    template<typename T>
    MUND static u64 laneLiveBytes(const JVec<T>& lane) {
        return static_cast<u64>(lane.size()) * static_cast<u64>(sizeof(T));
    }

    template<typename T>
    MUND static u64 laneCapacityBytes(const JVec<T>& lane) {
        return static_cast<u64>(lane.capacity()) * static_cast<u64>(sizeof(T));
    }

    MUND static u64 pairLiveBytes(const JVec<B1B2>& states,
                                  const JVec<u64>& hashes) {
        return laneLiveBytes(states) + laneLiveBytes(hashes);
    }

    MUND static u64 pairCapacityBytes(const JVec<B1B2>& states,
                                      const JVec<u64>& hashes) {
        return laneCapacityBytes(states) + laneCapacityBytes(hashes);
    }

    MUND static std::string fmtBytes(const u64 bytes) {
        return bytesFormatted<1000>(bytes);
    }

    MUND static double pct(const u64 num, const u64 den) {
        if (den == 0) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(num) / static_cast<double>(den);
    }

    MUND u64 workspaceLiveBytes() const {
        return pairLiveBytes(seen_, seenHashes_)
               + pairLiveBytes(frontier_, frontierHashes_)
               + pairLiveBytes(next_, nextHashes_)
               + pairLiveBytes(mergeScratch_, mergeScratchHashes_);
    }

    MUND u64 workspaceCapacityBytes() const {
        return pairCapacityBytes(seen_, seenHashes_)
               + pairCapacityBytes(frontier_, frontierHashes_)
               + pairCapacityBytes(next_, nextHashes_)
               + pairCapacityBytes(mergeScratch_, mergeScratchHashes_);
    }

    MU void updatePeakWorkspaceCapacity() {
        const u64 cur = workspaceCapacityBytes();
        if (cur > peakWorkspaceCapacityBytes_) {
            peakWorkspaceCapacityBytes_ = cur;
        }
    }

    MU void printLaneStats(const char* label,
                           const JVec<B1B2>& states,
                           const JVec<u64>& hashes) const {
        if (!verbose_) {
            return;
        }
        tcout << "    " << label
              << ": size=" << states.size()
              << ", cap=" << states.capacity()
              << ", live=" << fmtBytes(pairLiveBytes(states, hashes))
              << ", rsv=" << fmtBytes(pairCapacityBytes(states, hashes))
              << '\n';
    }

    MUND bool isFatPuzzle() const {
        return root_.getFatBool();
    }

    MUND u64 getBranchCap() const {
        return isFatPuzzle() ? FAT_BRANCH_CAP : NORMAL_BRANCH_CAP;
    }

    MUND static u64 mulSaturating(const u64 a, const u64 b) {
        if (a == 0 || b == 0) {
            return 0;
        }
        if (a > (std::numeric_limits<u64>::max() / b)) {
            return std::numeric_limits<u64>::max();
        }
        return a * b;
    }

    MUND u64 estimateNextReserve() const {
        const u64 hardUpper = mulSaturating(static_cast<u64>(frontier_.size()), getBranchCap());

        if (hardUpper == 0) {
            return 0;
        }

        if (lastStats_.rawGenerated == 0 || lastStats_.afterSeenSubtract == 0) {
            return hardUpper;
        }

        const double keepRatio =
                static_cast<double>(lastStats_.afterSeenSubtract) /
                static_cast<double>(lastStats_.rawGenerated);

        const double paddedRatio = std::clamp(keepRatio * 1.25, 0.10, 1.0);

        u64 estimate = static_cast<u64>(static_cast<double>(hardUpper) * paddedRatio + 64.0);

        if (estimate < frontier_.size()) {
            estimate = frontier_.size();
        }
        if (estimate > hardUpper) {
            estimate = hardUpper;
        }

        return estimate;
    }

    MU void initRootState() {
        seen_.clear();
        seenHashes_.clear();
        frontier_.clear();
        frontierHashes_.clear();
        next_.clear();
        nextHashes_.clear();
        mergeScratch_.clear();
        mergeScratchHashes_.clear();

        seen_.resize(1);
        seenHashes_.resize(1);
        frontier_.resize(1);
        frontierHashes_.resize(1);

        const B1B2 rootState = root_.asB1B2();
        const u64 rootHash = StateHash::computeHash(rootState);

        seen_[0] = rootState;
        seenHashes_[0] = rootHash;

        frontier_[0] = rootState;
        frontierHashes_[0] = rootHash;

        lastStats_ = {};
        lastStats_.frontierSize = 1;
        lastStats_.expandThreads = 1;

        peakWorkspaceCapacityBytes_ = 0;
        updatePeakWorkspaceCapacity();
    }

    MU void expandOneDepth(const u32 depth) {
        Timer stepTimer;
        if (verbose_) {
            tcout << "Generating NONE depth " << depth << "...\n" << std::flush;
        }

        next_.clear();
        nextHashes_.clear();

        const bool fatPuzzle = isFatPuzzle();
        const u64 reserveGuess = estimateNextReserve();
        frontier_recovery_detail::reserveStateHashLanes(
                next_,
                nextHashes_,
                static_cast<std::size_t>(reserveGuess)
        );

        updatePeakWorkspaceCapacity();

        lastStats_ = {};
        lastStats_.frontierSize = frontier_.size();
        lastStats_.expandThreads = frontier_recovery_detail::chooseExpandThreadCount(frontier_.size());

        if (verbose_) {
            tcout << "    frontier in: " << frontier_.size()
                  << " states, " << fmtBytes(pairLiveBytes(frontier_, frontierHashes_))
                  << " live, " << fmtBytes(pairCapacityBytes(frontier_, frontierHashes_))
                  << " rsv\n";

            tcout << "    seen so far: " << seen_.size()
                  << " states, " << fmtBytes(pairLiveBytes(seen_, seenHashes_))
                  << " live, " << fmtBytes(pairCapacityBytes(seen_, seenHashes_))
                  << " rsv\n";
        }

        const u64 hardUpper = static_cast<u64>(frontier_.size()) * getBranchCap();

        if (verbose_) {
            tcout << "    expand threads: " << lastStats_.expandThreads << '\n';

            tcout << "    reserve guess: " << reserveGuess
                  << " states, " << fmtBytes(reserveGuess * STATE_PAIR_BYTES) << '\n';

            tcout << "    hard upper bound: " << hardUpper
                  << " states, " << fmtBytes(hardUpper * STATE_PAIR_BYTES) << '\n';
        }

        frontier_recovery_detail::expandNoneFrontierByOne(
                frontier_,
                fatPuzzle,
                next_,
                nextHashes_,
                reserveGuess
        );

        lastStats_.rawGenerated = next_.size();

        updatePeakWorkspaceCapacity();

        if (verbose_) {
            tcout << "    raw size: " << next_.size()
                  << " states, " << fmtBytes(pairLiveBytes(next_, nextHashes_))
                  << " live, " << fmtBytes(pairCapacityBytes(next_, nextHashes_))
                  << " reserved\n";

            tcout << "    reserve utilization: "
                  << pct(static_cast<u64>(next_.size()), reserveGuess) << "%\n";

            tcout << "    avg branching: "
                  << (frontier_.empty()
                              ? 0.0
                              : static_cast<double>(lastStats_.rawGenerated) / static_cast<double>(frontier_.size()))
                  << '\n';
        }

        {
            Timer timerSort;
            sorter_.sortBoards(next_, nextHashes_, depth, colorCount_);
            frontier_recovery_detail::normalizeBucketsByState(next_, nextHashes_);
            if (verbose_) {
                tcout << "    sort time: " << timerSort.getSeconds() << '\n';
            }
        }

        {
            Timer timerDedupe;
            compactUniqueSortedStatesInPlace(next_, nextHashes_);
            lastStats_.afterSelfDedupe = next_.size();

            if (verbose_) {
                tcout << "    after self dedupe: " << next_.size()
                      << " states (" << pct(lastStats_.afterSelfDedupe, lastStats_.rawGenerated) << "% of raw)\n";

                tcout << "    self dedupe time: " << timerDedupe.getSeconds() << '\n';
            }
        }

        {
            Timer timerSubtract;
            removeStatesPresentInSortedSetLinear(
                    next_,
                    nextHashes_,
                    seen_,
                    seenHashes_
            );
            lastStats_.afterSeenSubtract = next_.size();

            if (verbose_) {
                tcout << "    after seen subtract: " << next_.size()
                      << " states (" << pct(lastStats_.afterSeenSubtract, lastStats_.rawGenerated) << "% of raw, "
                      << pct(lastStats_.afterSeenSubtract, lastStats_.afterSelfDedupe) << "% of self-deduped)\n";

                tcout << "    subtract seen time: " << timerSubtract.getSeconds() << '\n';
            }
        }

        {
            Timer timerMerge;
            mergeSortedUniqueStatesIntoSeen(
                    seen_,
                    seenHashes_,
                    next_,
                    nextHashes_,
                    mergeScratch_,
                    mergeScratchHashes_
            );
            if (verbose_) {
                tcout << "    cumulative seen size: " << seen_.size() << '\n';
                tcout << "    seen merge time: " << timerMerge.getSeconds() << '\n';
            }
        }

        frontier_.swap(next_);
        frontierHashes_.swap(nextHashes_);
        next_.clear();
        nextHashes_.clear();

        updatePeakWorkspaceCapacity();

        if (verbose_) {
            tcout << "    new frontier size: " << frontier_.size() << '\n';

            printLaneStats("frontier lane", frontier_, frontierHashes_);
            printLaneStats("seen lane", seen_, seenHashes_);
            printLaneStats("next scratch", next_, nextHashes_);
            printLaneStats("merge scratch", mergeScratch_, mergeScratchHashes_);

            tcout << "    workspace live: " << fmtBytes(workspaceLiveBytes()) << '\n';
            tcout << "    workspace rsv: " << fmtBytes(workspaceCapacityBytes()) << '\n';
            tcout << "    workspace peak rsv: " << fmtBytes(peakWorkspaceCapacityBytes_) << '\n';

            tcout << "    total depth time: " << stepTimer.getSeconds() << '\n';
        }
    }

public:
    MU FrontierBuilderB1B2() = default;

    MU explicit FrontierBuilderB1B2(const Board& root, const bool verbose = true)
        : root_(root), colorCount_(root.getColorCount()), verbose_(verbose) {
        initRootState();
    }

    MU void setVerbose(const bool verbose) {
        verbose_ = verbose;
    }

    MU void reset(const Board& root) {
        root_ = root;
        colorCount_ = root.getColorCount();
        initRootState();
    }

    MU void buildExactNoneDepth(const u32 targetDepth,
                                JVec<B1B2>& outDepth,
                                JVec<u64>& outHashes) {
        outDepth.clear();
        outHashes.clear();

        initRootState();
        sorter_.ensureDepthSlots(targetDepth);

        if (targetDepth == 0) {
            outDepth.resize(1);
            outHashes.resize(1);
            outDepth[0] = frontier_[0];
            outHashes[0] = frontierHashes_[0];
            return;
        }

        for (u32 depth = 1; depth <= targetDepth; ++depth) {
            expandOneDepth(depth);
        }

        outDepth.clear();
        outHashes.clear();
        outDepth.swap(frontier_);
        outHashes.swap(frontierHashes_);
    }
};

template<int DEPTH>
MU static void buildUniqueNoneDepthFrontierB1B2(const Board& root,
                                                JVec<B1B2>& outDepth,
                                                JVec<u64>& outHashes,
                                                const bool verbose = true) {
    FrontierBuilderB1B2 builder(root, verbose);
    builder.buildExactNoneDepth(DEPTH, outDepth, outHashes);
}