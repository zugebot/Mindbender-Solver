#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "utils/timestamped_cout.hpp"
#include "utils/timer.hpp"

#include "board.hpp"
#include "perms.hpp"
#include "sorter.hpp"

template<typename T>
MU static void sortStatesByHash(JVec<T>& states) {
    std::sort(states.begin(), states.end(), [](C T& a, C T& b) {
        return a.getHash() < b.getHash();
    });
}

template<typename T>
MU static void compactUniqueSortedStatesInPlace(JVec<T>& states) {
    if (states.empty()) {
        return;
    }

    std::size_t writeIndex = 1;
    for (std::size_t readIndex = 1; readIndex < states.size(); ++readIndex) {
        if (!(states[readIndex] == states[writeIndex - 1])) {
            states[writeIndex] = states[readIndex];
            ++writeIndex;
        }
    }

    states.resize(writeIndex);
}

template<typename T>
MU static void removeStatesPresentInSortedSetLinear(JVec<T>& states,
                                                    C JVec<T>& sortedSeenStates) {
    if (states.empty() || sortedSeenStates.empty()) {
        return;
    }

    std::size_t writeIndex = 0;
    std::size_t seenIndex = 0;

    for (std::size_t i = 0; i < states.size(); ++i) {
        while (seenIndex < sortedSeenStates.size()
               && sortedSeenStates[seenIndex].getHash() < states[i].getHash()) {
            ++seenIndex;
        }

        bool foundEqual = false;
        std::size_t probe = seenIndex;
        while (probe < sortedSeenStates.size()
               && sortedSeenStates[probe].getHash() == states[i].getHash()) {
            if (sortedSeenStates[probe] == states[i]) {
                foundEqual = true;
                break;
            }
            ++probe;
        }

        if (!foundEqual) {
            states[writeIndex] = states[i];
            ++writeIndex;
        }
    }

    states.resize(writeIndex);
}

template<typename T>
MU static void mergeSortedUniqueStatesIntoSeen(JVec<T>& seenStates,
                                               C JVec<T>& newStates,
                                               JVec<T>& scratch) {
    if (newStates.empty()) {
        return;
    }

    scratch.resize(seenStates.size() + newStates.size());

    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t out = 0;

    while (i < seenStates.size() && j < newStates.size()) {
        if (seenStates[i].getHash() < newStates[j].getHash()) {
            scratch[out++] = seenStates[i++];
            continue;
        }

        if (newStates[j].getHash() < seenStates[i].getHash()) {
            scratch[out++] = newStates[j++];
            continue;
        }

        if (seenStates[i] == newStates[j]) {
            scratch[out++] = seenStates[i];
            ++i;
            ++j;
            continue;
        }

        scratch[out++] = seenStates[i++];
        scratch[out++] = newStates[j++];
    }

    while (i < seenStates.size()) {
        scratch[out++] = seenStates[i++];
    }

    while (j < newStates.size()) {
        scratch[out++] = newStates[j++];
    }

    scratch.resize(out);
    seenStates = std::move(scratch);
    scratch.resize(0);
}

class FrontierBuilderB1B2 {
    Board root_{};
    BoardSorter<B1B2> sorter_{};
    JVec<B1B2> seen_{};
    JVec<B1B2> currentDepth_{};
    JVec<B1B2> mergeScratch_{};
    u32 colorCount_ = 0;

public:
    MU FrontierBuilderB1B2() = default;

    MU explicit FrontierBuilderB1B2(C Board& root)
        : root_(root), colorCount_(root.getColorCount()) {
        seen_.resize(1);
        seen_[0] = root_.asB1B2();
    }

    MU void reset(C Board& root) {
        root_ = root;
        colorCount_ = root.getColorCount();

        seen_.clear();
        currentDepth_.clear();
        mergeScratch_.clear();

        seen_.resize(1);
        seen_[0] = root_.asB1B2();
    }

    MU void buildExactNoneDepth(C u32 targetDepth,
                                JVec<B1B2>& outDepth) {
        outDepth.clear();

        if (targetDepth == 0) {
            outDepth.resize(1);
            outDepth[0] = root_.asB1B2();
            return;
        }

        for (u32 depth = 1; depth <= targetDepth; ++depth) {
            Timer stepTimer;
            tcout << "Generating NONE depth " << depth << "...\n" << std::flush;

            currentDepth_.clear();
            Perms<B1B2>::getDepthFunc<eSequenceDir::NONE>(root_, currentDepth_, depth, true);
            tcout << "    raw size: " << currentDepth_.size() << '\n';

            {
                Timer timerSort;
                sorter_.sortBoards(currentDepth_, depth, colorCount_);
                tcout << "    sort time: " << timerSort.getSeconds() << '\n';
            }

            {
                Timer timerDedupe;
                compactUniqueSortedStatesInPlace(currentDepth_);
                tcout << "    de-duped self size: " << currentDepth_.size() << '\n';
                tcout << "    self dedupe time: " << timerDedupe.getSeconds() << '\n';
            }

            {
                Timer timerSubtract;
                removeStatesPresentInSortedSetLinear(currentDepth_, seen_);
                tcout << "    removed shallower duplicates: " << currentDepth_.size() << '\n';
                tcout << "    subtract seen time: " << timerSubtract.getSeconds() << '\n';
            }

            {
                Timer timerMerge;
                mergeSortedUniqueStatesIntoSeen(seen_, currentDepth_, mergeScratch_);
                tcout << "    cumulative seen size: " << seen_.size() << '\n';
                tcout << "    seen merge time: " << timerMerge.getSeconds() << '\n';
            }

            tcout << "    total depth time: " << stepTimer.getSeconds() << '\n';

            if (depth == targetDepth) {
                outDepth = currentDepth_;
            }
        }
    }
};

template<int DEPTH>
MU static void buildUniqueNoneDepthFrontierB1B2(C Board& root,
                                                JVec<B1B2>& outDepth) {
    FrontierBuilderB1B2 builder(root);
    builder.buildExactNoneDepth(DEPTH, outDepth);
}

class RightFrontierIndexB1B2 {
public:
    using BucketMap = std::unordered_map<u64, std::vector<const B1B2*>>;

private:
    JVec<B1B2> states_{};
    BucketMap buckets_{};

public:
    MU void clear() {
        states_.clear();
        buckets_.clear();
    }

    MU void buildFromUniqueStates(JVec<B1B2>&& states) {
        states_ = std::move(states);
        buckets_.clear();

        buckets_.reserve(states_.size());

        for (C auto& state : states_) {
            buckets_[state.getHash()].push_back(&state);
        }
    }

    MUND C JVec<B1B2>& states() C {
        return states_;
    }

    MUND std::size_t size() C {
        return states_.size();
    }

    MU void collectMatches(C JVec<B1B2>& leftStates,
                           JVec<B1B2>& outUniqueMatches) C {
        outUniqueMatches.clear();
        
        if (leftStates.empty() || states_.empty()) {
            return;
        }

        outUniqueMatches.resize(leftStates.size());
        std::size_t writeIndex = 0;

        for (const B1B2& lhs : leftStates) {
            const auto it = buckets_.find(lhs.getHash());
            if (it == buckets_.end()) {
                continue;
            }

            for (const B1B2* rhs : it->second) {
                if (lhs == *rhs) {
                    outUniqueMatches[writeIndex] = lhs;
                    ++writeIndex;
                    break;
                }
            }
        }

        outUniqueMatches.resize(writeIndex);

        if (!outUniqueMatches.empty()) {
            sortStatesByHash(outUniqueMatches);
            compactUniqueSortedStatesInPlace(outUniqueMatches);
        }
    }
};

MU static Board makeBoardFromState(C B1B2& state) {
    Board out;
    out.b1 = state.b1;
    out.b2 = state.b2;
    return out;
}