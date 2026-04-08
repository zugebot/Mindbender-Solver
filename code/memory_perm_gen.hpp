#pragma once
// code/perms_gen.hpp

#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstring>

#include "include.hpp"
#include "utils/processor.hpp"
#include "utils/timestamped_cout.hpp"

struct MemoryPermGenPair {
    u32 start = 0;
    u32 end = 0;

    MemoryPermGenPair() = default;

    MemoryPermGenPair(const u32 theStart, const u32 theLength)
        : start(theStart), end(theStart + theLength) {}
};

class MemoryPermGen {
    std::vector<MemoryPermGenPair> myPairs;
    std::vector<u8> myToBePermuted;

public:
    std::vector<std::vector<u8>> myOutput{};

    void allPermutations(
            const std::vector<u8>& toBePermuted,
            const std::vector<MemoryPermGenPair>& thePairs) {
        myToBePermuted = toBePermuted;
        myPairs = thePairs;
        myOutput.clear();

        if (myPairs.empty()) {
            myOutput.emplace_back(myToBePermuted);
            return;
        }

        allPermutations(myPairs[0].start, 0);
    }

    MU void printVectors() {
        tcout << "\n";

        for (i32 index = 0; index < static_cast<i32>(myOutput.size()); ++index) {
            const std::vector<u8>& theVector = myOutput[index];

            tcout << std::setw(3) << index + 1 << ": { ";
            for (i32 i = 0; i < static_cast<i32>(theVector.size()); ++i) {
                if (theVector[i] == 0) {
                    tcout << ". ";
                } else {
                    tcout << static_cast<u32>(theVector[i]) << " ";
                }
            }
            tcout << "}\n";
        }
    }

private:
    void allPermutations(const u32 nextIndex, const u32 pairIndex) {
        const MemoryPermGenPair& pair = myPairs[pairIndex];

        if (nextIndex == pair.end) {
            if (pairIndex + 1 == myPairs.size()) {
                myOutput.emplace_back(myToBePermuted);
            } else {
                allPermutations(myPairs[pairIndex + 1].start, pairIndex + 1);
            }
            return;
        }

        for (u32 i = nextIndex; i < pair.end; ++i) {
            std::swap(myToBePermuted[i], myToBePermuted[nextIndex]);
            allPermutations(nextIndex + 1, pairIndex);
            std::swap(myToBePermuted[i], myToBePermuted[nextIndex]);
        }
    }
};

MUND static char getMoveDirection(const u8 action) {
    return allActStructList[action].name[0];
}

MUND static std::string moveToDirectString(const u8 move) {
    char temp[5] = {};
    std::memcpy(temp, allActStructList[move].name.data(), 4);
    return temp;
}

MUND static std::string moveVectorToDirectString(const std::vector<u8>& moves) {
    if (moves.empty()) {
        return "";
    }

    std::string out;
    out.reserve(moves.size() * 4);

    for (u32 i = 0; i < static_cast<u32>(moves.size()); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out += moveToDirectString(moves[i]);
    }

    return out;
}

MU static std::vector<MemoryPermGenPair> buildMemoryPermutationPairs(
        const std::vector<u8>& theMemory) {
    std::vector<MemoryPermGenPair> pairs;

    if (theMemory.empty()) {
        return pairs;
    }

    u32 groupStart = 0;
    u32 groupLength = 1;
    char currentDir = getMoveDirection(theMemory[0]);

    for (u32 i = 1; i < static_cast<u32>(theMemory.size()); ++i) {
        const char thisDir = getMoveDirection(theMemory[i]);

        if (thisDir == currentDir) {
            ++groupLength;
        } else {
            if (groupLength > 1) {
                pairs.emplace_back(groupStart, groupLength);
            }
            groupStart = i;
            groupLength = 1;
            currentDir = thisDir;
        }
    }

    if (groupLength > 1) {
        pairs.emplace_back(groupStart, groupLength);
    }

    return pairs;
}

MU static std::vector<std::vector<u8>> createMemoryPermutations(
        const std::vector<u8>& theMemory) {
    const std::vector<MemoryPermGenPair> pairs = buildMemoryPermutationPairs(theMemory);

    MemoryPermGen gen;
    gen.allPermutations(theMemory, pairs);
    return gen.myOutput;
}

// ============================================================
// Checked semantic expansion
// ============================================================

MUND static std::string encodeMoveVector(const std::vector<u8>& moves) {
    std::string out;
    out.reserve(moves.size() + 1);

    for (u8 move : moves) {
        out.push_back(static_cast<char>(move));
    }

    out.push_back('\xff');
    return out;
}

MUND static bool moveVectorSolvesBoard(
        const Board& start,
        const Board& goal,
        const std::vector<u8>& moves) {
    Board temp = start;
    for (u8 move : moves) {
        allActStructList[move].action(temp);
    }
    return temp == goal;
}

MUND static bool moveVectorSolvesBoardNoneLegal(
        const Board& start,
        const Board& goal,
        const std::vector<u8>& moves) {
    Board temp = start;

    for (u8 move : moves) {
        Board next = temp;
        allActStructList[move].action(next);

        if (next == temp) {
            return false;
        }

        temp = next;
    }

    return temp == goal;
}

MU static std::vector<Board> buildPrefixStates(
        const Board& start,
        const std::vector<u8>& moves) {
    std::vector<Board> states;
    states.resize(moves.size() + 1);
    states[0] = start;

    for (u32 i = 0; i < static_cast<u32>(moves.size()); ++i) {
        states[i + 1] = states[i];
        allActStructList[moves[i]].action(states[i + 1]);
    }

    return states;
}

MUND static const std::vector<u8>& getNormalMovesForDirection(const char dir) {
    static const std::vector<u8> rowMoves = [] {
        std::vector<u8> out;
        out.reserve(NORMAL_ROW_MOVE_COUNT);
        for (u32 i = 0; i < NORMAL_ROW_MOVE_COUNT; ++i) {
            out.push_back(static_cast<u8>(i));
        }
        return out;
    }();

    static const std::vector<u8> colMoves = [] {
        std::vector<u8> out;
        out.reserve(NORMAL_COL_MOVE_COUNT);
        for (u32 i = 0; i < NORMAL_COL_MOVE_COUNT; ++i) {
            out.push_back(static_cast<u8>(NORMAL_ROW_MOVE_COUNT + NORMAL_MOVE_GAP_COUNT + i));
        }
        return out;
    }();

    return dir == 'R' ? rowMoves : colMoves;
}

MU static std::vector<std::vector<u8>> generateReverseCrossCandidates(
        const Board& start,
        const std::vector<u8>& moves) {
    std::vector<std::vector<u8>> out;

    if (start.getFatBool()) {
        return out;
    }

    if (moves.size() < 2) {
        return out;
    }

    const std::vector<Board> prefixStates = buildPrefixStates(start, moves);

    for (u32 i = 0; i + 1 < static_cast<u32>(moves.size()); ++i) {
        const u8 firstMove = moves[i];
        const u8 secondMove = moves[i + 1];
        const char firstDir = getMoveDirection(firstMove);
        const char secondDir = getMoveDirection(secondMove);

        if (firstDir == secondDir) {
            continue;
        }

        const Board& beforePair = prefixStates[i];
        const Board& afterPair = prefixStates[i + 2];

        const std::vector<u8>& swappedFirstDirMoves = getNormalMovesForDirection(secondDir);
        const std::vector<u8>& swappedSecondDirMoves = getNormalMovesForDirection(firstDir);

        for (u8 replacementFirst : swappedFirstDirMoves) {
            Board temp = beforePair;
            allActStructList[replacementFirst].action(temp);

            for (u8 replacementSecond : swappedSecondDirMoves) {
                Board temp2 = temp;
                allActStructList[replacementSecond].action(temp2);

                if (!(temp2 == afterPair)) {
                    continue;
                }

                std::vector<u8> candidate = moves;
                candidate[i] = replacementFirst;
                candidate[i + 1] = replacementSecond;

                if (candidate != moves) {
                    out.push_back(std::move(candidate));
                }
            }
        }
    }

    return out;
}

MU static std::vector<std::vector<u8>> createMemoryPermutationsChecked(
        const Board& start,
        const Board& goal,
        const std::vector<u8>& theMemory) {
    std::vector<std::vector<u8>> result;
    std::unordered_set<std::string> seen;
    std::deque<std::vector<u8>> work;

    auto enqueueIfValid = [&](const std::vector<u8>& candidate) {
        if (!moveVectorSolvesBoardNoneLegal(start, goal, candidate)) {
            return;
        }

        const std::string key = encodeMoveVector(candidate);
        if (seen.insert(key).second) {
            work.push_back(candidate);
        }
    };

    enqueueIfValid(theMemory);

    const std::vector<std::vector<u8>> initialPerms = createMemoryPermutations(theMemory);
    for (const auto& perm : initialPerms) {
        enqueueIfValid(perm);
    }

    while (!work.empty()) {
        std::vector<u8> current = std::move(work.front());
        work.pop_front();

        result.push_back(current);

        const std::vector<std::vector<u8>> sameDirPerms = createMemoryPermutations(current);
        for (const auto& perm : sameDirPerms) {
            enqueueIfValid(perm);
        }

        const std::vector<std::vector<u8>> reverseCrossPerms =
                generateReverseCrossCandidates(start, current);
        for (const auto& candidate : reverseCrossPerms) {
            enqueueIfValid(candidate);
        }
    }

    std::sort(result.begin(), result.end(), [](const std::vector<u8>& a, const std::vector<u8>& b) {
        return moveVectorToDirectString(a) < moveVectorToDirectString(b);
    });

    return result;
}

MU static std::vector<std::string> createMemoryPermutationStringsChecked(
        const Board& start,
        const Board& goal,
        const std::vector<u8>& theMemory) {
    std::vector<std::string> out;
    const std::vector<std::vector<u8>> perms =
            createMemoryPermutationsChecked(start, goal, theMemory);

    out.reserve(perms.size());
    for (const auto& perm : perms) {
        out.push_back(moveVectorToDirectString(perm));
    }

    return out;
}