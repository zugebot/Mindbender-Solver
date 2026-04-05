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

    MemoryPermGenPair(C u32 theStart, C u32 theLength)
        : start(theStart), end(theStart + theLength) {}
};

class MemoryPermGen {
    std::vector<MemoryPermGenPair> myPairs;
    std::vector<u8> myToBePermuted;

public:
    std::vector<std::vector<u8>> myOutput{};

    void allPermutations(
            C std::vector<u8>& toBePermuted,
            C std::vector<MemoryPermGenPair>& thePairs) {
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
            C std::vector<u8>& theVector = myOutput[index];

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
    void allPermutations(C u32 nextIndex, C u32 pairIndex) {
        C MemoryPermGenPair& pair = myPairs[pairIndex];

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

MUND static char getMoveDirection(C u8 action) {
    return allActStructList[action].name[0];
}

MUND static std::string moveToDirectString(C u8 move) {
    char temp[5] = {};
    std::memcpy(temp, allActStructList[move].name.data(), 4);
    return temp;
}

MUND static std::string moveVectorToDirectString(C std::vector<u8>& moves) {
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
        C std::vector<u8>& theMemory) {
    std::vector<MemoryPermGenPair> pairs;

    if (theMemory.empty()) {
        return pairs;
    }

    u32 groupStart = 0;
    u32 groupLength = 1;
    char currentDir = getMoveDirection(theMemory[0]);

    for (u32 i = 1; i < static_cast<u32>(theMemory.size()); ++i) {
        C char thisDir = getMoveDirection(theMemory[i]);

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
        C std::vector<u8>& theMemory) {
    C std::vector<MemoryPermGenPair> pairs = buildMemoryPermutationPairs(theMemory);

    MemoryPermGen gen;
    gen.allPermutations(theMemory, pairs);
    return gen.myOutput;
}

// ============================================================
// Checked semantic expansion
// ============================================================

MUND static std::string encodeMoveVector(C std::vector<u8>& moves) {
    std::string out;
    out.reserve(moves.size() + 1);

    for (u8 move : moves) {
        out.push_back(static_cast<char>(move));
    }

    out.push_back('\xff');
    return out;
}

MUND static bool moveVectorSolvesBoard(
        C Board& start,
        C Board& goal,
        C std::vector<u8>& moves) {
    Board temp = start;
    for (u8 move : moves) {
        allActStructList[move].action(temp);
    }
    return temp == goal;
}

MUND static bool moveVectorSolvesBoardNoneLegal(
        C Board& start,
        C Board& goal,
        C std::vector<u8>& moves) {
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
        C Board& start,
        C std::vector<u8>& moves) {
    std::vector<Board> states;
    states.resize(moves.size() + 1);
    states[0] = start;

    for (u32 i = 0; i < static_cast<u32>(moves.size()); ++i) {
        states[i + 1] = states[i];
        allActStructList[moves[i]].action(states[i + 1]);
    }

    return states;
}

MUND static C std::vector<u8>& getNormalMovesForDirection(C char dir) {
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
        C Board& start,
        C std::vector<u8>& moves) {
    std::vector<std::vector<u8>> out;

    if (start.getFatBool()) {
        return out;
    }

    if (moves.size() < 2) {
        return out;
    }

    C std::vector<Board> prefixStates = buildPrefixStates(start, moves);

    for (u32 i = 0; i + 1 < static_cast<u32>(moves.size()); ++i) {
        C u8 firstMove = moves[i];
        C u8 secondMove = moves[i + 1];
        C char firstDir = getMoveDirection(firstMove);
        C char secondDir = getMoveDirection(secondMove);

        if (firstDir == secondDir) {
            continue;
        }

        C Board& beforePair = prefixStates[i];
        C Board& afterPair = prefixStates[i + 2];

        C std::vector<u8>& swappedFirstDirMoves = getNormalMovesForDirection(secondDir);
        C std::vector<u8>& swappedSecondDirMoves = getNormalMovesForDirection(firstDir);

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
        C Board& start,
        C Board& goal,
        C std::vector<u8>& theMemory) {
    std::vector<std::vector<u8>> result;
    std::unordered_set<std::string> seen;
    std::deque<std::vector<u8>> work;

    auto enqueueIfValid = [&](C std::vector<u8>& candidate) {
        if (!moveVectorSolvesBoardNoneLegal(start, goal, candidate)) {
            return;
        }

        C std::string key = encodeMoveVector(candidate);
        if (seen.insert(key).second) {
            work.push_back(candidate);
        }
    };

    enqueueIfValid(theMemory);

    C std::vector<std::vector<u8>> initialPerms = createMemoryPermutations(theMemory);
    for (C auto& perm : initialPerms) {
        enqueueIfValid(perm);
    }

    while (!work.empty()) {
        std::vector<u8> current = std::move(work.front());
        work.pop_front();

        result.push_back(current);

        C std::vector<std::vector<u8>> sameDirPerms = createMemoryPermutations(current);
        for (C auto& perm : sameDirPerms) {
            enqueueIfValid(perm);
        }

        C std::vector<std::vector<u8>> reverseCrossPerms =
                generateReverseCrossCandidates(start, current);
        for (C auto& candidate : reverseCrossPerms) {
            enqueueIfValid(candidate);
        }
    }

    std::sort(result.begin(), result.end(), [](C std::vector<u8>& a, C std::vector<u8>& b) {
        return moveVectorToDirectString(a) < moveVectorToDirectString(b);
    });

    return result;
}

MU static std::vector<std::string> createMemoryPermutationStringsChecked(
        C Board& start,
        C Board& goal,
        C std::vector<u8>& theMemory) {
    std::vector<std::string> out;
    C std::vector<std::vector<u8>> perms =
            createMemoryPermutationsChecked(start, goal, theMemory);

    out.reserve(perms.size());
    for (C auto& perm : perms) {
        out.push_back(moveVectorToDirectString(perm));
    }

    return out;
}