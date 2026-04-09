#include "code/include.hpp"
#include "code/solver/frontier_builder.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <iostream>

namespace {
    void printBoardSample(const char* label,
                          const JVec<B1B2>& states,
                          const JVec<u64>& hashes,
                          std::size_t maxCount = 5) {
        tcout << label << ": count=" << states.size() << '\n';

        const std::size_t shown = std::min<std::size_t>(states.size(), maxCount);
        for (std::size_t i = 0; i < shown; ++i) {
            const Board board = makeBoardFromState(states[i]);
            tcout << "  [" << i << "] hash=" << hashes[i]
                  << " fatXY=" << static_cast<u32>(board.getFatXY()) << '\n';
            tcout << board.toBlandString() << "\n\n";
        }
    }

    void printIntersectionSample(const std::vector<std::pair<const B1B2*, const B1B2*>>& matches,
                                 std::size_t maxCount = 5) {
        const std::size_t shown = std::min<std::size_t>(matches.size(), maxCount);
        for (std::size_t i = 0; i < shown; ++i) {
            const Board lhs = makeBoardFromState(*matches[i].first);
            const Board rhs = makeBoardFromState(*matches[i].second);

            tcout << "match[" << i << "] lhs:\n" << lhs.toBlandString() << "\n\n";
            tcout << "match[" << i << "] rhs:\n" << rhs.toBlandString() << "\n\n";
        }
    }
}

int main() {
    constexpr const char* PUZZLE_NAME = "4-4";
    constexpr u32 LEFT_DEPTH = 2;
    constexpr u32 RIGHT_DEPTH = 5;

    const BoardPair* pair = BoardLookup::getBoardPair(PUZZLE_NAME);
    if (pair == nullptr) {
        tcout << "Failed to find puzzle " << PUZZLE_NAME << "\n";
        return 1;
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();

    tcout << "Puzzle: " << pair->getName() << '\n';
    tcout << pair->toString() << "\n\n";
    tcout << "Start fat: " << start.getFatBool()
          << " @ (" << static_cast<u32>(start.getFatX())
          << ", " << static_cast<u32>(start.getFatY()) << ")\n";
    tcout << "Goal fat:  " << goal.getFatBool()
          << " @ (" << static_cast<u32>(goal.getFatX())
          << ", " << static_cast<u32>(goal.getFatY()) << ")\n\n";

    StateHash::refreshHashFunc(start);

    JVec<B1B2> leftStates;
    JVec<u64> leftHashes;
    JVec<B1B2> rightStates;
    JVec<u64> rightHashes;

    {
        Timer timer;
        buildUniqueNoneDepthFrontierB1B2<LEFT_DEPTH>(start, leftStates, leftHashes);
        tcout << "Built left exact NONE depth " << LEFT_DEPTH
              << " in " << timer.getSeconds() << "s\n";
    }

    {
        Timer timer;
        buildUniqueNoneDepthFrontierB1B2<RIGHT_DEPTH>(goal, rightStates, rightHashes);
        tcout << "Built right exact NONE depth " << RIGHT_DEPTH
              << " in " << timer.getSeconds() << "s\n";
    }

    tcout << "\nBefore final sort:\n";
    tcout << "  left size:  " << leftStates.size() << '\n';
    tcout << "  right size: " << rightStates.size() << "\n\n";

    {
        Timer timer;
        sortStatesByHash(leftStates, leftHashes);
        compactUniqueSortedStatesInPlace(leftStates, leftHashes);
        tcout << "Left final sort+dedupe:  " << timer.getSeconds() << "s\n";
    }

    {
        Timer timer;
        sortStatesByHash(rightStates, rightHashes);
        compactUniqueSortedStatesInPlace(rightStates, rightHashes);
        tcout << "Right final sort+dedupe: " << timer.getSeconds() << "s\n";
    }

    tcout << "\nAfter final sort:\n";
    tcout << "  left size:  " << leftStates.size() << '\n';
    tcout << "  right size: " << rightStates.size() << "\n\n";

    printBoardSample("Left sample", leftStates, leftHashes);
    printBoardSample("Right sample", rightStates, rightHashes);

    std::vector<std::pair<const B1B2*, const B1B2*>> matches;
    {
        Timer timer;
        matches = intersection(leftStates, leftHashes, rightStates, rightHashes);
        tcout << "Intersection time: " << timer.getSeconds() << "s\n";
    }

    tcout << "Intersection count: " << matches.size() << "\n\n";
    printIntersectionSample(matches);

    return 0;
}
