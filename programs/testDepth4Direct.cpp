#include "code/include.hpp"
#include "code/solver/memory_perm_gen.hpp"

#include <iostream>
#include <set>
#include <string>
#include <vector>

int main() {
    const auto pair = BoardLookup::getBoardPair("3-2");
    if (pair == nullptr) {
        tcout << "Failed to load puzzle 3-2\n";
        return 1;
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();

    JVec<Board> boards;
    const bool should_alloc = true;

    Perms<Board>::getDepthFunc<eSequenceDir::ASCENDING>(start, boards, 4, should_alloc);

    tcout << "Amount of boards generated: " << boards.size() << '\n';

    std::size_t canonicalMatchCount = 0;
    std::set<std::string> originalSolutions;
    std::set<std::string> expandedSolutions;

    for (const Board& board : boards) {
        if (!(board == goal)) {
            continue;
        }

        ++canonicalMatchCount;

        const Memory& memory = board.getMemory();
        const std::string original = memory.asmStringForwards();
        originalSolutions.insert(original);

        std::vector<u8> baseMoves;
        baseMoves.reserve(memory.getMoveCount());
        for (u8 i = 0; i < memory.getMoveCount(); ++i) {
            baseMoves.push_back(memory.getMove(i));
        }

        const std::vector<std::vector<u8>> perms =
                createMemoryPermutationsChecked(start, goal, baseMoves);

        for (const std::vector<u8>& perm : perms) {
            Memory tempMemory;
            for (u8 move : perm) {
                tempMemory.setNextNMove<1>(move);
            }
            expandedSolutions.insert(tempMemory.asmStringForwards());
        }
    }

    tcout << "Canonical matches: " << canonicalMatchCount << '\n';
    tcout << "Unique original solutions: " << originalSolutions.size() << '\n';
    tcout << "Unique expanded solutions: " << expandedSolutions.size() << '\n';

    tcout << "\n=== Original solutions ===\n";
    for (const std::string& solution : originalSolutions) {
        tcout << solution << '\n';
    }

    tcout << "\n=== Expanded solutions ===\n";
    for (const std::string& solution : expandedSolutions) {
        tcout << solution << '\n';
    }

    tcout << "\n=== Expanded but not original ===\n";
    for (const std::string& solution : expandedSolutions) {
        if (!originalSolutions.contains(solution)) {
            tcout << solution << '\n';
        }
    }

    tcout << "\n=== Original but not expanded ===\n";
    for (const std::string& solution : originalSolutions) {
        if (!expandedSolutions.contains(solution)) {
            tcout << solution << '\n';
        }
    }

    return 0;
}