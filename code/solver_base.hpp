#pragma once
// code/solver_base.hpp

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils/format_bytes.hpp"
#include "utils/timer.hpp"
#include "utils/get_free_memory.hpp"
#include "utils/timestamped_cout.hpp"

#include "levels.hpp"
#include "memory_perm_gen.hpp"
#include "sorter.hpp"

#include "include/ghc/fs_std.hpp"

class MU BoardSolverBase {
public:
    static constexpr u32 MAX_DEPTH = 5;
    static constexpr u32 LEFT_STREAM_CHUNK_SIZE = 1u << 20;

protected:
    std::vector<JVec<Board>> board1Table;
    std::vector<JVec<u64>> board1HashesTable;

    std::vector<JVec<Board>> board2Table;
    std::vector<JVec<u64>> board2HashesTable;

    std::unordered_set<std::string> resultSet;
    std::set<std::string> expandedResultSet;

    BoardSorter<Board> boardSorter;

    const BoardPair* pair;
    Board board1;
    Board board2;
    bool hasFat;
    fs::path outDirectory;

    u32 depthSideMax{};
    u32 depthTotalMax{};
    u32 depthGuessMax{};

public:
    MU explicit BoardSolverBase(const BoardPair* pairIn) {
        pair = pairIn;
        board1 = pair->getStartState();
        board2 = pair->getEndState();
        hasFat = board1.getFatBool();

        StateHash::refreshHashFunc(board1);
    }

    MU void setDepthParams(const u32 depthSideMaxIn, const u32 depthGuessMaxIn, const u32 depthTotalMaxIn) {
        depthSideMax = depthSideMaxIn;
        depthTotalMax = depthTotalMaxIn;
        depthGuessMax = depthGuessMaxIn;

        board1Table.resize(depthSideMax + 1);
        board1HashesTable.resize(depthSideMax + 1);

        board2Table.resize(depthSideMax + 1);
        board2HashesTable.resize(depthSideMax + 1);
    }

    MU void setWriteDirectory(const std::string directory) {
        outDirectory = directory;
    }

    MU void preAllocateMemory(const u32 maxDepth = MAX_DEPTH) {
        const u32 highestDepth = std::max(1U, std::min(maxDepth, depthTotalMax + 1) / 2);

        Perms<Board>::reserveForDepth<eSequenceDir::ASCENDING>(
                board1,
                board1Table[highestDepth],
                board1HashesTable[highestDepth],
                highestDepth
        );
        Perms<Board>::reserveForDepth<eSequenceDir::DESCENDING>(
                board1,
                board1Table[highestDepth],
                board1HashesTable[highestDepth],
                highestDepth
        );

        if (highestDepth != 1) {
            Perms<Board>::reserveForDepth<eSequenceDir::ASCENDING>(
                    board1,
                    board1Table[highestDepth - 1],
                    board1HashesTable[highestDepth - 1],
                    highestDepth - 1
            );
            Perms<Board>::reserveForDepth<eSequenceDir::DESCENDING>(
                    board1,
                    board1Table[highestDepth - 1],
                    board1HashesTable[highestDepth - 1],
                    highestDepth - 1
            );
        }

        Perms<Board>::reserveForDepth<eSequenceDir::ASCENDING>(
                board2,
                board2Table[highestDepth],
                board2HashesTable[highestDepth],
                highestDepth
        );
        Perms<Board>::reserveForDepth<eSequenceDir::DESCENDING>(
                board2,
                board2Table[highestDepth],
                board2HashesTable[highestDepth],
                highestDepth
        );

        if (highestDepth != 1) {
            Perms<Board>::reserveForDepth<eSequenceDir::ASCENDING>(
                    board2,
                    board2Table[highestDepth - 1],
                    board2HashesTable[highestDepth - 1],
                    highestDepth - 1
            );
            Perms<Board>::reserveForDepth<eSequenceDir::DESCENDING>(
                    board2,
                    board2Table[highestDepth - 1],
                    board2HashesTable[highestDepth - 1],
                    highestDepth - 1
            );
        }

        boardSorter.ensureAux(highestDepth, 0);
        if (highestDepth != 1) {
            boardSorter.ensureAux(highestDepth - 1, 0);
        }
    }

    MUND std::string getMemorySize() const {
        u64 allocMemory = 0;

        for (const auto& boardTable : board1Table) {
            if (!boardTable.empty()) {
                allocMemory += boardTable.size() * sizeof(boardTable[0]);
            }
        }
        for (const auto& hashTable : board1HashesTable) {
            if (!hashTable.empty()) {
                allocMemory += hashTable.size() * sizeof(hashTable[0]);
            }
        }

        for (const auto& boardTable : board2Table) {
            if (!boardTable.empty()) {
                allocMemory += boardTable.size() * sizeof(boardTable[0]);
            }
        }
        for (const auto& hashTable : board2HashesTable) {
            if (!hashTable.empty()) {
                allocMemory += hashTable.size() * sizeof(hashTable[0]);
            }
        }

        return bytesFormatted<1000>(allocMemory);
    }

protected:
    MU static std::vector<std::string> tokenizeMoves(const std::string& line) {
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }

        return tokens;
    }

    MU static bool applyMovesAndCheckGoal(Board board,
                                          const Board& goal,
                                          const std::vector<u8>& moves) {
        for (const u8 move : moves) {
            allActStructList[move].action(board);
        }
        return board == goal;
    }

    MU static bool tryParseValidatedLine(const std::string& rawLine,
                                         const bool isFatPuzzle,
                                         std::vector<u8>& outMoves,
                                         std::string& outCanonicalLine) {
        const std::vector<std::string> inputTokens = tokenizeMoves(rawLine);
        if (inputTokens.empty()) {
            return false;
        }

        try {
            std::string temp = rawLine;
            outMoves = isFatPuzzle
                               ? Memory::parseFatMoveString(temp)
                               : Memory::parseNormMoveString(temp);
        } catch (...) {
            return false;
        }

        if (outMoves.size() != inputTokens.size()) {
            return false;
        }

        outCanonicalLine = moveVectorToDirectString(outMoves);
        const std::vector<std::string> canonicalTokens = tokenizeMoves(outCanonicalLine);

        if (canonicalTokens.size() != inputTokens.size()) {
            return false;
        }

        for (std::size_t i = 0; i < inputTokens.size(); ++i) {
            if (inputTokens[i] != canonicalTokens[i]) {
                return false;
            }
        }

        return true;
    }

    MU static bool tryExpandNormalLine(const Board& start,
                                       const Board& goal,
                                       const std::string& solution,
                                       std::set<std::string>& expandedSolutions) {
        std::vector<u8> baseMoves;
        std::string canonicalBaseLine;

        if (!tryParseValidatedLine(solution, false, baseMoves, canonicalBaseLine)) {
            return false;
        }

        if (!applyMovesAndCheckGoal(start, goal, baseMoves)) {
            return false;
        }

        std::vector<std::vector<u8>> perms;
        try {
            perms = createMemoryPermutationsChecked(start, goal, baseMoves);
        } catch (...) {
            return false;
        }

        expandedSolutions.insert(canonicalBaseLine);

        for (const auto& perm : perms) {
            if (!applyMovesAndCheckGoal(start, goal, perm)) {
                continue;
            }

            expandedSolutions.insert(moveVectorToDirectString(perm));
        }

        return true;
    }

    MU static bool tryKeepFatLine(const Board& start,
                                  const Board& goal,
                                  const std::string& solution,
                                  std::set<std::string>& fatSolutions) {
        std::vector<u8> baseMoves;
        std::string canonicalLine;

        if (!tryParseValidatedLine(solution, true, baseMoves, canonicalLine)) {
            return false;
        }

        if (!applyMovesAndCheckGoal(start, goal, baseMoves)) {
            return false;
        }

        fatSolutions.insert(canonicalLine);
        return true;
    }

    MU static void writeLines(const fs::path& filepath, const std::set<std::string>& lines) {
        std::ofstream file(filepath, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open output file");
        }

        for (const auto& line : lines) {
            file << line << '\n';
        }
    }

    MU void expandRawSolutionsIntoFinalSet() {
        expandedResultSet.clear();

        std::size_t validInputSolutions = 0;
        std::size_t skippedInputSolutions = 0;

        tcout << "\nExpanding solver results...\n";

        for (const auto& solution : resultSet) {
            const bool ok = hasFat
                                ? tryKeepFatLine(board1, board2, solution, expandedResultSet)
                                : tryExpandNormalLine(board1, board2, solution, expandedResultSet);

            if (ok) {
                ++validInputSolutions;
            } else {
                ++skippedInputSolutions;
                tcout << "Skipping invalid solver solution: " << solution << '\n';
            }
        }

        tcout << "Original solver solutions: " << resultSet.size() << '\n';
        tcout << "Valid base solutions: " << validInputSolutions << '\n';
        tcout << "Skipped base solutions: " << skippedInputSolutions << '\n';
        tcout << "Expanded total solutions: " << expandedResultSet.size() << '\n';
    }

    MU void writeExpandedSolutions(const u32 currentDepth) {
        if (expandedResultSet.empty()) {
            tcout << "No expanded solutions produced.\n";
            return;
        }

        const std::size_t moveCount = tokenizeMoves(*expandedResultSet.begin()).size();

        fs::path outputDir = outDirectory / "levels";
        fs::create_directories(outputDir);

        const std::string filename = pair->getName()
                                 + "_c" + std::to_string(moveCount)
                                 + "_" + std::to_string(expandedResultSet.size())
                                 + ".txt";

        const fs::path filepath = outputDir / filename;
        tcout << "Saving expanded results to " << filepath << ".\n";
        writeLines(filepath, expandedResultSet);
    }

    MU static void validateStateHashLanes(const JVec<Board>& states,
                                          const JVec<u64>& hashes,
                                          const char* label) {
        if (states.size() != hashes.size()) {
            throw std::runtime_error(std::string("state/hash lane size mismatch in ") + label);
        }
    }
};