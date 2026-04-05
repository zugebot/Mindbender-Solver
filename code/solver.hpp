#pragma once

#ifdef BOOST_FOUND
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#endif

#include <atomic>
#include <mutex>
#include <thread>

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

#include "intersection.hpp"
#include "levels.hpp"
#include "perms.hpp"
#include "memory_perm_gen.hpp"
#include "sorter.hpp"
#include "frontier_recovery.hpp"

#include "include/ghc/fs_std.hpp"

#define IF_DEBUG_COUT(stuff) if constexpr (debug) { tcout << stuff; }

class MU BoardSolver {
public:
    static constexpr u32 MAX_DEPTH = 5;

    std::vector<JVec<Memory>> board1Table;
    std::vector<JVec<Memory>> board2Table;

    std::unordered_set<std::string> resultSet;
    std::set<std::string> expandedResultSet;

#ifdef BOOST_FOUND
#else
    BoardSorter<Memory> boardSorter;
#endif

    C BoardPair* pair;
    Board board1;
    Board board2;
    bool hasFat;
    fs::path outDirectory;

    u32 depthSideMax{};
    u32 depthTotalMax{};
    u32 depthGuessMax{};

private:
    struct RecoveryBoardFrontierCache {
        Board root{};
        u32 depth = 0;
        bool valid = false;
        JVec<Board> states;
        BoardSorter<Board> sorter;
    };
    
    
    RightFrontierIndexB1B2 rightFrontierIndex_;
    RecoveryBoardFrontierCache prefixLeftCache_;
    RecoveryBoardFrontierCache goalRightCache_;

public:
    MU explicit BoardSolver(C BoardPair* pairIn) {
        pair = pairIn;
        board1 = pair->getStartState();
        board2 = pair->getEndState();
        hasFat = board1.getFatBool();

        B1B2::refreshHashFunc(board1);
        Board::refreshHashFunc(board1);
        Memory::refreshHashFunc(board1);
    }

    MU void setDepthParams(C u32 depthSideMaxIn, C u32 depthGuessMaxIn, C u32 depthTotalMaxIn) {
        depthSideMax = depthSideMaxIn;
        depthTotalMax = depthTotalMaxIn;
        depthGuessMax = depthGuessMaxIn;

        board1Table.resize(depthSideMax + 1);
        board2Table.resize(depthSideMax + 1);
    }

    MU void setWriteDirectory(C std::string directory) {
        outDirectory = directory;
    }

    MU void preAllocateMemory(C u32 maxDepth = MAX_DEPTH) {
        C u32 highestDepth = std::max(1U, std::min(maxDepth, depthTotalMax + 1) / 2);

        Perms<Memory>::reserveForDepth<eSequenceDir::ASCENDING>(board1, board1Table[highestDepth], highestDepth);
        Perms<Memory>::reserveForDepth<eSequenceDir::DESCENDING>(board1, board1Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms<Memory>::reserveForDepth<eSequenceDir::ASCENDING>(board1, board1Table[highestDepth - 1], highestDepth - 1);
            Perms<Memory>::reserveForDepth<eSequenceDir::DESCENDING>(board1, board1Table[highestDepth - 1], highestDepth - 1);
        }

        Perms<Memory>::reserveForDepth<eSequenceDir::ASCENDING>(board2, board2Table[highestDepth], highestDepth);
        Perms<Memory>::reserveForDepth<eSequenceDir::DESCENDING>(board2, board2Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms<Memory>::reserveForDepth<eSequenceDir::ASCENDING>(board2, board2Table[highestDepth - 1], highestDepth - 1);
            Perms<Memory>::reserveForDepth<eSequenceDir::DESCENDING>(board2, board2Table[highestDepth - 1], highestDepth - 1);
        }

#ifdef BOOST_FOUND
#else
        if (!hasFat) {
            boardSorter.ensureAux(highestDepth, BOARD_PRE_MAX_MALLOC_SIZES[highestDepth]);
            if (highestDepth != 1) {
                boardSorter.ensureAux(highestDepth - 1, BOARD_PRE_MAX_MALLOC_SIZES[highestDepth - 1]);
            }
        } else {
            boardSorter.ensureAux(highestDepth, BOARD_FAT_MAX_MALLOC_SIZES[highestDepth]);
            if (highestDepth != 1) {
                boardSorter.ensureAux(highestDepth - 1, BOARD_FAT_MAX_MALLOC_SIZES[highestDepth - 1]);
            }
        }
#endif
    }

    std::string getMemorySize() C {
        u64 allocMemory = 0;
        for (C auto& boardTable : board1Table) {
            if (!boardTable.empty()) {
                allocMemory += boardTable.size() * sizeof(boardTable[0]);
            }
        }
        for (C auto& boardTable : board2Table) {
            if (!boardTable.empty()) {
                allocMemory += boardTable.size() * sizeof(boardTable[0]);
            }
        }
        return bytesFormatted<1000>(allocMemory);
    }

private:
    MU static std::vector<std::string> tokenizeMoves(C std::string& line) {
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }

        return tokens;
    }

    MU static bool applyMovesAndCheckGoal(Board board,
                                          C Board& goal,
                                          C std::vector<u8>& moves) {
        for (C u8 move : moves) {
            allActStructList[move].action(board);
        }
        return board == goal;
    }

    MU static bool tryParseValidatedLine(C std::string& rawLine,
                                         C bool isFatPuzzle,
                                         std::vector<u8>& outMoves,
                                         std::string& outCanonicalLine) {
        C std::vector<std::string> inputTokens = tokenizeMoves(rawLine);
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
        C std::vector<std::string> canonicalTokens = tokenizeMoves(outCanonicalLine);

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

    MU static bool tryExpandNormalLine(C Board& start,
                                       C Board& goal,
                                       C std::string& solution,
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

        for (C auto& perm : perms) {
            if (!applyMovesAndCheckGoal(start, goal, perm)) {
                continue;
            }

            expandedSolutions.insert(moveVectorToDirectString(perm));
        }

        return true;
    }

    MU static bool tryKeepFatLine(C Board& start,
                                  C Board& goal,
                                  C std::string& solution,
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

    MU static void writeLines(C fs::path& filepath, C std::set<std::string>& lines) {
        std::ofstream file(filepath, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open output file");
        }

        for (C auto& line : lines) {
            file << line << '\n';
        }
    }

    MU void expandRawSolutionsIntoFinalSet() {
        expandedResultSet.clear();

        std::size_t validInputSolutions = 0;
        std::size_t skippedInputSolutions = 0;

        tcout << "\nExpanding solver results...\n";

        for (C auto& solution : resultSet) {
            C bool ok = hasFat
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

    MU void writeExpandedSolutions(C u32 currentDepth) {
        if (expandedResultSet.empty()) {
            tcout << "No expanded solutions produced.\n";
            return;
        }

        C std::size_t moveCount = tokenizeMoves(*expandedResultSet.begin()).size();

        fs::path outputDir = outDirectory / "levels";
        fs::create_directories(outputDir);

        C std::string filename = pair->getName()
                                 + "_c" + std::to_string(moveCount)
                                 + "_" + std::to_string(expandedResultSet.size())
                                 + ".txt";

        C fs::path filepath = outputDir / filename;
        tcout << "Saving expanded results to '" << filepath << "'.\n";
        writeLines(filepath, expandedResultSet);
    }

private:
    MU static void buildBoardExactNoneFrontier(C Board& root,
                                               C u32 depth,
                                               RecoveryBoardFrontierCache& cache) {
        cache.states.clear();
        Perms<Board>::getDepthFunc<eSequenceDir::NONE>(root, cache.states, depth, true);
        cache.sorter.sortBoards(cache.states, depth, root.getColorCount());
    }

    MU void ensureCache(RecoveryBoardFrontierCache& cache,
                        C Board& root,
                        C u32 depth) {
        if (cache.valid && cache.depth == depth && cache.root == root) {
            return;
        }

        cache.root = root;
        cache.depth = depth;
        cache.valid = true;
        buildBoardExactNoneFrontier(cache.root, cache.depth, cache);
    }

    MU static void recoverExactNormalSplit(C Board& leftRoot,
                                           C u32 leftDepth,
                                           C JVec<Board>& leftStates,
                                           C Board& rightRoot,
                                           C u32 rightDepth,
                                           C JVec<Board>& rightStates,
                                           std::set<std::string>& outPaths) {
        outPaths.clear();

        C auto matches = (leftDepth != 0 && rightDepth != 0)
                                 ? intersection_threaded(leftStates, rightStates)
                                 : intersection(leftStates, rightStates);

        for (C auto& [fst, snd] : matches) {
            C Board temp1 = makeBoardWithMoves(leftRoot, fst->memory);
            C Board temp2 = makeBoardWithMoves(rightRoot, snd->memory);

            if (temp1 == temp2) {
                outPaths.insert(fst->memory.asmString(&snd->memory));
            }
        }
    }

    MU void recoverSeedPrefixes(C Board& seedBoard,
                                C u32 prefixLeftDepth,
                                C u32 prefixRightDepth,
                                std::set<std::string>& outPrefixes) {
        outPrefixes.clear();

        if (prefixLeftDepth == 0 && prefixRightDepth == 0) {
            outPrefixes.insert("");
            return;
        }

        ensureCache(prefixLeftCache_, board1, prefixLeftDepth);

        RecoveryBoardFrontierCache seedPrefixRightCache;
        ensureCache(seedPrefixRightCache, seedBoard, prefixRightDepth);

        recoverExactNormalSplit(
                board1,
                prefixLeftDepth,
                prefixLeftCache_.states,
                seedBoard,
                prefixRightDepth,
                seedPrefixRightCache.states,
                outPrefixes
        );
    }

    MU void recoverSeedToMiddle(C Board& seedBoard,
                                C B1B2& middleState,
                                C u32 seedLeftDepth,
                                C u32 middleRightDepth,
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
                middleBoard,
                middleRightDepth,
                middleRightCache.states,
                outPaths
        );
    }

    MU void recoverMiddleToGoal(C B1B2& middleState,
                                C u32 middleLeftDepth,
                                C u32 goalRightDepth,
                                RecoveryBoardFrontierCache& middleLeftCache,
                                std::set<std::string>& outPaths) {
        Board middleBoard = makeBoardFromState(middleState);

        ensureCache(middleLeftCache, middleBoard, middleLeftDepth);
        ensureCache(goalRightCache_, board2, goalRightDepth);

        recoverExactNormalSplit(
                middleBoard,
                middleLeftDepth,
                middleLeftCache.states,
                board2,
                goalRightDepth,
                goalRightCache_.states,
                outPaths
        );
    }

    MU static void appendJoinedSolutions(C std::set<std::string>& prefixes,
                                         C std::set<std::string>& middles,
                                         C std::set<std::string>& suffixes,
                                         std::unordered_set<std::string>& outRaw) {
        for (C auto& p : prefixes) {
            for (C auto& m : middles) {
                std::string leftHalf;
                if (p.empty()) {
                    leftHalf = m;
                } else if (m.empty()) {
                    leftHalf = p;
                } else {
                    leftHalf = p + " " + m;
                }

                for (C auto& s : suffixes) {
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

public:
    template<bool debug = true>
    void findSolutionsAtDepth(C u32 index, C u32 depth1, C u32 depth2, C bool searchResults = true) {
        C std::string start_both = "[" + std::to_string(index) + "] ";
        C std::string start_left = "[" + std::to_string(index) + "L] ";
        C std::string start_right = "[" + std::to_string(index) + "R] ";

        if (board1Table[depth1].empty()) {
            IF_DEBUG_COUT(start_left << "doing getDepthFunc for " << depth1)

            C Timer timer;
            C bool should_alloc = board1Table[depth1].capacity() == 0;
            Perms<Memory>::getDepthFunc<eSequenceDir::ASCENDING>(board1, board1Table[depth1], depth1, should_alloc);

            IF_DEBUG_COUT("\n" << start_left << "Size: " << board1Table[depth1].size())
            IF_DEBUG_COUT("\n" << start_left << "Make Time: " << timer.getSeconds() << "\n")

            C Timer timerSort1;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board1Table[depth1].begin(), board1Table[depth1].end());
#else
            u8 colorCount = board1.getColorCount();
            boardSorter.sortBoards(board1Table[depth1], depth1, colorCount);
#endif
            IF_DEBUG_COUT(start_left << "Sort Time: " << timerSort1.getSeconds() << "\n\n")
        }

        if (board2Table[depth2].empty()) {
            IF_DEBUG_COUT(start_right << "doing getDepthFunc for " << depth2)

            C Timer timer;
            C bool should_alloc = board2Table[depth2].capacity() == 0;
            Perms<Memory>::getDepthFunc<eSequenceDir::DESCENDING>(board2, board2Table[depth2], depth2, should_alloc);

            IF_DEBUG_COUT("\n" << start_right << "Size: " << board2Table[depth2].size())
            IF_DEBUG_COUT("\n" << start_right << "Make Time: " << timer.getSeconds() << "\n")

            C Timer timerSort2;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board2Table[depth2].begin(), board2Table[depth2].end());
#else
            u8 colorCount = board2.getColorCount();
            boardSorter.sortBoards(board2Table[depth2], depth2, colorCount);
#endif
            IF_DEBUG_COUT(start_right << "Sort Time: " << timerSort2.getSeconds() << "\n\n")
        }

        if (searchResults) {
            IF_DEBUG_COUT(start_both << "Solving for depths [" << depth1 << ", " << depth2 << "]")

            C Timer timerInter;
            std::vector<std::pair<C Memory*, C Memory*>> results;
            if (depth1 != 0 && depth2 != 0) {
                results = intersection_threaded(board1Table[depth1], board2Table[depth2]);
            } else {
                results = intersection(board1Table[depth1], board2Table[depth2]);
            }
            C auto timerInterEnd = timerInter.getSeconds();

            IF_DEBUG_COUT(" found: " << results.size() << "\n")
            IF_DEBUG_COUT(start_both << "Inter Time: " << timerInterEnd << "\n")

            if (hasFat) {
                C i32 xy1 = board1.getFatXY();
                C i32 xy2 = board2.getFatXY();

                for (C auto& [fst, snd] : results) {
                    C Board temp1 = makeBoardWithFatMoves(board1, *fst);
                    C Board temp2 = makeBoardWithFatMoves(board2, *snd);
                    if (temp1 == temp2) {
                        std::string moveset = fst->asmFatString(xy1, snd, xy2);
                        resultSet.insert(moveset);
                    }
                }
            } else {
                for (C auto& [fst, snd] : results) {
                    C Board temp1 = makeBoardWithMoves(board1, *fst);
                    C Board temp2 = makeBoardWithMoves(board2, *snd);
                    if (temp1 == temp2) {
                        std::string moveset = fst->asmString(snd);
                        resultSet.insert(moveset);
                    }
                }
            }
        }

        tcout << std::flush;
    }

    template<bool debug = true>
    MU void findSolutions() {
        resultSet.clear();
        expandedResultSet.clear();

        u32 currentDepth = depthGuessMax;
        C Timer totalTime;

        while (currentDepth <= depthTotalMax) {
            C auto permutationsFromDepth = Perms<Memory>::depthMap.at(currentDepth);
            i32 permCount = 0;

            if (currentDepth > 1 && currentDepth % 2 == 1) {
                IF_DEBUG_COUT("\nSolving for (depth - 1): " << currentDepth - 1 << "\n\n")
                C auto oneBefore = Perms<Memory>::depthMap.at(currentDepth - 1);
                findSolutionsAtDepth<debug>(permCount, oneBefore[0].first, oneBefore[0].second, false);
            }

            for (C auto& [fst, snd] : permutationsFromDepth) {
                if (fst > depthSideMax) { continue; }
                if (snd > depthSideMax) { continue; }

                findSolutionsAtDepth<debug>(permCount, fst, snd);
                ++permCount;

                if (permCount != static_cast<i32>(permutationsFromDepth.size()) - 1) {
                    if (!resultSet.empty()) {
                        IF_DEBUG_COUT("Unique Solutions so far: " << resultSet.size() << std::endl)
                    }
                } else if (!resultSet.empty()) {
                    IF_DEBUG_COUT("Total Unique Solutions: " << resultSet.size() << std::endl)
                }
            }

            if (!resultSet.empty()) {
                break;
            }

            ++currentDepth;
        }

        tcout << "Total Time: " << totalTime.getSeconds() << std::endl;

        C std::string allocMemory = getMemorySize();
        tcout << "Alloc Memory: " << allocMemory << std::endl;

        if (!resultSet.empty()) {
            expandRawSolutionsIntoFinalSet();
            writeExpandedSolutions(currentDepth);
        } else {
            tcout << "No solutions found...\n";
        }
    }

public:
    template<int SEED_DEPTH = 1, int LEFT_FRONTIER_DEPTH = 5, int RIGHT_FRONTIER_DEPTH = 5, bool debug = true>
    MU void findSolutionsFrontier() {
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

        C Timer totalTime;

        JVec<B1B2> leftSeeds;
        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(board1, leftSeeds);
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';

        JVec<B1B2> rightFrontierStates;
        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(board2, rightFrontierStates);
        tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH << ") final unique size: "
              << rightFrontierStates.size() << '\n';

        rightFrontierIndex_.buildFromUniqueStates(std::move(rightFrontierStates));
        tcout << "right frontier buckets built for " << rightFrontierIndex_.size() << " states\n";

        JVec<B1B2> leftFrontierStates;
        JVec<B1B2> middleMatches;

        RecoveryBoardFrontierCache seedLeftCache;
        RecoveryBoardFrontierCache middleRightCache;
        RecoveryBoardFrontierCache middleLeftCache;

        std::set<std::string> prefixPaths;
        std::set<std::string> seedToMiddlePaths;
        std::set<std::string> middleToGoalPaths;

        for (std::size_t i = 0; i < leftSeeds.size(); ++i) {
            tcout << "[seed " << (i + 1) << "/" << leftSeeds.size()
                  << "] generating left frontier +" << LEFT_FRONTIER_DEPTH
                  << " from seed(" << SEED_DEPTH << ")\n" << std::flush;

            Board seedBoard = makeBoardFromState(leftSeeds[i]);

            leftFrontierStates.clear();
            Perms<B1B2>::getDepthFunc<eSequenceDir::ASCENDING>(
                    seedBoard,
                    leftFrontierStates,
                    LEFT_FRONTIER_DEPTH,
                    true
            );

            tcout << "    produced left frontier states: " << leftFrontierStates.size() << '\n';

            rightFrontierIndex_.collectMatches(leftFrontierStates, middleMatches);
            tcout << "    middle matches: " << middleMatches.size() << '\n';

            if (middleMatches.empty()) {
                tcout << "    unique raw solutions so far: " << resultSet.size() << '\n';
                continue;
            }

            prefixPaths.clear();
            if constexpr (SEED_DEPTH == 0) {
                prefixPaths.insert("");
            } else {
                recoverSeedPrefixes(
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

                recoverMiddleToGoal(
                        middleMatches[m],
                        GOAL_LEFT_DEPTH,
                        GOAL_RIGHT_DEPTH,
                        middleLeftCache,
                        middleToGoalPaths
                );

                if (middleToGoalPaths.empty()) {
                    continue;
                }

                appendJoinedSolutions(
                        prefixPaths,
                        seedToMiddlePaths,
                        middleToGoalPaths,
                        resultSet
                );
            }

            tcout << "    unique raw solutions so far: " << resultSet.size() << '\n';
        }

        tcout << "\nTotal Time: " << totalTime.getSeconds() << '\n';
        tcout << "Seed depth: " << SEED_DEPTH << '\n';
        tcout << "Left frontier depth: " << LEFT_FRONTIER_DEPTH << '\n';
        tcout << "Right frontier depth: " << RIGHT_FRONTIER_DEPTH << '\n';

        if (!resultSet.empty()) {
            expandRawSolutionsIntoFinalSet();
            writeExpandedSolutions(TOTAL_DEPTH);
        } else {
            tcout << "No solutions found...\n";
        }
    }

    template<int SEED_DEPTH = 1, int LEFT_FRONTIER_DEPTH = 5, int RIGHT_FRONTIER_DEPTH = 5, bool debug = true>
    MU void findSolutionsFrontierThreaded() {
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

        prefixLeftCache_.valid = false;
        goalRightCache_.valid = false;
        rightFrontierIndex_.clear();

        C Timer totalTime;

        JVec<B1B2> leftSeeds;
        buildUniqueNoneDepthFrontierB1B2<SEED_DEPTH>(board1, leftSeeds);
        tcout << "seed(" << SEED_DEPTH << ") final unique size: " << leftSeeds.size() << '\n';

        JVec<B1B2> rightFrontierStates;
        buildUniqueNoneDepthFrontierB1B2<RIGHT_FRONTIER_DEPTH>(board2, rightFrontierStates);
        tcout << "right frontier(" << RIGHT_FRONTIER_DEPTH << ") final unique size: "
              << rightFrontierStates.size() << '\n';

        rightFrontierIndex_.buildFromUniqueStates(std::move(rightFrontierStates));
        tcout << "right frontier buckets built for " << rightFrontierIndex_.size() << " states\n";

        if constexpr (SEED_DEPTH != 0) {
            ensureCache(prefixLeftCache_, board1, PREFIX_LEFT_DEPTH);
        }
        ensureCache(goalRightCache_, board2, GOAL_RIGHT_DEPTH);

        std::atomic<std::size_t> nextIndex = 0;
        std::mutex printMutex;
        std::mutex resultMutex;

        static constexpr std::size_t WORKER_COUNT = 2;

        auto worker = [&](const std::size_t workerId) {
            JVec<B1B2> leftFrontierStates;
            JVec<B1B2> middleMatches;

            RecoveryBoardFrontierCache seedLeftCache;
            RecoveryBoardFrontierCache seedPrefixRightCache;
            RecoveryBoardFrontierCache middleRightCache;
            RecoveryBoardFrontierCache middleLeftCache;

            std::set<std::string> prefixPaths;
            std::set<std::string> seedToMiddlePaths;
            std::set<std::string> middleToGoalPaths;
            std::unordered_set<std::string> localRecovered;

            while (true) {
                C std::size_t i = nextIndex.fetch_add(1);
                if (i >= leftSeeds.size()) {
                    break;
                }

                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "[worker " << workerId
                          << "] [seed " << (i + 1) << "/" << leftSeeds.size()
                          << "] generating left frontier +" << LEFT_FRONTIER_DEPTH
                          << " from seed(" << SEED_DEPTH << ")\n" << std::flush;
                }

                Board seedBoard = makeBoardFromState(leftSeeds[i]);

                leftFrontierStates.clear();
                Perms<B1B2>::getDepthFunc<eSequenceDir::ASCENDING>(
                        seedBoard,
                        leftFrontierStates,
                        LEFT_FRONTIER_DEPTH,
                        true
                );

                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "    [worker " << workerId
                          << "] produced left frontier states: " << leftFrontierStates.size() << '\n';
                }

                rightFrontierIndex_.collectMatches(leftFrontierStates, middleMatches);

                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "    [worker " << workerId
                          << "] middle matches: " << middleMatches.size() << '\n';
                }

                if (middleMatches.empty()) {
                    continue;
                }

                prefixPaths.clear();
                if constexpr (SEED_DEPTH == 0) {
                    prefixPaths.insert("");
                } else {
                    ensureCache(seedPrefixRightCache, seedBoard, PREFIX_RIGHT_DEPTH);

                    recoverExactNormalSplit(
                            board1,
                            PREFIX_LEFT_DEPTH,
                            prefixLeftCache_.states,
                            seedBoard,
                            PREFIX_RIGHT_DEPTH,
                            seedPrefixRightCache.states,
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

                    recoverMiddleToGoal(
                            middleMatches[m],
                            GOAL_LEFT_DEPTH,
                            GOAL_RIGHT_DEPTH,
                            middleLeftCache,
                            middleToGoalPaths
                    );

                    if (middleToGoalPaths.empty()) {
                        continue;
                    }

                    appendJoinedSolutions(
                            prefixPaths,
                            seedToMiddlePaths,
                            middleToGoalPaths,
                            localRecovered
                    );
                }

                {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    for (C auto& s : localRecovered) {
                        resultSet.insert(s);
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    tcout << "    [worker " << workerId
                          << "] unique raw solutions so far: " << resultSet.size() << '\n';
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(WORKER_COUNT);

        for (std::size_t workerId = 0; workerId < WORKER_COUNT; ++workerId) {
            workers.emplace_back(worker, workerId);
        }

        for (auto& t : workers) {
            t.join();
        }

        tcout << "\nTotal Time: " << totalTime.getSeconds() << '\n';
        tcout << "Seed depth: " << SEED_DEPTH << '\n';
        tcout << "Left frontier depth: " << LEFT_FRONTIER_DEPTH << '\n';
        tcout << "Right frontier depth: " << RIGHT_FRONTIER_DEPTH << '\n';

        if (!resultSet.empty()) {
            expandRawSolutionsIntoFinalSet();
            writeExpandedSolutions(TOTAL_DEPTH);
        } else {
            tcout << "No solutions found...\n";
        }
    }
};