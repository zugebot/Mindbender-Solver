#pragma once
// code/solver_direct.hpp

#ifdef BOOST_FOUND
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#endif

#include "solver_base.hpp"
#include "intersection.hpp"

#define IF_DEBUG_COUT(stuff) if constexpr (debug) { tcout << stuff; }

class MU BoardSolverDirect : public BoardSolverBase {
public:
    using BoardSolverBase::BoardSolverBase;

    template<bool debug = true>
    void findSolutionsAtDepth(C u32 index, C u32 depth1, C u32 depth2, C bool searchResults = true) {
        validateStateHashLanes(board1Table[depth1], board1HashesTable[depth1], "board1Table");
        validateStateHashLanes(board2Table[depth2], board2HashesTable[depth2], "board2Table");

        C std::string start_both = "[" + std::to_string(index) + "] ";
        C std::string start_left = "[" + std::to_string(index) + "L] ";
        C std::string start_right = "[" + std::to_string(index) + "R] ";

        if (board1Table[depth1].empty()) {
            IF_DEBUG_COUT(start_left << "doing getDepthFunc for " << depth1)

            C Timer timer;
            C bool should_alloc = board1Table[depth1].capacity() == 0;

            Perms<Board>::getDepthFunc<eSequenceDir::ASCENDING>(
                    board1,
                    board1Table[depth1],
                    board1HashesTable[depth1],
                    depth1,
                    should_alloc
            );

            IF_DEBUG_COUT("\n" << start_left << "Size: " << board1Table[depth1].size())
            IF_DEBUG_COUT("\n" << start_left << "Make Time: " << timer.getSeconds() << "\n")

            C Timer timerSort1;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board1Table[depth1].begin(), board1Table[depth1].end());
#else
            u8 colorCount = board1.getColorCount();
            boardSorter.sortBoards(
                    board1Table[depth1],
                    board1HashesTable[depth1],
                    depth1,
                    colorCount
            );
#endif
            IF_DEBUG_COUT(start_left << "Sort Time: " << timerSort1.getSeconds() << "\n\n")
        }

        if (board2Table[depth2].empty()) {
            IF_DEBUG_COUT(start_right << "doing getDepthFunc for " << depth2)

            C Timer timer;
            C bool should_alloc = board2Table[depth2].capacity() == 0;

            Perms<Board>::getDepthFunc<eSequenceDir::DESCENDING>(
                    board2,
                    board2Table[depth2],
                    board2HashesTable[depth2],
                    depth2,
                    should_alloc
            );

            IF_DEBUG_COUT("\n" << start_right << "Size: " << board2Table[depth2].size())
            IF_DEBUG_COUT("\n" << start_right << "Make Time: " << timer.getSeconds() << "\n")

            C Timer timerSort2;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board2Table[depth2].begin(), board2Table[depth2].end());
#else
            u8 colorCount = board2.getColorCount();
            boardSorter.sortBoards(
                    board2Table[depth2],
                    board2HashesTable[depth2],
                    depth2,
                    colorCount
            );
#endif
            IF_DEBUG_COUT(start_right << "Sort Time: " << timerSort2.getSeconds() << "\n\n")
        }

        if (searchResults) {
            IF_DEBUG_COUT(start_both << "Solving for depths [" << depth1 << ", " << depth2 << "]")

            C Timer timerInter;
            std::vector<std::pair<C Board*, C Board*>> results;

            if (depth1 != 0 && depth2 != 0) {
                results = intersection_threaded(
                        board1Table[depth1],
                        board1HashesTable[depth1],
                        board2Table[depth2],
                        board2HashesTable[depth2]
                );
            } else {
                results = intersection(
                        board1Table[depth1],
                        board1HashesTable[depth1],
                        board2Table[depth2],
                        board2HashesTable[depth2]
                );
            }

            C auto timerInterEnd = timerInter.getSeconds();

            IF_DEBUG_COUT(" found: " << results.size() << "\n")
            IF_DEBUG_COUT(start_both << "Inter Time: " << timerInterEnd << "\n")

            if (hasFat) {
                C i32 xy1 = board1.getFatXY();
                C i32 xy2 = board2.getFatXY();

                for (C auto& [fst, snd] : results) {
                    C Board temp1 = makeBoardWithFatMoves(board1, fst->memory);
                    C Board temp2 = makeBoardWithFatMoves(board2, snd->memory);
                    if (temp1 == temp2) {
                        std::string moveset = fst->memory.asmFatString(xy1, &snd->memory, xy2);
                        resultSet.insert(moveset);
                    }
                }
            } else {
                for (C auto& [fst, snd] : results) {
                    C Board temp1 = makeBoardWithMoves(board1, fst->memory);
                    C Board temp2 = makeBoardWithMoves(board2, snd->memory);
                    if (temp1 == temp2) {
                        std::string moveset = fst->memory.asmString(&snd->memory);
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
            C auto permutationsFromDepth = Perms<Board>::depthMap.at(currentDepth);
            i32 permCount = 0;

            if (currentDepth > 1 && currentDepth % 2 == 1) {
                IF_DEBUG_COUT("\nSolving for (depth - 1): " << currentDepth - 1 << "\n\n")
                C auto oneBefore = Perms<Board>::depthMap.at(currentDepth - 1);
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
};