#pragma once

#ifdef BOOST_FOUND
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#endif
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "MindbenderSolver/utils/format_bytes.hpp"

#include "MindbenderSolver/utils/timer.hpp"
#include "intersection.hpp"
#include "levels.hpp"
#include "perms.hpp"

#include "sorter.hpp"



#define IF_DEBUG_COUT(stuff) if constexpr (debug) { std::cout << stuff }


class MU BoardSolver {
public:
    static constexpr u32 MAX_DEPTH = 5;

    std::vector<JVec<Memory>> board1Table;
    std::vector<JVec<Memory>> board2Table;
    std::unordered_set<std::string> resultSet;
#ifdef BOOST_FOUND
#else
    BoardSorter<Memory> boardSorter;
#endif
    C BoardPair* pair;
    Board board1;
    Board board2;
    bool hasFat;
    std::string outDirectory;


    u32 depthSideMax{};
    u32 depthTotalMax{};
    u32 depthGuessMax{};



    MU explicit BoardSolver(C BoardPair* pairIn) {
        pair = pairIn;
        board1 = pair->getStartState();
        board2 = pair->getEndState();
        hasFat = board1.getFatBool();

    }


    MU void setDepthParams(C u32 depthSideMaxIn, C u32 depthGuessMaxIn, C u32 depthTotalMaxIn) {
        depthSideMax = depthSideMaxIn;
        depthTotalMax = depthTotalMaxIn;
        depthGuessMax = depthGuessMaxIn;

        board1Table.resize(depthSideMax + 1);
        board2Table.resize(depthSideMax + 1);
    }


    MU void setWriteDirectory(C std::string& directory) {
        outDirectory = directory;
    }


    MU void preAllocateMemory(C u32 maxDepth = MAX_DEPTH) {
        C u32 highestDepth = std::max(1U, std::min(maxDepth, depthTotalMax + 1) / 2);
        Perms<Memory>::reserveForDepth(board1, board1Table[highestDepth], highestDepth);
        Perms<Memory>::reserveForDepth(board1, board1Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms<Memory>::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1);
            Perms<Memory>::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1);
        }

        Perms<Memory>::reserveForDepth(board2, board2Table[highestDepth], highestDepth);
        Perms<Memory>::reserveForDepth(board2, board2Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms<Memory>::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1);
            Perms<Memory>::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1);
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
        for (C auto& boardTable : board1Table)
            allocMemory += boardTable.size() * sizeof(boardTable[0]);
        for (C auto& boardTable : board2Table)
            allocMemory += boardTable.size() * sizeof(boardTable[0]);
        return bytesFormatted<1000>(allocMemory);
    }


    template<bool debug=true>
    void findSolutionsAtDepth(C u32 index, C u32 depth1, C u32 depth2, C bool searchResults = true) {
        C std::string start_both = "[" + std::to_string(index) + "] ";
        C std::string start_left = "[" + std::to_string(index) + "L] ";
        C std::string start_right = "[" + std::to_string(index) + "R] ";

        if (board1Table[depth1].empty()) {
            IF_DEBUG_COUT(start_left<<"doing getDepthFunc for "<<depth1;)

            C Timer timer;
            C bool should_alloc = board1Table[depth1].capacity() == 0;
            Perms<Memory>::getDepthFunc<true>(board1, board1Table[depth1], depth1, should_alloc);

            IF_DEBUG_COUT("\n"<<start_left<<"Size: "<<board1Table[depth1].size();)
            IF_DEBUG_COUT("\n"<<start_left<<"Make Time: "<<timer.getSeconds()<<"\n";)

            C Timer timerSort1;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board1Table[depth1].begin(), board1Table[depth1].end());
#else
            u8 colorCount = board1.getColorCount();
            boardSorter.sortBoards(board1Table[depth1], depth1, colorCount);
#endif
            IF_DEBUG_COUT(start_left<<"Sort Time: "<<timerSort1.getSeconds()<<"\n\n";)
        }


        if (board2Table[depth2].empty()) {
            IF_DEBUG_COUT(start_right<<"doing getDepthFunc for "<<depth2;)

            C Timer timer;
            C bool should_alloc = board2Table[depth2].capacity() == 0;
            Perms<Memory>::getDepthFunc<false>(board2, board2Table[depth2], depth2, should_alloc);

            IF_DEBUG_COUT("\n"<<start_right<<"Size: "<<board2Table[depth2].size();)
            IF_DEBUG_COUT("\n"<<start_right<<"Make Time: "<<timer.getSeconds()<<"\n";)

            C Timer timerSort2;
#ifdef BOOST_FOUND
            boost::sort::block_indirect_sort(board2Table[depth2].begin(), board2Table[depth2].end());
#else
            u8 colorCount = board2.getColorCount();
            boardSorter.sortBoards(board2Table[depth2], depth2, colorCount);
#endif
            IF_DEBUG_COUT(start_right<<"Sort Time: "<<timerSort2.getSeconds()<<"\n\n";)
        }


        if (searchResults) {
            IF_DEBUG_COUT(start_both<<"Solving for depths ["<<depth1<<", "<<depth2<<"]";)

            C Timer timerInter;
            std::vector<std::pair<C Memory*, C Memory*>> results;
            if (depth1 != 0 && depth2 != 0) {
                results = intersection_threaded(board1Table[depth1], board2Table[depth2]);
            } else {
                results = intersection(board1Table[depth1], board2Table[depth2]);
            }
            auto timerInterEnd = timerInter.getSeconds();

            IF_DEBUG_COUT(" found: " << results.size() << "\n";)
            IF_DEBUG_COUT(start_both<<"Inter Time: "<<timerInterEnd<<"\n";)

            // verify the results
            // this filters out board states with identical hashes
            if (hasFat) {
                C int xy1 = board1.getFatXY();
                C int xy2 = board2.getFatXY();
                for (C auto &[fst, snd]: results) {
                    C Board temp1 = makeBoardWithFatMoves(board1, *fst);
                    C Board temp2 = makeBoardWithFatMoves(board2, *snd);
                    if (temp1 == temp2) {
                        std::string moveset = fst->asmFatString(xy1, snd, xy2);
                        resultSet.insert(moveset);
                    }
                }
            } else {
                for (auto& [fst, snd]: results) {
                    C Board temp1 = makeBoardWithMoves(board1, *fst);
                    C Board temp2 = makeBoardWithMoves(board2, *snd);
                    if (temp1 == temp2) {
                        std::string moveset = fst->asmString(snd);
                        resultSet.insert(moveset);
                    }
                }
            }
        }

        std::cout << std::flush;
    }


    template<bool debug=true>
    MU void findSolutions() {

        u32 currentDepth = depthGuessMax;
        C Timer totalTime;
        while (currentDepth <= depthTotalMax) {
            auto permutationsFromDepth = Perms<Memory>::depthMap.at(currentDepth);
            int permCount = 0;

            // if depth == 9, pre-calculate (4, 4) ex.
            if (currentDepth > 1 && currentDepth % 2 == 1) {
                IF_DEBUG_COUT("\nSolving for (depth - 1): "<<currentDepth - 1<<"\n\n";)
                auto oneBefore = Perms<Memory>::depthMap.at(currentDepth - 1);
                findSolutionsAtDepth<debug>(permCount, oneBefore[0].first, oneBefore[0].second, false);
            }

            for (C auto &[fst, snd] : permutationsFromDepth) {
                if (fst > depthSideMax) { continue; }
                if (snd > depthSideMax) { continue; }

                findSolutionsAtDepth<debug>(permCount, fst, snd);
                permCount++;
                if (permCount != permutationsFromDepth.size() - 1) {
                    if (!resultSet.empty()) {
                        IF_DEBUG_COUT("Unique Solutions so far: "<<resultSet.size()<<std::endl;)
                    }
                } else if (!resultSet.empty()){
                    IF_DEBUG_COUT("Total Unique Solutions: "<<resultSet.size()<<std::endl;)
                }
            }
            if (!resultSet.empty()) { break; }
            currentDepth++;
        }
        std::cout<<"Total Time: "<<totalTime.getSeconds()<<std::endl;

        C std::string allocMemory = getMemorySize();
        std::cout<<"Alloc Memory: "<<allocMemory<<std::endl;

        if (!resultSet.empty()) {
            C std::string filename = pair->getName()
                                         + "_c" + std::to_string(currentDepth)
                                         + "_" + std::to_string(resultSet.size())
                                         + ".txt";
            std::cout<<"Saving results to '"<<filename<<"'.\n";
            std::ofstream outfile(outDirectory + "\\levels\\" + filename);
            for (C auto& str: resultSet) {
                outfile << str << std::endl;
            }
            outfile.close();
        } else {
            std::cout<<"No solutions found...\n";
        }
    }

};