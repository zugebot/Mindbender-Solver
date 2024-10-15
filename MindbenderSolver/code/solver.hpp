#pragma once

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "MindbenderSolver/utils/format_bytes.hpp"
#include "intersection.hpp"
#include "levels.hpp"
#include "perms.hpp"
#include "sorter.hpp"


#define IF_DEBUG(stuff) if constexpr (debug) { stuff }
#define IF_DEBUG_COUT(stuff) if constexpr (debug) { std::cout << stuff }


class BoardSolver {
public:
    std::vector<std::vector<HashMem>> board1Table;
    std::vector<std::vector<HashMem>> board2Table;
    std::unordered_set<std::string> resultSet;
    BoardSorter<HashMem> boardSorter;
    const BoardPair* pair;
    Board board1;
    Board board2;
    bool hasFat;
    std::string outDirectory;


    u32 depthSideMax{};
    u32 depthTotalMax{};
    u32 depthGuessMax{};



    explicit BoardSolver(const BoardPair* pairIn) {
        pair = pairIn;
        board1 = pair->getInitialState();
        board2 = pair->getSolutionState();
        hasFat = board1.getFatBool();

    }


    void setDepthParams(c_u32 depthSideMaxIn, c_u32 depthGuessMaxIn, c_u32 depthTotalMaxIn) {
        depthSideMax = depthSideMaxIn;
        depthTotalMax = depthTotalMaxIn;
        depthGuessMax = depthGuessMaxIn;

        board1Table.resize(depthSideMax + 1);
        board2Table.resize(depthSideMax + 1);
    }


    void setWriteDirectory(const std::string& directory) {
        outDirectory = directory;
    }


    void preAllocateMemory(c_u32 maxDepth = 5) {
        c_u32 highestDepth = std::max(1U, std::min(maxDepth, depthTotalMax + 1) / 2);
        Perms::reserveForDepth(board1, board1Table[highestDepth], highestDepth);
        Perms::reserveForDepth(board1, board1Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1);
            Perms::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1);
        }

        Perms::reserveForDepth(board2, board2Table[highestDepth], highestDepth);
        Perms::reserveForDepth(board2, board2Table[highestDepth], highestDepth);

        if (highestDepth != 1) {
            Perms::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1);
            Perms::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1);
        }

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
    }


    std::string getMemorySize() const {
        u64 allocMemory = 0;
        for (const auto& boardTable : board1Table)
            allocMemory += boardTable.size() * sizeof(boardTable[0]);
        for (const auto& boardTable : board2Table)
            allocMemory += boardTable.size() * sizeof(boardTable[0]);
        return bytesFormatted<1000>(allocMemory);
    }


    template<bool debug=true>
    void findSolutionsAtDepth(c_u32 index, c_u32 depth1, c_u32 depth2, c_bool searchResults = true) {
        const std::string start_both = "[" + std::to_string(index) + "] ";
        const std::string start_left = "[" + std::to_string(index) + "L] ";
        const std::string start_right = "[" + std::to_string(index) + "R] ";

        c_u32 colorCount = !hasFat ? board1.getColorCount() : 4;

        if (board1Table[depth1].empty()) {
            IF_DEBUG_COUT(start_left<<"doing getDepthFunc for "<<depth1;)

            const Timer timer;
            c_bool should_alloc = board1Table[depth1].capacity() == 0;
            Perms::getDepthFunc<true>(board1, board1Table[depth1], depth1, should_alloc);

            IF_DEBUG_COUT("\n"<<start_left<<"Creation Time: "<<timer.getSeconds();)
            IF_DEBUG_COUT("\n"<<start_left<<"Size: "<<board1Table[depth1].size() << "\n";)

            const Timer timerSort1;
            boardSorter.sortBoards(board1Table[depth1], depth1, colorCount);
            IF_DEBUG(std::cout<<start_left<<"Sort Time: "<<timerSort1.getSeconds()<<"\n\n";)
        }


        if (board2Table[depth2].empty()) {
            IF_DEBUG_COUT("\n"<<start_right<<"doing getDepthFunc for "<<depth2;)

            const Timer timer;
            c_bool should_alloc = board2Table[depth2].capacity() == 0;
            Perms::getDepthFunc<false>(board2, board2Table[depth2], depth2, should_alloc);

            IF_DEBUG_COUT("\n"<<start_right<<"Creation Time: "<<timer.getSeconds();)
            IF_DEBUG_COUT("\n"<<start_right<<"Size: "<<board2Table[depth2].size()<<"\n";)

            const Timer timerSort2;
            boardSorter.sortBoards(board2Table[depth2], depth2, colorCount);
            IF_DEBUG_COUT(start_right<<"Sort Time: "<<timerSort2.getSeconds()<<"\n\n";)
        }


        if (searchResults) {
            IF_DEBUG_COUT(start_both<<"Solving for depths ["<<depth1<<", "<<depth2<<"]";)

            std::vector<std::pair<HashMem*, HashMem*>> results;
            if (depth1 != 0 && depth2 != 0) {
                results = intersection_threaded(board1Table[depth1], board2Table[depth2]);
            } else {
                results = intersection(board1Table[depth1], board2Table[depth2]);
            }

            IF_DEBUG(if (!results.empty()) {
                std::cout << " found: " << results.size() << "\n";
            } else {
                std::cout << " found: 0\n";
            })

            if (hasFat) {
                c_int xy1 = board1.getFatXY();
                c_int xy2 = board2.getFatXY();

                for (const auto &[fst, snd]: results) {

                    Board temp1 = board1;
                    applyMoves(temp1, *fst);

                    Board temp2 = board2;
                    applyMoves(temp2, *snd);

                    // if (temp1 == temp2) {
                        std::string moveset = fst->getMemoryConst(
                            ).asmFatString(xy1, &snd->getMemoryConst(), xy2);
                        resultSet.insert(moveset);
                    // }
                }
            } else {

                // verify the results
                // this filters out board states with identical hashes
                for (auto& [fst, snd]: results) {

                    Board temp1 = board1;
                    applyMoves(temp1, *fst);

                    Board temp2 = board2;
                    applyMoves(temp2, *snd);

                    if (temp1 == temp2) {
                        std::string moveset = fst->getMemoryConst(
                            ).asmString(&snd->getMemoryConst());
                        resultSet.insert(moveset);
                    }
                }
            }
        }

        std::cout << std::flush;
    }


    template<bool debug=true>
    void findSolutions() {

        u32 currentDepth = depthGuessMax;
        const Timer totalTime;
        while (currentDepth <= depthTotalMax) {
            auto permutationsFromDepth = Perms::depthMap.at(currentDepth);
            int permCount = 0;

            // if depth == 9, pre-calculate (4, 4) ex.
            if (currentDepth > 1 && currentDepth % 2 == 1) {
                IF_DEBUG_COUT("\nSolving for (depth - 1): "<<currentDepth - 1<<"\n\n";)
                auto oneBefore = Perms::depthMap.at(currentDepth - 1);
                findSolutionsAtDepth<debug>(permCount, oneBefore[0].first, oneBefore[0].second, false);
            }

            for (const auto &[fst, snd] : permutationsFromDepth) {
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

        const std::string allocMemory = getMemorySize();
        std::cout<<"Alloc Memory: "<<allocMemory<<std::endl;

        if (!resultSet.empty()) {
            const std::string filename = pair->getName()
                                   + "_c" + std::to_string(currentDepth)
                                   + "_" + std::to_string(resultSet.size())
                                   + ".txt";
            std::cout<<"Saving results to '"<<filename<<"'.\n";
            std::ofstream outfile(outDirectory + "\\levels\\" + filename);
            for (const auto& str: resultSet) {
                outfile << str << std::endl;
            }
            outfile.close();
        } else {
            std::cout<<"No solutions found...\n";
        }
    }

};