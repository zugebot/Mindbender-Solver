

#include "levels.hpp"
#include "perms.hpp"
#include "sorter.hpp"
#include "intersection.hpp"

#include <fstream>
#include <unordered_set>
#include <string>
#include <vector>


class BoardSolver {
public:
    std::vector<std::vector<Board>> board1Table;
    std::vector<std::vector<Board>> board2Table;
    std::unordered_set<std::string> resultSet;
    BoardSorter boardSorter;
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
        hasFat = board1.hasFat();

    }


    void setDepthParams(u32 depthSideMaxIn, u32 depthGuessMaxIn, u32 depthTotalMaxIn) {
        depthSideMax = depthSideMaxIn;
        depthTotalMax = depthTotalMaxIn;
        depthGuessMax = depthGuessMaxIn;

        board1Table.resize(depthSideMax + 1);
        board2Table.resize(depthSideMax + 1);
    }


    void setWriteDirectory(const std::string& directory) {
        outDirectory = directory;
    }


    void preAllocateMemory() {
        u32 highestDepth = std::max(1, std::min(5, (int)(depthTotalMax + 1) / 2));
        Permutations::reserveForDepth(board1, board1Table[highestDepth], highestDepth, hasFat);
        Permutations::reserveForDepth(board1, board1Table[highestDepth], highestDepth, hasFat);

        if (highestDepth != 1) {
            Permutations::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1, hasFat);
            Permutations::reserveForDepth(board1, board1Table[highestDepth - 1], highestDepth - 1, hasFat);
        }

        Permutations::reserveForDepth(board2, board2Table[highestDepth], highestDepth, hasFat);
        Permutations::reserveForDepth(board2, board2Table[highestDepth], highestDepth, hasFat);

        if (highestDepth != 1) {
            Permutations::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1, hasFat);
            Permutations::reserveForDepth(board2, board2Table[highestDepth - 1], highestDepth - 1, hasFat);
        }

        if (!hasFat) {
            boardSorter.ensureAux(highestDepth, BOARD_PRE_ALLOC_SIZES[highestDepth]);
            if (highestDepth != 1) {
                boardSorter.ensureAux(highestDepth - 1, BOARD_PRE_ALLOC_SIZES[highestDepth - 1]);
            }

        } else {
            boardSorter.ensureAux(highestDepth, BOARD_FAT_PRE_ALLOC_SIZES[highestDepth]);
            if (highestDepth != 1) {
                boardSorter.ensureAux(highestDepth - 1, BOARD_FAT_PRE_ALLOC_SIZES[highestDepth - 1]);
            }
        }
    }


#define IF_DEBUG(stuff) if constexpr (debug) { stuff }


    template<bool allowGetDepthPlus1 = true, bool debug=true>
    void findSolutionsAtDepth(u32 index, c_u32 depth1, c_u32 depth2, bool searchResults = true) {
        std::string start_both = "[" + std::to_string(index) + "] ";
        std::string start_left = "[" + std::to_string(index) + "L] ";
        std::string start_right = "[" + std::to_string(index) + "R] ";

        uint32_t colorCount = board1.getColorCount();
        if (hasFat) {
            colorCount = 4;
        }

        if (board1Table[depth1].empty()) {
            bool should_alloc = board1Table[depth1].capacity() == 0;
            const Timer timer;

            if (allowGetDepthPlus1 && depth1 > 0 && !board1Table[depth1 - 1].empty() && !hasFat) {
                IF_DEBUG(std::cout << start_left << "doing getDepthPlus1Func for " << depth1;)
                Permutations::getDepthPlus1Func(board1Table[depth1 - 1], board1Table[depth1], should_alloc);
            } else {
                IF_DEBUG(std::cout << start_left << "doing getDepthFunc for " << depth1;)
                Permutations::getDepthFunc(board1, board1Table[depth1], depth1, should_alloc);
            }

            IF_DEBUG(std::cout << "\n" << start_left << "Creation Time: " << timer.getSeconds();)
            IF_DEBUG(std::cout << "\n" << start_left << "Size: " << board1Table[depth1].size() << "\n";)

            const Timer timerSort1;
            boardSorter.sortBoards(board1Table[depth1], depth1, colorCount);
            IF_DEBUG(std::cout << start_left << "Sort Time: " << timerSort1.getSeconds() << "\n\n";)
        }


        if (board2Table[depth2].empty()) {
            bool should_alloc = board2Table[depth2].capacity() == 0;
            const Timer timer;

            if (allowGetDepthPlus1 && depth2 > 0 && !board2Table[depth2 - 1].empty() && !hasFat) {
                IF_DEBUG(std::cout  << "\n" << start_right << "doing getDepthPlus1Func for " << depth2;)
                Permutations::getDepthPlus1Func(board2Table[depth2 - 1], board2Table[depth2], should_alloc);
            } else {
                IF_DEBUG(std::cout << "\n" << start_right << "doing getDepthFunc for " << depth2;)
                Permutations::getDepthFunc(board2, board2Table[depth2], depth2, should_alloc);
            }

            IF_DEBUG(std::cout << "\n" << start_right << "Creation Time: " << timer.getSeconds();)
            IF_DEBUG(std::cout << "\n" << start_right << "Size: " << board2Table[depth2].size() << "\n";)

            const Timer timerSort2;
            boardSorter.sortBoards(board2Table[depth2], depth2, colorCount);
            IF_DEBUG(std::cout << start_right << "Sort Time: " << timerSort2.getSeconds() << "\n\n";)
        }



        if (searchResults) {
            IF_DEBUG(std::cout << start_both << "Solving for depths [" << depth1 << ", " << depth2 << "]";)

            std::vector<std::pair<Board*, Board*>> results;
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
                int xy1 = board1.getFatXY();
                int xy2 = board2.getFatXY();
                for (auto result_pair: results) {
                    std::string moveset = result_pair.first->mem.assembleFatMoveString(xy1, &result_pair.second->mem, xy2);
                    resultSet.insert(moveset);
                }
            } else {
                for (auto result_pair: results) {
                    std::string moveset = result_pair.first->mem.assembleMoveString(&result_pair.second->mem);
                    resultSet.insert(moveset);
                }
            }
        }

        std::cout << std::flush;
    }


    template<bool allowGetDepthPlus1 = true, bool debug=true>
    void findSolutions() {

        u32 currentDepth = depthGuessMax;
        const Timer totalTime;
        while (currentDepth <= depthTotalMax) {
            auto permutationsFromDepth = Permutations::depthMap.at(currentDepth);
            int permCount = 0;


            // if depth == 9, pre-calculate (4, 4) ex.
            IF_DEBUG(std::cout << "\nSolving for (depth - 1): " << currentDepth - 1 << "\n\n";)
            if (currentDepth > 1 && currentDepth % 2 == 1) {
                auto oneBefore = Permutations::depthMap.at(currentDepth - 1);
                findSolutionsAtDepth<allowGetDepthPlus1, debug>(permCount, oneBefore[0].first, oneBefore[0].second, false);
            }



            for (const auto& permPair : permutationsFromDepth) {
                if (permPair.first > depthSideMax) { continue; }
                if (permPair.second > depthSideMax) { continue; }

                findSolutionsAtDepth<allowGetDepthPlus1, debug>(permCount, permPair.first, permPair.second);
                permCount++;
                if (permCount != permutationsFromDepth.size() - 1) {
                    if (!resultSet.empty()) {
                        IF_DEBUG(std::cout << "Unique Solutions so far: " << resultSet.size() << std::endl;)
                    }
                } else if (!resultSet.empty()){
                    IF_DEBUG(std::cout << "Total Unique Solutions: " << resultSet.size() << std::endl;)
                }
            }
            if (!resultSet.empty()) {
                break;
            }
            currentDepth++;
        }
        std::cout << "Total Time: " << totalTime.getSeconds() << std::endl;


        if (!resultSet.empty()) {
            std::string filename = pair->getName()
                                   + "_c" + std::to_string(currentDepth)
                                   + "_" + std::to_string(resultSet.size())
                                   + ".txt";
            std::cout << "Saving results to '" << filename << "'.\n";
            std::ofstream outfile(outDirectory + "\\levels\\" + filename);
            for (const auto& str: resultSet) {
                outfile << str << std::endl;
            }
            outfile.close();
        } else {
            std::cout << "No solutions found...\n";
        }

    }




};


