#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/memory.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/sorter.hpp"
#include "MindbenderSolver/code/intersection.hpp"

#include "MindbenderSolver/code/rotations.hpp"
#include "MindbenderSolver/utils/timer.hpp"


#include <fstream>
#include <iostream>
#include <set>
#include <vector>



void findSolutions(std::set<std::string>& resultSet, int index,
                   BoardSorter& boardSorter,
                   std::vector<std::vector<Board>>& board1Table,
                   std::vector<std::vector<Board>>& board2Table,
                   Board board1, Board board2, const int depth1, const int depth2) {
    std::string start_both = "[" + std::to_string(index) + "] ";
    std::string start_left = "[" + std::to_string(index) + "L] ";
    std::string start_right = "[" + std::to_string(index) + "R] ";

    uint32_t colorCount = board1.getColorCount();
    if (board1.hasFat()) {
        colorCount = 4;
    }

    auto permFuncs = !board1.hasFat() ? makePermutationListFuncs : makeFatPermutationListFuncs;

    std::cout << "\n";
    std::cout << start_both << "Solving for depths [" << depth1 << ", " << depth2 << "]" << std::endl;
    while (board1Table.size() <= depth1) {
        const Timer timer1;
        const int tempDepth = (int)board1Table.size();
        std::vector<Board> boards = (permFuncs[tempDepth])(board1, colorCount);
        std::cout << start_left << "Creation Time: " << timer1.getSeconds() << "\n";
        std::cout << start_left << "Size: " << boards.size() << std::endl;

        const Timer timerSort1;
        // std::sort(boards.begin(), boards.end(), [](const Board &a, const Board &b) { return a.hash < b.hash; });
        boardSorter.sortBoards(boards, tempDepth, colorCount);
        std::cout << start_left << "Sort Time: " << timerSort1.getSeconds() << "\n";

        board1Table.push_back(boards);
    }

    while (board2Table.size() <= depth2) {
        const Timer timer2;
        const int tempDepth = (int)board2Table.size();
        std::vector<Board> boards = (permFuncs[tempDepth])(board2, colorCount);
        std::cout << start_right << "Creation Time: " << timer2.getSeconds() << "\n";
        std::cout << start_right << "Size: " << boards.size() << std::endl;

        const Timer timerSort2;
        // std::sort(boards.begin(), boards.end(), [](const Board &a, const Board &b) { return a.hash < b.hash; });
        boardSorter.sortBoards(boards, tempDepth, colorCount);
        std::cout << start_right << "Sort Time: " << timerSort2.getSeconds() << "\n";

        board2Table.push_back(boards);
    }

    std::vector<std::pair<Board*, Board*>> results;
    if (depth1 != 0 && depth2 != 0) {

        results = intersection_threaded(board1Table[depth1], board2Table[depth2]);
    } else {
        results = intersection(board1Table[depth1], board2Table[depth2]);
    }


    std::cout << start_both << "Solutions: " << results.size() << std::endl;

    if (board1.hasFat()) {
        int xy1 = board1.getFatXY();
        int xy2 = board2.getFatXY();
        for (auto pair: results) {
            std::string moveset = pair.first->mem.assembleFatMoveString(xy1, &pair.second->mem, xy2);
            resultSet.insert(moveset);
        }
    } else {
        for (auto pair: results) {
            std::string moveset = pair.first->mem.assembleMoveString(&pair.second->mem);
            resultSet.insert(moveset);
        }
    }


}


/*
Remaining:
11: 7-5, 8-3
12: 7-5, 9-2
Fats: 4-2, 4-4, 5-1, 8-2, 8-4, 9-1
 */
int main() {

    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    auto pair = BoardLookup::getBoardPair("6-5");
    Board board1 = pair->getInitialState();
    Board board2 = pair->getSolutionState();



    static constexpr int START_DEPTH = 1;





    std::cout << board1.toString(board2) << std::endl;
    std::cout << std::flush;


    std::set<std::string> resultSet;
    int total_depth = START_DEPTH;

    std::vector<std::vector<Board>> board1Table;
    std::vector<std::vector<Board>> board2Table;
    BoardSorter boardSorter;

    if (board1.hasFat()) {
        boardSorter.ensureAux(254803968);
    } else {
        boardSorter.ensureAux(173325000);
    }

    const Timer totalTime;
    while (total_depth <= 10) {
        auto permutationsFromDepth = permutationDepthMap.at(total_depth);
        int permCount = 0;
        for (const auto& permPair : permutationsFromDepth) {
            findSolutions(resultSet, permCount, boardSorter, board1Table, board2Table,
                          board1, board2, permPair.first, permPair.second);
            permCount++;
            if (permCount != permutationsFromDepth.size() - 1) {
                if (!resultSet.empty()) {
                    std::cout << "Unique Solutions so far: " << resultSet.size() << std::endl;
                }
            } else {
                std::cout << "Total Unique Solutions: " << resultSet.size() << std::endl;
            }
        }
        if (!resultSet.empty()) {
            break;
        }
        total_depth++;
    }

    std::cout << "Total Time: " << totalTime.getSeconds() << std::endl;
    if (!resultSet.empty()) {
        std::string filename = pair->getName()
                               + "_c" + std::to_string(total_depth)
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

    return 0;


}
