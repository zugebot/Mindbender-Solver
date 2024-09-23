#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/memory.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/sorter.hpp"
#include "MindbenderSolver/code/intersection.hpp"
#include "MindbenderSolver/code/rotations.hpp"

#include "MindbenderSolver/utils/colors.hpp"
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
    const uint32_t colorCount = board1.getColorCount();

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
        boardSorter.sortBoards(boards, tempDepth, colorCount);
        std::cout << start_left << "Sort Time: " << timerSort1.getSeconds() << "\n";

        board1Table.push_back(boards);
    }

    while (board2Table.size() <= depth2) {
        const Timer timer2;
        const int tempDepth = (int)board2Table.size();
        std::vector<Board> boards = (makePermutationListFuncs[tempDepth])(board2, colorCount);
        std::cout << start_right << "Creation Time: " << timer2.getSeconds() << "\n";
        std::cout << start_right << "Size: " << boards.size() << std::endl;

        const Timer timerSort2;
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

    for (auto pair: results) {
        std::string moveset = pair.first->mem.assembleMoveString(&pair.second->mem);
        resultSet.insert(moveset);
    }
}


/*
Remaining:
11: 7-5, 8-3
12: 7-5, 9-2
Fats: 4-2, 4-4, 5-1, 8-2, 8-4, 9-1
 */
int main() {

    /*
    Board board1 = BoardLookup::getBoardPair("9-1")->getInitialState();
    Board board0 = BoardLookup::getBoardPair("9-1")->getInitialState();
    Board board2 = BoardLookup::getBoardPair("9-1")->getInitialState();

    int _xy1 = board1.getFatXY();
    int _x1 = board1.getFatX();
    int _y1 = board1.getFatY();
    auto actions1 = fatActions[_x1 * 5 + _y1];
    auto func1 = actions1[15];
    func1(board1);
    int _x2 = board1.getFatX();
    int _y2 = board1.getFatY();
    auto actions2 = fatActions[_x2 * 5 + _y2];
    auto func2 = actions2[36];
    func2(board1);
    int _x3 = board1.getFatX();
    int _y3 = board1.getFatY();
    auto actions3 = fatActions[_x3 * 5 + _y3];
    auto func3 = actions3[22];
    func3(board1);
    int _x4 = board1.getFatX();
    int _y4 = board1.getFatY();
    auto actions4 = fatActions[_x4 * 5 + _y4];
    auto func4 = actions4[47];
    func4(board1);
    board1.mem.setNext4Move(15 | 36 << 6 | 22 << 12 | 47 << 18);





    int _xy5 = board2.getFatXY();
    int _x5 = board2.getFatX();
    int _y5 = board2.getFatY();
    auto actions5 = fatActions[_x5 * 5 + _y5];
    auto func5 = actions5[19];
    func5(board2);
    int _x6 = board2.getFatX();
    int _y6 = board2.getFatY();
    auto actions6 = fatActions[_x6 * 5 + _y6];
    auto func6 = actions6[5];
    func6(board2);
    int _x7 = board2.getFatX();
    int _y7 = board2.getFatY();
    auto actions7 = fatActions[_x7 * 5 + _y7];
    auto func7 = actions7[36];
    func7(board2);
    int _x8 = board2.getFatX();
    int _y8 = board2.getFatY();
    auto actions8 = fatActions[_x8 * 5 + _y8];
    auto func8 = actions8[0];
    func8(board2);
    board2.mem.setNext4Move(19 | 5 << 6 | 36 << 12 | 0 << 18);


    std::string b1moves = board1.mem.assembleFatMoveStringBackwards(_xy1);
    std::string b2moves = board2.mem.assembleFatMoveStringForwards(_xy5);
    // std::string moveStr = board2.mem.assembleFatMoveString(_xy5, &board1.mem, _xy1);

    std::cout << board1.toString() << std::endl;

    std::cout << b1moves << " " << b2moves << std::endl;


    std::cout << board2.toString() << std::endl;

    */

    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    auto pair = BoardLookup::getBoardPair("9-1");
    Board board1 = pair->getInitialState();
    Board board2 = pair->getSolutionState();

    int xy1 = board1.getFatXY();
    int xy2 = board2.getFatXY();





    std::vector<Board> boards1 = makeFatPermutationListFuncs[4](board1, 4);
    std::vector<Board> boards2 = makeFatPermutationListFuncs[4](board2, 4);

    std::cout << "did perms" << std::endl;

    std::sort(boards1.begin(), boards1.end(), [](const Board &a, const Board &b) { return a.hash < b.hash; });
    std::sort(boards2.begin(), boards2.end(), [](const Board &a, const Board &b) { return a.hash < b.hash; });

    std::cout << "did sorting" << std::endl;

    auto results = intersection(boards1, boards2);

    std::cout << "did intersection" << std::endl;

    std::cout << "solution count: " << results.size() << std::endl;

    if (!results.empty()) {
        std::cout << board1.toString() << std::endl;
        std::cout << results[0].first->mem.assembleFatMoveStringForwards(xy1) << std::endl;
        std::cout << results[0].first->toString() << std::endl;
        // std::cout << results[0].second->mem.assembleFatMoveStringBackwards(results[0].second->getFatXY()) << std::endl;
        std::cout << results[0].second->mem.assembleFatMoveStringBackwards(xy2) << std::endl;

        std::cout << board2.toString() << std::endl;
    }

    // TODO: walk through the order of moves performed on board2 / results[0].second,
    // TODO: figure out what xy to pass to aFMSB, whether to increment or decrement, and so on
    // TODO: maybe use the forward dict for the backwards, but then just flip the number after getting the right func?

    volatile int _ = 0;







     /*


    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    auto pair = BoardLookup::getBoardPair("9-4");
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
    boardSorter.ensureAux(173325000);

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

    */
}
