#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/memory.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/rotations.hpp"
#include "MindbenderSolver/utils/timer.hpp"

#include <algorithm>
#include <bitset>
#include <iostream>


std::vector<std::pair<Board *, Board *>> intersection(std::vector<Board>& boards1, std::vector<Board>& boards2) {
    std::vector<std::pair<Board *, Board *>> results;
    auto it1 = boards1.begin();
    auto it2 = boards2.begin();
    while (it1 != boards1.end() && it2 != boards2.end()) {
        if (it1->getHash() == it2->getHash()) {
            auto it1_end = it1;
            auto it2_end = it2;
            // find range of matching hashes in boards1
            while (it1_end != boards1.end() && it1_end->getHash() == it1->getHash()) {
                ++it1_end;
            }
            // find range of matching hashes in boards2
            while (it2_end != boards2.end() && it2_end->getHash() == it2->getHash()) {
                ++it2_end;
            }
            // make pairs for all combinations of matching hashes
            for (auto it1_match = it1; it1_match != it1_end; ++it1_match) {
                for (auto it2_match = it2; it2_match != it2_end; ++it2_match) {
                    results.emplace_back(&*it1_match, &*it2_match);
                }
            }
            it1 = it1_end;
            it2 = it2_end;
        } else if (it1->getHash() < it2->getHash()) {
            ++it1;
        } else {
            ++it2;
        }
    }
    return results;
}


auto BOARD_SORTER = [](const Board &a, const Board &b) { return a.getHash() < b.getHash(); };



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

    /*
std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
auto pair = BoardLookup::getBoardPair("9-1");
Board board1 = pair->getInitialState();
Board board2 = pair->getSolutionState();

int xy1 = board1.getFatXY();
int xy2 = board2.getFatXY();





std::vector<Board> boards1 = makeFatPermutationListFuncs[4](board1, 4);
std::vector<Board> boards2 = makeFatPermutationListFuncs[4](board2, 4);

std::cout << "did perms" << std::endl;

std::sort(boards1.begin(), boards1.end(), [](const Board &a, const Board &b) { return a.getHash() < b.getHash(); });
std::sort(boards2.begin(), boards2.end(), [](const Board &a, const Board &b) { return a.getHash() < b.getHash(); });

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

volatile int _ = 0;

*/








        int main() {
    static constexpr uint64_t MASK = 0x003F'FFFF'FFFF'FFFF;
    static constexpr uint64_t VAL = 0xAA80'0000'0000'0000;

    auto level = LevelBoardPair::p5_3;

    Board board, check, solve;
    board = level.getInitialState();
    check = level.getInitialState();
    solve = level.getSolutionState();

    (board.*board.getHashFunc())();

    uint64_t hash1 = board.getHash();
    uint64_t hash2 = check.getHash();
    board.b1 |= VAL;
    board.b2 |= VAL;
    check.b1 |= VAL;
    check.b2 |= VAL;

    board.precomputeHash();
    check.precomputeHash();

    if (hash1 != hash2 || hash2 != board.getHash() || board.getHash() != check.getHash()) {
        std::cout << "hashes aren't lining up for the same board state...\n";
        return -1;
    }

    for (int action = 0; action < 60; action++) {
        for (int i = 0; i < 6; i++) {
            actions[action](board);
            if ((board.b1 & ~MASK) != VAL) {
                std::cout << "upper 10 bits changed... ("
                          << action << ")" << std::endl;
            }
        }
        board.precomputeHash();
        uint64_t hash_to_check = board.getHash();
        if (hash_to_check != hash1) {
            std::cout << "hashes aren't lining up for the same board state...\n";
            return -1;
        }

        if (board.b1 != check.b1 || board.b2 != check.b2) {
            std::cout << "Action #" << action << " failed.\n";
            return -1;
        }
    }

    board.b1 &= 0x3FFFFFFFFFFFFF;
    board.b2 &= 0x3FFFFFFFFFFFFF;
    check.b1 &= 0x3FFFFFFFFFFFFF;
    check.b2 &= 0x3FFFFFFFFFFFFF;

    std::cout << "All actions work.\n";



    Board solved5_3 = level.getInitialState();
    R_1_2(solved5_3);
    R_2_3(solved5_3);
    C_0_2(solved5_3);
    C_4_2(solved5_3);
    C_2_4(solved5_3);
    C_3_2(solved5_3);
    R_2_4(solved5_3);
    if (solve.b1 != solved5_3.b1 || solve.b2 != solved5_3.b2) {
        std::cout << "The permutations of a solution performed on the initial board state did not solve it.\n";
        std::cout << "The board in question:\n";
        std::cout << solved5_3.toString() << std::endl;

        uint64_t var1 = solve.b1;
        uint64_t var2 = solved5_3.b1;
        uint64_t var3 = solve.b2;
        uint64_t var4 = solved5_3.b2;
        std::cout << (var1 >> 54) << std::endl;
        std::cout << (var2 >> 54) << std::endl;
        std::cout << (var3 >> 54) << std::endl;
        std::cout << (var4 >> 54) << std::endl;
        return -1;
    }



    // TEST THE SOLUTION FROM BOTH DIRECTIONS
    // R12 R23 C02 C42 C24 C32 R24

    // R12 R23 C02 C42
    Board board_left = level.getInitialState();
    R_1_2(board_left);
    R_2_3(board_left);
    C_0_2(board_left);
    C_4_2(board_left);
    board_left.precomputeHash();

    // C24 C32 R24
    Board board_right = level.getSolutionState();
    allActionsList[11](board_right); // R_2_2(board_right);
    allActionsList[41](board_right); // C_2_2(board_right);
    allActionsList[48](board_right); // C_3_4(board_right);
    (board_right.hashMem.mem.setNextMoveFuncs[2])(11 | 41 << 6 | 48 << 12);

    board_right.precomputeHash();

    if (board_left.b1 != board_right.b1 || board_left.b2 != board_right.b2) {
        std::cout << "left and right should be the same but they aren't.";
        std::cout << "The boards in question:\n";
        std::cout << board_left.toString() << std::endl;
        std::cout << board_right.toString() << "\n\n";

        uint64_t var1 = board_left.b1;
        uint64_t var2 = board_right.b1;
        uint64_t var3 = board_left.b2;
        uint64_t var4 = board_right.b2;
        std::cout << (var1 >> 54) << std::endl;
        std::cout << (var2 >> 54) << std::endl;
        std::cout << (var3 >> 54) << std::endl;
        std::cout << (var4 >> 54) << std::endl;
        return -1;
    }

    if (board_left.getHash() != board_right.getHash()) {
        std::cout << "even if the boards are the same, the hashes do not match\n";
        std::cout << "left:  " << board_left.getHash() << std::endl;
        std::cout << "right: " << board_right.getHash() << std::endl;
    }






    std::vector<Board> boards_depth_1 = makePermutationListFuncs[1](board);
    std::vector<Board> boards_depth_2 = makePermutationListFuncs[2](board);
    std::vector<Board> boards_depth_3 = makePermutationListFuncs[3](board);
    std::vector<Board> boards_depth_4 = makePermutationListFuncs[4](board);
    bool found1 = false;
    for (const auto& iterBoard : boards_depth_4) {
        if (iterBoard.getHash() == board_left.getHash()) {
            found1 = true;
        }
    }
    if (!found1) {
        std::cout << "(boards) makePermListFuncs[4] isn't correct???\n";
        return -1;
    }
    std::sort(boards_depth_1.begin(), boards_depth_1.end(), BOARD_SORTER);
    std::sort(boards_depth_2.begin(), boards_depth_2.end(), BOARD_SORTER);
    std::sort(boards_depth_3.begin(), boards_depth_3.end(), BOARD_SORTER);
    std::sort(boards_depth_4.begin(), boards_depth_4.end(), BOARD_SORTER);



    std::vector<Board> solves_depth_1 = makePermutationListFuncs[1](solve);
    std::vector<Board> solves_depth_2 = makePermutationListFuncs[2](solve);
    std::vector<Board> solves_depth_3 = makePermutationListFuncs[3](solve);
    std::vector<Board> solves_depth_4 = makePermutationListFuncs[4](solve);
    bool found2 = false;
    for (const auto& iterBoard : solves_depth_3) {
        std::string str = iterBoard.hashMem.mem.toString();
        if (str == "Move[11, 41, 48]") {
            volatile int x = 0;
        }
        if (iterBoard.getHash() == board_right.getHash()) {
            found2 = true;
        }
    }
    if (!found2) {
        std::cout << "(solves) makePermListFuncs[3] isn't correct???\n";
        return -1;
    }
    std::sort(solves_depth_1.begin(), solves_depth_1.end(), BOARD_SORTER);
    std::sort(solves_depth_2.begin(), solves_depth_2.end(), BOARD_SORTER);
    std::sort(solves_depth_3.begin(), solves_depth_3.end(), BOARD_SORTER);
    std::sort(solves_depth_4.begin(), solves_depth_4.end(), BOARD_SORTER);


    auto results1 = intersection(boards_depth_4, solves_depth_4);
    auto results2 = intersection(boards_depth_3, solves_depth_4);
    auto results3 = intersection(boards_depth_4, solves_depth_3);

    std::cout << "solutions [4, 4]: " << results1.size() << "\n";
    std::cout << "solutions [3, 4]: " << results2.size() << "\n";
    std::cout << "solutions [4, 3]: " << results3.size() << "\n";

    volatile int x = 0;

}