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
        if (it1->hash == it2->hash) {
            auto it1_end = it1;
            auto it2_end = it2;
            // find range of matching hashes in boards1
            while (it1_end != boards1.end() && it1_end->hash == it1->hash) {
                ++it1_end;
            }
            // find range of matching hashes in boards2
            while (it2_end != boards2.end() && it2_end->hash == it2->hash) {
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
        } else if (it1->hash < it2->hash) {
            ++it1;
        } else {
            ++it2;
        }
    }
    return results;
}


auto BOARD_SORTER = [](const Board &a, const Board &b) { return a.hash < b.hash; };




int main() {
    static constexpr uint64_t MASK = 0x003F'FFFF'FFFF'FFFF;
    static constexpr uint64_t VAL = 0xAA80'0000'0000'0000;

    auto level = LevelBoardPair::p5_3;

    Board board, check, solve;
    board = level.getInitialState();
    check = level.getInitialState();
    solve = level.getSolutionState();

    board.precomputeHash();
    check.precomputeHash();

    uint64_t hash1 = board.hash;
    uint64_t hash2 = check.hash;
    board.b1 |= VAL;
    board.b2 |= VAL;
    check.b1 |= VAL;
    check.b2 |= VAL;

    board.precomputeHash();
    check.precomputeHash();

    if (hash1 != hash2 || hash2 != board.hash || board.hash != check.hash) {
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
        uint64_t hash_to_check = board.hash;
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
    actions[11](board_right); // R_2_2(board_right);
    actions[41](board_right); // C_2_2(board_right);
    actions[48](board_right); // C_3_4(board_right);
    (board_right.mem.*setNextMoveFuncs[2])(11 | 41 << 6 | 48 << 12);

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

    if (board_left.hash != board_right.hash) {
        std::cout << "even if the boards are the same, the hashes do not match\n";
        std::cout << "left:  " << board_left.hash << std::endl;
        std::cout << "right: " << board_right.hash << std::endl;
    }






    std::vector<Board> boards_depth_1 = makePermutationListFuncs[1](board);
    std::vector<Board> boards_depth_2 = makePermutationListFuncs[2](board);
    std::vector<Board> boards_depth_3 = makePermutationListFuncs[3](board);
    std::vector<Board> boards_depth_4 = makePermutationListFuncs[4](board);
    bool found1 = false;
    for (const auto& iterBoard : boards_depth_4) {
        if (iterBoard.hash == board_left.hash) {
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
        std::string str = iterBoard.mem.toString();
        if (str == "Move[11, 41, 48]") {
            volatile int x = 0;
        }
        if (iterBoard.hash == board_right.hash) {
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