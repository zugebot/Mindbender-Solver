
#include "MindbenderSolver/solver/board.hpp"
#include "MindbenderSolver/solver/rotations.hpp"

#include "MindbenderSolver/utils/timer.hpp"

#include <iostream>


int main() {
    std::cout << "Hello, World!" << std::endl;

    Board board1;
    {
        const uint8_t values[36] = {
            1, 2, 3, 4, 5, 6,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        };
        board1.setState(values);
    }

    Board board2;
    {
        const uint8_t values[36] = {
            7, 0, 0, 0, 6, 5,
            0, 7, 0, 0, 0, 0,
            0, 0, 7, 0, 0, 0,
            0, 0, 0, 7, 0, 0,
            0, 0, 0, 0, 7, 0,
            0, 0, 0, 0, 0, 7,
        };
        board2.setState(values);
    }

    const uint64_t score1 = board1.getScore1(board2);

    std::cout << "Score: " << score1 << std::endl;

    std::cout << board1.toString() << std::endl;
    R_0_1(board1);
    std::cout << board1.toString() << std::endl;




    // std::cout << board1.toString() << "\n";
    R_0_1(board1);
    // std::cout << board1.toString() << "\n";



    auto *boards = new Board[12960000]; // 12960000
    auto *scores = new uint64_t[12960000];
    std::cout << "Allocated the boards\n";
    const Timer timer;
    int count = 0;
    for (int a = 0; a < 60; a++) {
        for (int b = 0; b < 60; b++) {
            for (int c = 0; c < 60; c++) {
                for (int d = 0; d < 60; d++) {
                    boards[count] = board1;
                    actions[a](boards[count]);
                    actions[b](boards[count]);
                    actions[c](boards[count]);
                    actions[d](boards[count]);
                    // scores[count] = boards[count].getScore1(board2);
                    count++;
                }
            }
        }
    }
    std::cout << timer.getSeconds() << "\n";


    return 0;
}
