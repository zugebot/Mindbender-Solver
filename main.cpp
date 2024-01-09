
#include "MindbenderSolver/solver/board.hpp"
#include "MindbenderSolver/solver/rotations.hpp"

#include "MindbenderSolver/utils/timer.hpp"

#include <iostream>


int main() {
    std::cout << "Hello, World!" << std::endl;

    Board board;
    const uint8_t values1[36] = {
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5,
    };
    board.setBoard(values1);

    Board solve;
    const uint8_t values2[36] = {
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
    };
    solve.setBoard(values2);



    auto *boards = new Board[12960000]; // 12960000
    std::cout << "Allocated the boards\n";
    const Timer timer;
    int count = 0;
    for (int a = 0; a < 60; a++) {
        for (int b = 0; b < 60; b++) {
            for (int c = 0; c < 60; c++) {
                for (int d = 0; d < 60; d++) {
                    boards[count] = board;
                    actions[a](boards[count]);
                    actions[b](boards[count]);
                    actions[c](boards[count]);
                    actions[d](boards[count]);
                    count++;
                }
            }
        }
    }

    std::cout << timer.getSeconds() << "\n";


    return 0;
}
