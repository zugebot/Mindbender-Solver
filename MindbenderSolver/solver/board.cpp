#pragma once

#include "board.hpp"

#include <iostream>


void Board::setBoard(const uint8_t values[36]) {
    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = (b1 << 3) | values[i];
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = (b2 << 3) | values[i];
    }
}


void Board::printState() const {
    std::cout << "\n";
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 6; x++) {
            uint64_t val = b1 >> (51 - 3 * (x + y * 6)) & 0b111;
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 6; x++) {
            uint64_t val = b2 >> (51 - 3 * (x + y * 6)) & 0b111;
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}


