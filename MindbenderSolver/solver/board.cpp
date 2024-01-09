#include "board.hpp"

#include <iostream>


void Board::setState(const uint8_t values[36]) {
    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = b1 << 3 | values[i];
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = b2 << 3 | values[i];
    }
}


uint32_t Board::getScore1() const {
    // TODO: Implement
    return b1 + b2;
}


uint32_t Board::getScore2() const {
    // TODO: Implement
    return b1 + b2;
}


uint64_t Board::hash() const {
    constexpr uint64_t prime = 31;
    uint64_t hash = 17;
    hash = hash * prime + (b1 ^ (b1 >> 32));
    hash = hash * prime + (b2 ^ (b2 >> 32));
    return hash;
}


std::string Board::toString() const {
    std::string str;

    auto appendBoardToString = [&str](const uint64_t board) {
        for (int y = 0; y < 54; y += 18) {
            for (int x = 0; x < 18; x += 3) {
                str.append(std::to_string(board >> 51 - x - y & 0b111));
                str.append(" ");
            }
            str.append("\n");
        }
    };

    appendBoardToString(b1);
    appendBoardToString(b2);
    return str;
}

