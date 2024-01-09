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
    return b1 + b2;
}


uint32_t Board::getScore2() const {
    return b1 + b2;
}


uint64_t Board::hash() const {
    uint64_t hash = 0;
    hash ^= b1;
    hash ^= b2 << 27 | b2 >> 37;
    return hash;
}








std::string Board::toString() const {
    std::string str;

    auto appendBoardToString = [&str](const uint64_t board) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 6; x++) {
                const uint64_t val = board >> 51 - 3 * (x + y * 6) & 0b111;
                str.append(std::to_string(val));
                str.append(" ");
            }
            str.append("\n");
        }
    };

    appendBoardToString(b1);
    appendBoardToString(b2);

    return str;
}


