#include "board.hpp"

#include <iostream>
#include <emmintrin.h>


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


uint64_t score1Helper(const uint64_t& sect) {
    static constexpr uint64_t M0 = 0x0009249249249249;
    static constexpr uint64_t M1 = 0x0012492492492492;
    static constexpr uint64_t M2 = 0x0024924924924924;
    static constexpr uint64_t M3 = 0x0000E070381C0E07;
    static constexpr uint64_t M4 = 0x000000007800000F;
    static constexpr uint64_t M5 = 0x0000000007FFFFFF;

    const uint64_t p0 = sect & M0
        | (sect & M1) >> 1 | (sect & M2) >> 2;
    const uint64_t p1 = p0 + (p0 >> 3) + (p0 >> 6) & M3;
    const uint64_t p2 = p1 + (p1 >> 9) + (p1 >> 18) & M4;
    return (p2 & M5) + (p2 >> 27);
}


uint64_t Board::getScore1(const Board &other) const {
    return score1Helper(b1 ^ other.b1) +
        score1Helper(b2 ^ other.b2);
}


uint64_t Board::hash() const {
    constexpr uint64_t prime = 31;
    uint64_t hash = 17;
    hash = hash * prime + (b1 ^ b1 >> 32);
    hash = hash * prime + (b2 ^ b2 >> 32);
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

