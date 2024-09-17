#pragma once

#include <string>
#include "MindbenderSolver/utils/processor.hpp"
#include "memory.hpp"


class Board {
public:
    static constexpr uint32_t BOARD_SIZE = 6;

    /// uses the lower 54 bits, holds upper 3x6 cell grid
    uint64_t b1 = 0;

    /// uses the lower 54 bits, holds lower 3x6 cell grid
    uint64_t b2 = 0;

    uint64_t hash = 0;

    Memory mem;


    MU void setState(const uint8_t values[36]);
    MU void precompute_hash();

    MU ND uint64_t getScore1(const Board& other) const;
    MU ND uint64_t getScore3(const Board& other) const;

    MU ND std::string toString() const;

    MU ND std::string assembleMoveString(Board* other) const;
    MU ND std::string assembleMoveStringForwards() const;
    MU ND std::string assembleMoveStringBackwards() const;

    bool operator==(const Board& other) const {
        static constexpr uint64_t MASK = 0x3FFFFFFFFFFFFF;
        return (b1 & MASK) == (other.b1 & MASK)
               && (b2 & MASK) == (other.b2 & MASK);
    }
};
