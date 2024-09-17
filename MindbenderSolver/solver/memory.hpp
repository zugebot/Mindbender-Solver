#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include <cstdint>
#include <string>



/**
 * first 4 bits: move count
 * next 10 * 6 bits: moves
 */
class Memory {
    uint64_t moves;
public:
    static constexpr uint64_t MOVE_MASK = 0x3F;

    Memory() : moves(0) {}

    MU ND inline uint8_t getMoveCount() const {
        return moves & 0xF;
    }

    void setNext1Move(uint64_t moveValue);
    void setNext2Move(uint64_t moveValue);
    void setNext3Move(uint64_t moveValue);
    void setNext4Move(uint64_t moveValue);
    void setNext5Move(uint64_t moveValue);

    MU ND uint8_t getMove(uint8_t index) const;

    MU ND std::string toString() const;


};


typedef void (Memory::*SetNextMoveFunc)(uint64_t);


extern SetNextMoveFunc setNextMoveFuncs[];