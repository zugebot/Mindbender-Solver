#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include <string>



/**
 * first 4 bits: move count
 * next 10 * 6 bits: moves
 */
class Memory {
public:
    uint64_t moves;
    Memory() : moves(0) {}

    MUND inline u8 getMoveCount() const { return moves & 0xF; }
    MUND u8 getMove(u8 index) const;
    MUND u8 getLastMove() const;

    void setNext1Move(uint64_t moveValue);
    void setNext2Move(uint64_t moveValue);
    void setNext3Move(uint64_t moveValue);
    void setNext4Move(uint64_t moveValue);
    void setNext5Move(uint64_t moveValue);

    MUND std::string toString() const;
    MUND std::string assembleMoveString(Memory* other) const;
    MUND std::string assembleMoveStringForwards() const;
    MUND std::string assembleMoveStringBackwards() const;

    MUND std::string assembleFatMoveString(u8 fatPos, Memory* other, u8 fatPosOther) const;
    MUND std::string assembleFatMoveStringForwards(u8 fatPos) const;
    MUND std::string assembleFatMoveStringBackwards(u8 fatPos) const;
};


typedef void (Memory::*SetNextMoveFunc)(uint64_t);
extern SetNextMoveFunc setNextMoveFuncs[];