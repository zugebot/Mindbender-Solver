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

    MUND u8 getMoveCount() const { return moves & 0xF; }
    MUND u8 getMove(u8 index) const;
    MUND u8 getLastMove() const;

    void setNext1Move(u64 moveValue);
    void setNext2Move(u64 moveValue);
    void setNext3Move(u64 moveValue);
    void setNext4Move(u64 moveValue);
    void setNext5Move(u64 moveValue);

    MUND std::string toString() const;
    MUND std::string assembleMoveString(const Memory* other) const;
    MUND std::string assembleMoveStringForwards() const;
    MUND std::string assembleMoveStringBackwards() const;

    MUND std::string assembleFatMoveString(u8 fatPos, const Memory* other, u8 fatPosOther) const;
    MUND std::string assembleFatMoveStringForwards(u8 fatPos) const;
    MUND std::string assembleFatMoveStringBackwards(u8 fatPos) const;
};


typedef void (Memory::*SetNextMoveFunc)(uint64_t);
extern SetNextMoveFunc setNextMoveFuncs[];