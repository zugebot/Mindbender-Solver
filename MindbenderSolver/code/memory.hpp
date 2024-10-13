#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include <string>


static constexpr u64 MEMORY_MOVE_TYPE_MASK = 0'77;
static constexpr u32 MEMORY_MOVE_TYPE_BITSIZE = 6;
static constexpr u64 MEMORY_MOVE_DATA_MASK = 0xF;
static constexpr u32 MEMORY_MOVE_DATA_BITSIZE = 4;

class Memory {

    static u8 getShift(c_u32 moveCount) { return MEMORY_MOVE_DATA_BITSIZE + moveCount * MEMORY_MOVE_TYPE_BITSIZE; }

public:
    /***
     * first 4 bits: move count
     * next 10 * 6 bits: moves
     */
    u64 moves;
    Memory() : moves(0) {}

    MUND u8 getMoveCount() const;
    MUND u8 getMove(u8 index) const;
    MUND u8 getLastMove() const;

    template<int COUNT> void setNextNMove(u64 moveValue);



    // ############################################################
    // #            To String -Similar- Functions                 #
    // ############################################################

    MUND std::string toString() const;

    MUND static std::string formatMoveString(u8 move, bool isBackwards) ;

    MUND std::string asmString(const Memory* other) const;
    MUND std::string asmStringForwards() const;
    MUND std::string asmStringBackwards() const;

    MUND std::string asmFatString(u8 fatPos, const Memory* other, u8 fatPosOther) const;
    MUND std::string asmFatStringForwards(u8 fatPos) const;
    MUND std::string asmFatStringBackwards(u8 fatPos) const;
};




template<int COUNT>
void Memory::setNextNMove(c_u64 moveValue) {
    static_assert(COUNT >= 1 && COUNT <= 5, "Template argument must be in range 1-5");

    constexpr u64 MOVE_SET_SHIFT = (5 - COUNT) * MEMORY_MOVE_TYPE_BITSIZE;
    constexpr u64 MOVE_SET_MASK = 0'77'77'77'77'77 >> MOVE_SET_SHIFT;

    c_u32 moveCount = moves & MEMORY_MOVE_DATA_MASK;
    c_u8 shiftAmount = getShift(moveCount);

    c_u64 p1 = (moves & ~((MOVE_SET_MASK << shiftAmount) | MEMORY_MOVE_DATA_MASK));
    c_u64 p2 = (moveValue << shiftAmount);
    c_u64 p3 = ((moveCount + COUNT) & MEMORY_MOVE_DATA_MASK);

    moves = p1 | p2 | p3;
}