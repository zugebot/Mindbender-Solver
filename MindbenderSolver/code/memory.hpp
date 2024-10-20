#pragma once

#include "MindbenderSolver/utils/processor.hpp"


#include <string>
#include <vector>


class Board;


static constexpr u64 MEMORY_MOVE_TYPE_MASK = 0'77;
static constexpr u32 MEMORY_MOVE_TYPE_BITSIZE = 6;
static constexpr u64 MEMORY_MOVE_DATA_MASK = 0xF;
static constexpr u32 MEMORY_MOVE_DATA_BITSIZE = 4;

class Memory {
    static u8 getShift(c_u32 moveCount) { return MEMORY_MOVE_DATA_BITSIZE + moveCount * MEMORY_MOVE_TYPE_BITSIZE; }

    /**
     * Todo: actually probably don't do this so 16+ moves don't require a re-write?
     * make it so that
     * 1-bit. - isSolved
     * 1-bit. - has Fat
     * 3-bits - fat X
     * 3-bits - fat Y
     * 3/4bit - color count
     * this would leave mem to having a max of 8 moves.
     */
    u64 hash;
    u64 mem;
public:

    /***
     * first 4 bits: move count
     * next 10 * 6 bits: moves
     */
    Memory() : mem(0), hash(0) {}

    // ############################################################
    // #                       u64 hash                           #
    // ############################################################

    typedef void (Memory::*HasherPtr)(u64, u64);

    MUND inline u64 getHash() const { return hash; }
    MU inline void setHash(c_u64 value) { hash = value; }

    MU void precomputeHash2(c_u64 b1, c_u64 b2);
    MU void precomputeHash3(c_u64 b1, c_u64 b2);
    MU void precomputeHash4(c_u64 b1, c_u64 b2);
    MUND static HasherPtr getHashFunc(const Board& board);

    __forceinline bool operator==(const Memory& other) const { return hash == other.hash; }
    __forceinline bool operator<(const Memory& other) const { return hash < other.hash; }
    __forceinline bool operator>(const Memory& other) const { return hash > other.hash; }

    // ############################################################
    // #                       u64 moves                          #
    // ############################################################

    MUND u8 getMoveCount() const;
    MUND u8 getMove(u8 index) const;
    MUND u8 getLastMove() const;

    template<int COUNT> void setNextNMove(u64 moveValue);

    // ############################################################
    // #            To String -Similar- Functions                 #
    // ############################################################

    MUND std::string toString() const;

    MUND static std::string formatMoveString(u8 move, bool isForwards) ;

    MUND std::string asmString(const Memory* other) const;
    MUND std::string asmStringForwards() const;
    MUND std::string asmStringBackwards() const;

    MUND std::string asmFatString(u8 fatPos, const Memory* other, u8 fatPosOther) const;
    MUND std::string asmFatStringForwards(u8 fatPos) const;
    MUND std::string asmFatStringBackwards(u8 fatPos) const;

    MUND static std::vector<u8> parseNormMoveString(const std::string& input);
    MUND static std::vector<u8> parseFatMoveString(const std::string& input);
};


template<int COUNT>
void Memory::setNextNMove(c_u64 moveValue) {
    static_assert(COUNT >= 1 && COUNT <= 5, "Template argument must be in range 1-5");

    constexpr u64 MOVE_SET_SHIFT = (5 - COUNT) * MEMORY_MOVE_TYPE_BITSIZE;
    constexpr u64 MOVE_SET_MASK = 0'77'77'77'77'77 >> MOVE_SET_SHIFT;

    c_u32 moveCount = mem & MEMORY_MOVE_DATA_MASK;
    c_u8 shiftAmount = getShift(moveCount);

    mem = (mem & ~((MOVE_SET_MASK << shiftAmount) | MEMORY_MOVE_DATA_MASK)) // p1
          |
          (moveValue << shiftAmount) // p2
          |
          ((moveCount + COUNT) & MEMORY_MOVE_DATA_MASK) // p3
            ;
}