#pragma once

#include "MindbenderSolver/utils/processor.hpp"


#include <string>
#include <vector>


class Board;


static constexpr u32 MEMORY_MOVE_TYPE_BITSIZE = 6;
static constexpr u64 MEMORY_MOVE_TYPE_MASK = 0'77;
static constexpr u32 MEMORY_MOVE_DATA_BITSIZE = 4;
static constexpr u64 MEMORY_MOVE_DATA_MASK = 0xF;

class Memory {
    static u8 getShift(C u32 moveCount) { return MEMORY_MOVE_DATA_BITSIZE + moveCount * MEMORY_MOVE_TYPE_BITSIZE; }

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
    Memory() : hash(0), mem(0) {}

    // ############################################################
    // #                       u64 hash                           #
    // ############################################################

    typedef void (Memory::*HasherPtr)(u64, u64);

    MUND u64 getMem() C { return mem; }
    MU void setMem(C u64 value) { mem = value; }

    MUND u64 getHash() C { return hash; }
    MU void setHash(C u64 value) { hash = value; }

    MU void precomputeHash2(u64 b1, u64 b2);
    MU void precomputeHash3(u64 b1, u64 b2);
    MU void precomputeHash4(u64 b1, u64 b2);
    MUND static HasherPtr getHashFunc(C Board& board);

    __forceinline bool operator==(C Memory& other) C { return hash == other.hash; }
    __forceinline bool operator<(C Memory& other) C { return hash < other.hash; }
    __forceinline bool operator>(C Memory& other) C { return hash > other.hash; }

    // ############################################################
    // #                       u64 moves                          #
    // ############################################################

    MUND u8 getMoveCount() C;
    MUND u8 getMove(u8 index) C;
    MUND u8 getLastMove() C;

    template<int COUNT> void setNextNMove(u64 moveValue);

    // ############################################################
    // #            To String -Similar- Functions                 #
    // ############################################################

    MUND std::string toString() C;

    MUND static std::string formatMoveString(u8 move, bool isForwards) ;

    MUND std::string asmString(C Memory* other) C;
    MUND std::string asmStringForwards() C;
    MUND std::string asmStringBackwards() C;

    MUND std::string asmFatString(u8 fatPos, C Memory* other, u8 fatPosOther) C;
    MUND std::string asmFatStringForwards(u8 fatPos) C;
    MUND std::string asmFatStringBackwards(u8 fatPos) C;

    MUND static std::vector<u8> parseNormMoveString(C std::string& input);
    MUND static std::vector<u8> parseFatMoveString(C std::string& input);
};


template<int COUNT>
void Memory::setNextNMove(C u64 moveValue) {
    static_assert(COUNT >= 1 && COUNT <= 5, "Template argument must be in range 1-5");

    constexpr u64 MOVE_SET_SHIFT = (5 - COUNT) * MEMORY_MOVE_TYPE_BITSIZE; // 6
    constexpr u64 MOVE_SET_MASK = 0'77'77'77'77'77 >> MOVE_SET_SHIFT; // 6

    C u32 moveCount = mem & MEMORY_MOVE_DATA_MASK;
    C u8 shiftAmount = getShift(moveCount);

    mem = (mem & ~((MOVE_SET_MASK << shiftAmount) | MEMORY_MOVE_DATA_MASK)) // p1
          |
          (moveValue << shiftAmount) // p2
          |
          ((moveCount + COUNT) & MEMORY_MOVE_DATA_MASK) // p3
            ;
}