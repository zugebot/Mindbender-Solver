#pragma once
// code/board.hpp

#include "utils/processor.hpp"

#include <array>
#include <vector>
#include <string>


class B1B2;
using Action = void (*)(B1B2&);


class B1B2 {
public:
    using ColorArray_t = std::array<i8, 8>;

    /**
     *  3 bits: fat x position
     *  3 bits: fat y position
     *  1 bit : 1 if has a fat, otherwise 0
     *  3 bits: (# of colors - 1)
     * 54 bits: holds upper 3x6 cell grid (3 bits each, 18 total)
     */
    u64 b1 = 0;

    /**
     * 54 bits: holds lower 3x6 cell grid (3 bits each, 18 total)
     */
    u64 b2 = 0;

    B1B2() = default;
    HD B1B2(C u64 theB1, C u64 theB2) : b1(theB1), b2(theB2) {}

    MU void setState(C u8 values[36]);
    MU ColorArray_t setStateAndRetColors(C u8 values[36]);

    MU HD void setColorCount(u64 colorCount);
    MUND HD u32 getColorCount() C;

    MU HD void setFatBool(bool flag);
    MUND HD bool getFatBool() C;

    MU HD void setFatX(u64 x);
    MU HD void addFatX(u64 x);
    MUND HD u8 getFatX() C;

    MU HD void setFatY(u64 y);
    MU HD void addFatY(u64 y);
    MUND HD u8 getFatY() C;

    MU HD void setFatXY(u64 x, u64 y);
    MUND HD u8 getFatXY() C { return getFatX() * 5 + getFatY();}
    MUND HD u8 getFatXYFast() C { return (getFatX() << 3) + getFatY();}

    MUND HD u8 getColor(u8 x, u8 y) C;

    MUND HD u64 getScore1(C B1B2& other) C;
    
    MUND HD i32 getExactRowColLowerBound(C B1B2& other) C;
    template<i32 MAX_DEPTH>
    MUND HD bool getExactRowColLowerBoundTill(C B1B2& other) C;
    MUND HD bool couldBeSolvedIn1Move(const B1B2 theOther) C;

    MU __host__ void doMoves(std::initializer_list<Action> theInitList);

    FORCEINLINE HD bool operator==(C B1B2& other) C {
        return b1 == other.b1 && b2 == other.b2;
    }
    
    FORCEINLINE HD bool operator<(C B1B2& other) C {
        return (b1 < other.b1) || (b1 == other.b1 && b2 < other.b2);
    }
    
    FORCEINLINE HD bool operator>(C B1B2& other) C {
        return (b1 > other.b1) || (b1 == other.b1 && b2 > other.b2);
    }
};






MU static constexpr u32 MEMORY_MOVE_TYPE_BITSIZE = 6;
MU static constexpr u64 MEMORY_MOVE_TYPE_MASK = 0'77;
MU static constexpr u32 MEMORY_MOVE_DATA_BITSIZE = 4;
MU static constexpr u64 MEMORY_MOVE_DATA_MASK = 0xF;

class Board;

class Memory {
    static HD u8 getShift(C u32 moveCount) {
        return MEMORY_MOVE_DATA_BITSIZE + moveCount * MEMORY_MOVE_TYPE_BITSIZE;
    }

    u64 mem;

public:

    HD Memory() : mem(0) {}

    HD Memory(MU C std::initializer_list<u64> moveValues) : mem(0) {
        for (C auto& moveValue : moveValues) {
            setNextNMove<1>(moveValue);
        }
    }

    // ############################################################
    // #                       u64 hash                           #
    // ############################################################

    MUND HD u64 getMem() C { return mem; }
    MU HD void setMem(C u64 value) { mem = value; }

    // ############################################################
    // #                       u64 moves                          #
    // ############################################################

    MUND FORCEINLINE HD u8 getMoveCount() C {
        return mem & MEMORY_MOVE_DATA_MASK;
    }

    MUND FORCEINLINE HD u8 getMove(C u8 index) C {
        return mem >> getShift(index) & MEMORY_MOVE_TYPE_MASK;
    }

    MUND FORCEINLINE HD u8 getLastMove() C {
        return mem >> getShift(getMoveCount() - 1) & MEMORY_MOVE_TYPE_MASK;
    }

    template<i32 COUNT>
    HD void setNextNMove(u64 moveValue);

    MU HD void setNextMoves(std::initializer_list<u64> moveValues);

    // ############################################################
    // #            To String -Similar- Functions                 #
    // ############################################################

    MUND std::string toString() C;

    MUND static std::string formatMoveString(u8 move, bool isForwards);

    MUND std::string asmString(C Memory* other) C;
    MUND std::string asmStringForwards() C;
    MUND std::string asmStringBackwards() C;

    MUND std::string asmFatString(u8 fatPos, C Memory* other, u8 fatPosOther) C;
    MUND std::string asmFatStringForwards(u8 fatPos) C;
    MUND std::string asmFatStringBackwards(u8 fatPos) C;

    MUND static std::vector<u8> parseNormMoveString(C std::string& input);
    MUND static std::vector<u8> parseFatMoveString(C std::string& input);
};

template<i32 COUNT>
HD void Memory::setNextNMove(C u64 moveValue) {
    static_assert(COUNT >= 1 && COUNT <= 5, "Template argument must be in range 1-5");

    constexpr u64 MOVE_SET_SHIFT = (5 - COUNT) * MEMORY_MOVE_TYPE_BITSIZE;
    constexpr u64 MOVE_SET_MASK = 0'77'77'77'77'77 >> MOVE_SET_SHIFT;

    C u32 moveCount = mem & MEMORY_MOVE_DATA_MASK;
    C u8 shiftAmount = getShift(moveCount);

    mem = (mem & ~((MOVE_SET_MASK << shiftAmount) | MEMORY_MOVE_DATA_MASK))
          | (moveValue << shiftAmount)
          | ((moveCount + COUNT) & MEMORY_MOVE_DATA_MASK);
}












class Board : public B1B2 {
public:

    static ColorArray_t ColorsDefault;

    struct PrintSettings {
        bool useAscii;
        ColorArray_t trueColors = ColorsDefault;

        PrintSettings() : useAscii(true) {}

        MU PrintSettings(C bool useAsciiIn, C ColorArray_t colorsIn)
            : useAscii(useAsciiIn), trueColors(colorsIn) {}
    };

    Memory memory;

    Board() = default;

    Board(C std::initializer_list<u8> values) { setState(values.begin()); }
    explicit Board(C u8 values[36]) { setState(values); }
    explicit Board(C u8 values[36], u8 x, u8 y);

    MUND HD B1B2 asB1B2() C { return {b1, b2}; }
    
    MUND HD Memory& getMemory() { return memory; }
    MUND HD C Memory& getMemory() C { return memory; }

    MUND HD bool doActISColMatch(u8 x1, u8 y1, u8 m, u8 n) C;
    MUND HD u8 doActISColMatchBatched(u8 x1, u8 y1, u8 m) C;
    MUND HD static double getDuplicateEstimateAtDepth(u32 depth);
    MUND HD u64 getRowColIntersections(u32 x, u32 y) C;

    MUND HD u32 getRowCC() C;
    MUND HD u32 getColCC() C;

    MU __device__ void setRowColCC(u32* ptr) C;

    MU static void appendBoardToString(std::string& str, C Board* board, i32 curY, PrintSettings theSettings = {});
    MUND std::string toString(C Board& other, PrintSettings theSettings = {}) C;
    MUND std::string toString(C Board* other, PrintSettings theSettings = {}) C;
    MUND std::string toStringSingle(PrintSettings theSettings) C;
    MUND std::string toBlandString() C;

    FORCEINLINE HD bool operator==(C Board& other) C {
        return b1 == other.b1 && b2 == other.b2;
    }

    FORCEINLINE HD bool operator<(C Board& other) C {
        return (b1 < other.b1) || (b1 == other.b1 && b2 < other.b2);
    }
    
    FORCEINLINE HD bool operator>(C Board& other) C {
        return (b1 > other.b1) || (b1 == other.b1 && b2 > other.b2);
    }
};




extern template HD bool B1B2::getExactRowColLowerBoundTill<1>(C B1B2& other) C;
extern template HD bool B1B2::getExactRowColLowerBoundTill<2>(C B1B2& other) C;
extern template HD bool B1B2::getExactRowColLowerBoundTill<3>(C B1B2& other) C;
extern template HD bool B1B2::getExactRowColLowerBoundTill<4>(C B1B2& other) C;
extern template HD bool B1B2::getExactRowColLowerBoundTill<5>(C B1B2& other) C;

#ifdef USE_CUDA
namespace my_cuda {
    MU __constant__ extern u8 ROW_COL_OFFSETS[30];
}
#endif










class StateHash {
public:
    enum class HashKind : u8 {
        Hash2,
        Hash3,
        Hash4,
    };
    
    using HashFuncPtr = u64 (*)(C B1B2& state);

    MUND HD static u64 computeHash(C B1B2& state);
    MUND HD static HashFuncPtr getHashFunc();
    MU static void refreshHashFunc(C B1B2& state);
    MU static void setHashKind(HashKind kind);

private:
    MUND HD static HashKind chooseHashKind(C B1B2& state);

    MUND HD static u64 computeHash2(C B1B2& state);
    MUND HD static u64 computeHash3(C B1B2& state);
    MUND HD static u64 computeHash4(C B1B2& state);

    static HashFuncPtr gHashFunc_;
    static HashKind gHashKind_;
};










