#pragma once
// code/board.hpp

#include "utils/processor.hpp"

#include <array>
#include <vector>
#include <string>


class B1B2;
using Action = void (*)(B1B2&);







namespace b1b2 {
    

    constexpr u64 MAKE_MASK(const u64 offset, const u64 bits) {
        return ~(((1ULL << bits) - 1ULL) << offset);
    }

    static constexpr u64 COLOR_COUNT_OFFSET = 54ULL;
    static constexpr u64 COLOR_COUNT_BITS = 3ULL;
    static constexpr u64 COLOR_COUNT_MASK = MAKE_MASK(COLOR_COUNT_OFFSET, COLOR_COUNT_BITS);
    
    static constexpr u64 FAT_BOOL_OFFSET = 57ULL;
    static constexpr u64 FAT_BOOL_BITS = 1ULL;
    static constexpr u64 FAT_BOOL_MASK = MAKE_MASK(FAT_BOOL_OFFSET, FAT_BOOL_BITS);
    
    static constexpr u64 FAT_Y_OFFSET = 58ULL;
    static constexpr u64 FAT_Y_BITS = 3ULL;
    static constexpr u64 FAT_Y_MASK = MAKE_MASK(FAT_Y_OFFSET, FAT_Y_BITS);
    
    static constexpr u64 FAT_X_OFFSET = 61ULL;
    static constexpr u64 FAT_X_BITS = 3ULL;
    static constexpr u64 FAT_X_MASK = MAKE_MASK(FAT_X_OFFSET, FAT_X_BITS);
    
    
    template<typename T1, typename T2>
    FORCEINLINE u64 getShiftAmount(const T1 x, const T2 y) {
        static constexpr u64 MAGIC = 0x33210F33210F;
        return (MAGIC >> (y * 8)) - (x * 3);
    }

}

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
    HD B1B2(const u64 theB1, const u64 theB2) : b1(theB1), b2(theB2) {}

    MU void setState(const u8 values[36]);
    MU ColorArray_t setStateAndRetColors(const u8 values[36]);

    MU FORCEINLINE HD void setColorCount(const u64 colorCount) {
        b1 = (b1 & b1b2::COLOR_COUNT_MASK) | ((colorCount - 1) << b1b2::COLOR_COUNT_OFFSET);
    }
    
    MUND FORCEINLINE HD u32 getColorCount() const {
        const u64 colorCount = (b1 & ~b1b2::COLOR_COUNT_MASK) >> b1b2::COLOR_COUNT_OFFSET;
        return colorCount + 1;
    }

    MU FORCEINLINE HD void setFatBool(const bool flag) {
        b1 = (b1 & b1b2::FAT_BOOL_MASK) | (static_cast<u64>(flag) << b1b2::FAT_BOOL_OFFSET);
    }
    MUND FORCEINLINE HD bool getFatBool() const {
        return (b1 >> b1b2::FAT_BOOL_OFFSET) & 1;
    }

    MU FORCEINLINE HD void setFatX(const u64 x) {
        b1 = (b1 & b1b2::FAT_X_MASK) | (x << b1b2::FAT_X_OFFSET);
    }
    MU FORCEINLINE HD void addFatX(const u64 x) {
        static constexpr u64 ADD_FAT_MAGIC = 0x8D116344;
        b1 = (b1 & b1b2::FAT_X_MASK)
             | ((((ADD_FAT_MAGIC >> (3 * (getFatX() + x) - 1)) & 0b111)) << b1b2::FAT_X_OFFSET);
    }
    
    MUND FORCEINLINE HD u8 getFatX() const {
        return (b1 & ~b1b2::FAT_X_MASK) >> b1b2::FAT_X_OFFSET;
    }

    MU FORCEINLINE HD void setFatY(const u64 y) {
        b1 = (b1 & b1b2::FAT_Y_MASK) | (y << b1b2::FAT_Y_OFFSET);
    }
    MU FORCEINLINE HD void addFatY(const u64 y) {
        static constexpr u64 ADD_FAT_MAGIC = 0x8D116344;
        b1 = (b1 & b1b2::FAT_Y_MASK)
             | ((((ADD_FAT_MAGIC >> (3 * (getFatY() + y) - 1)) & 0b111)) << b1b2::FAT_Y_OFFSET);
    }
    MUND FORCEINLINE HD u8 getFatY() const {
        return (b1 & ~b1b2::FAT_Y_MASK) >> b1b2::FAT_Y_OFFSET;
    }

    MU FORCEINLINE HD void setFatXY(const u64 x, const u64 y) {
        setFatX(x);
        setFatY(y);
        setFatBool(true);
    }
    
    MUND FORCEINLINE HD u8 getFatXY() const { 
        return getFatX() * 5 + getFatY();
    }
    MUND FORCEINLINE HD u8 getFatXYFast() const { 
        return (getFatX() << 3) + getFatY();
    }
    
    MU FORCEINLINE u8 HD getColor(const u8 x, const u8 y) const {
        const u64 shift_amount = b1b2::getShiftAmount<u8, u8>(x, y);
        return (*(&b1 + (y >= 3)) >> shift_amount) & 0'7;
    }
    
    MU FORCEINLINE u8 HD getColor(const u64 x, const u64 y) const {
        const u64 shift_amount = b1b2::getShiftAmount<u64, u64>(x, y);
        return (*(&b1 + (y >= 3)) >> shift_amount) & 0'7;
    }

    MUND HD u64 getScore1(const B1B2& other) const;
    
    MUND HD i32 getExactRowColLowerBound(const B1B2& other) const;
    template<i32 MAX_DEPTH>
    MUND HD bool getExactRowColLowerBoundTill(const B1B2& other) const;
    MUND HD bool couldBeSolvedIn1Move(const B1B2 theOther) const;

    MU __host__ void doMoves(std::initializer_list<Action> theInitList);

    FORCEINLINE HD bool operator==(const B1B2& other) const {
        return b1 == other.b1 && b2 == other.b2;
    }
    
    FORCEINLINE HD bool operator<(const B1B2& other) const {
        return (b1 < other.b1) || (b1 == other.b1 && b2 < other.b2);
    }
    
    FORCEINLINE HD bool operator>(const B1B2& other) const {
        return (b1 > other.b1) || (b1 == other.b1 && b2 > other.b2);
    }
};






MU static constexpr u32 MEMORY_MOVE_TYPE_BITSIZE = 6;
MU static constexpr u64 MEMORY_MOVE_TYPE_MASK = 0'77;
MU static constexpr u32 MEMORY_MOVE_DATA_BITSIZE = 4;
MU static constexpr u64 MEMORY_MOVE_DATA_MASK = 0xF;

class Board;

class Memory {
    static HD u8 getShift(const u32 moveCount) {
        return MEMORY_MOVE_DATA_BITSIZE + moveCount * MEMORY_MOVE_TYPE_BITSIZE;
    }

    u64 mem;

public:

    HD Memory() : mem(0) {}

    HD Memory(MU const std::initializer_list<u64> moveValues) : mem(0) {
        for (const auto& moveValue : moveValues) {
            setNextNMove<1>(moveValue);
        }
    }

    // ############################################################
    // #                       u64 hash                           #
    // ############################################################

    MUND FORCEINLINE HD u64 getMem() const { return mem; }
    MU FORCEINLINE HD void setMem(const u64 value) { mem = value; }

    // ############################################################
    // #                       u64 moves                          #
    // ############################################################

    MUND FORCEINLINE HD u8 getMoveCount() const {
        return mem & MEMORY_MOVE_DATA_MASK;
    }

    MUND FORCEINLINE HD u8 getMove(const u8 index) const {
        return mem >> getShift(index) & MEMORY_MOVE_TYPE_MASK;
    }

    MUND FORCEINLINE HD u8 getLastMove() const {
        return mem >> getShift(getMoveCount() - 1) & MEMORY_MOVE_TYPE_MASK;
    }

    template<i32 COUNT>
    HD void setNextNMove(u64 moveValue);

    MU HD void setNextMoves(const std::initializer_list<u64> moveValues) {
        for (const auto& moveValue : moveValues) {
            setNextNMove<1>(moveValue);
        }
    }

    // ############################################################
    // #            To String -Similar- Functions                 #
    // ############################################################

    MUND std::string toString() const;

    MUND static std::string formatMoveString(u8 move, bool isForwards);

    MUND std::string asmString(const Memory* other) const;
    MUND std::string asmStringForwards() const;
    MUND std::string asmStringBackwards() const;

    MUND std::string asmFatString(u8 fatPos, const Memory* other, u8 fatPosOther) const;
    MUND std::string asmFatStringForwards(u8 fatPos) const;
    MUND std::string asmFatStringBackwards(u8 fatPos) const;

    MUND static std::vector<u8> parseNormMoveString(const std::string& input);
    MUND static std::vector<u8> parseFatMoveString(const std::string& input);
};

template<i32 COUNT>
HD void Memory::setNextNMove(const u64 moveValue) {
    static_assert(COUNT >= 1 && COUNT <= 5, "Template argument must be in range 1-5");

    constexpr u64 MOVE_SET_SHIFT = (5 - COUNT) * MEMORY_MOVE_TYPE_BITSIZE;
    constexpr u64 MOVE_SET_MASK = 0'77'77'77'77'77 >> MOVE_SET_SHIFT;

    const u32 moveCount = mem & MEMORY_MOVE_DATA_MASK;
    const u8 shiftAmount = getShift(moveCount);

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

        MU PrintSettings(const bool useAsciiIn, const ColorArray_t colorsIn)
            : useAscii(useAsciiIn), trueColors(colorsIn) {}
    };

    Memory memory;

    Board() = default;

    Board(const std::initializer_list<u8> values) { setState(values.begin()); }
    explicit Board(const u8 values[36]) { setState(values); }
    explicit Board(const u8 values[36], const u8 fatX, const u8 fatY) {
        setState(values);
        setFatXY(fatX, fatY);
    }

    MUND FORCEINLINE HD B1B2 asB1B2() const { return {b1, b2}; }
    
    MUND FORCEINLINE HD Memory& getMemory() { return memory; }
    MUND FORCEINLINE HD const Memory& getMemory() const { return memory; }

    MUND HD bool doActISColMatch(u8 x1, u8 y1, u8 m, u8 n) const;
    MUND HD u8 doActISColMatchBatched(u8 x1, u8 y1, u8 m) const;
    MUND HD static double getDuplicateEstimateAtDepth(u32 depth);
    MUND HD u64 getRowColIntersections(u32 x, u32 y) const;

    MUND HD u32 getRowCC() const;
    MUND HD u32 getColCC() const;

    MU __device__ void setRowColCC(u32* ptr) const;

    MU static void appendBoardToString(std::string& str, const Board* board, i32 curY, PrintSettings theSettings = {});
    MUND std::string toString(const Board& other, PrintSettings theSettings = {}) const;
    MUND std::string toString(const Board* other, PrintSettings theSettings = {}) const;
    MUND std::string toStringSingle(PrintSettings theSettings) const;
    MUND std::string toBlandString() const;

    FORCEINLINE HD bool operator==(const Board& other) const {
        return b1 == other.b1 && b2 == other.b2;
    }

    FORCEINLINE HD bool operator<(const Board& other) const {
        return (b1 < other.b1) || (b1 == other.b1 && b2 < other.b2);
    }
    
    FORCEINLINE HD bool operator>(const Board& other) const {
        return (b1 > other.b1) || (b1 == other.b1 && b2 > other.b2);
    }
};




extern template HD bool B1B2::getExactRowColLowerBoundTill<1>(const B1B2& other) const;
extern template HD bool B1B2::getExactRowColLowerBoundTill<2>(const B1B2& other) const;
extern template HD bool B1B2::getExactRowColLowerBoundTill<3>(const B1B2& other) const;
extern template HD bool B1B2::getExactRowColLowerBoundTill<4>(const B1B2& other) const;
extern template HD bool B1B2::getExactRowColLowerBoundTill<5>(const B1B2& other) const;

#ifdef USE_CUDA
namespace my_cuda {
    MU __constant__ extern u8 ROW_COL_OFFSETS[30];
}
#endif










class StateHash {
public:
    MUND FORCEINLINE HD static u64 computeHash(const B1B2& state) {
        const u64 x = state.b1;
        const u64 y = state.b2;
        return x ^ ((y << 1) | (y >> 63)) ^ (x >> 17) ^ (y << 13);
    }
};










