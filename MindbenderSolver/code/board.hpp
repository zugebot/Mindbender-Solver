#pragma once

#include "MindbenderSolver/utils/processor.hpp"
#include "memory.hpp"

#include <string>
#include <vector>



class Board {
public:
    typedef void (Board::*HasherPtr)();

    /**
     *  3 bits: fat x position
     *  1 bit : 1 if has a fat, otherwise 0
     *  4 bits: (# of colors - 1)
     *  2 bits: unused
     * 54 bits: holds upper 3x6 cell grid
     */
    u64 b1 = 0;
    /**
     *  3 bits: fat y position
     *  7 bits: unused
     * 54 bits: holds lower 3x6 cell grid
     */
    u64 b2 = 0;

    u64 hash = 0;

    Memory mem;

    explicit Board() = default;
    explicit Board(const u8 values[36]);
    explicit Board(const u8 values[36], u8 x, u8 y);

    MU void setState(const u8 values[36]);

    MU void setFat(u8 x, u8 y);
    MU void setFatX(u8 x);
    MU void setFatY(u8 y);
    MU void addFatX(u8 x);
    MU void addFatY(u8 y);
    MUND u8 getFatX() const;
    MUND u8 getFatY() const;
    MUND u8 getFatXY() const;
    MUND bool hasFat() const;

    MUND u8 getColor(u8 x, u8 y) const;

    MUND bool doActISColMatch(u8 x1, u8 y1, u8 m, u8 n) const;
    MUND u8 doActISColMatchBatched(u8 x1, u8 y1, u8 m) const;
    MUND static double getDuplicateEstimateAtDepth(u32 depth) ;


    // MUND static u32 getRow(const Board *board, u64 y); // WORKS (18-bits)
    // MUND static u32 getCol(const Board *board, u32 x); // WORKS (18-bits)
    // MUND static u32 constructMapCenter(u32 row, u32 x); // WORKS (18-bits)
    // MUND static u32 getScore1ShiftComp(u32 sect, u32 mapCent); // WORKS (6-bits)
    // static void shiftLeft(u32 &sect, u32 index); // WORKS (in-place on 6-bits)
    MUND u64 getRowColIntersections(u32 x, u32 y) const;

    MUND u32 getColorCount() const;


    MU void precomputeHash2();
    MU void precomputeHash3();
    MU void precomputeHash4();
    MUND HasherPtr getHashFunc() const;

    MUND u64 getScore1(const Board& other) const;
    MUND static u64 getScore2(const Board& other) ;

    MU static void appendBoardToString(std::string& str, const Board* board, c_i32 curY);
    MUND std::string toString() const;
    MUND std::string toString(const Board& other) const;
    MUND std::string toString(const Board* other) const;


    __forceinline bool operator==(const Board& other) const {
        MU static constexpr u64 MASK = 0x3FFFFFFFFFFFFF;
        return (b1) == (other.b1)
               && (b2) == (other.b2);
    }

    __forceinline bool operator<(const Board& other) const {
        return this->hash < other.hash;
    }


    __forceinline bool operator>(const Board& other) const {
        return this->hash > other.hash;
    }
};

typedef std::vector<Board> vecBoard_t;