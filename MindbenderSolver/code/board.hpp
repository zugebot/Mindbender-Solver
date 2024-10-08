#pragma once

#include "MindbenderSolver/utils/processor.hpp"
#include "memory.hpp"

#include <string>
#include <vector>


/**
 * Todo:
 * make it so that
 * 1-bit. - isSolved
 * 1-bit. - has Fat
 * 3-bits - fat X
 * 3-bits - fat Y
 * 3/4bit - color count
 * this would leave mem to having a max of 8 moves.
 */
class HashMem {
    u64 hash = 0;
    Memory mem;
public:

    explicit HashMem() = default;

    MUND Memory& getMemory() { return mem; }
    MUND const Memory& getMemory() const { return mem; }

    MUND u64 getHash() const { return hash; }
    MU void setHash(u64 value) { hash = value; }
};



class Board {
public:
    typedef void (Board::*HasherPtr)();

    /**
     *  3 bits: fat x position
     *  1 bit : 1 if has a fat, otherwise 0
     *  4 bits: (# of colors - 1)
     *  2 bits: unused
     * 54 bits: holds upper 3x6 cell grid (3 bits each)
     */
    u64 b1 = 0;
    /**
     *  3 bits: fat y position
     *  7 bits: unused
     * 54 bits: holds lower 3x6 cell grid (3 bits each)
     */
    u64 b2 = 0;



    HashMem hashMem;


    explicit Board() = default;
    explicit Board(c_u8 values[36]);
    explicit Board(c_u8 values[36], u8 x, u8 y);

    MU void setState(c_u8 values[36]);

    MU void setFatXY(u8 x, u8 y);
    MUND u8 getFatXY() const;

    MU void setFatBool(bool flag);
    MUND bool getFatBool() const;

    MU void setFatX(u8 x);
    MU void addFatX(u8 x);
    MUND u8 getFatX() const;

    MU void setFatY(u8 y);
    MU void addFatY(u8 y);
    MUND u8 getFatY() const;


    MUND u8 getColor(u8 x, u8 y) const;


    MUND u64 getHash() const { return hashMem.getHash(); }
    MUND Memory& getMemory() { return hashMem.getMemory(); }
    MUND const Memory& getMemory() const { return hashMem.getMemory(); }

    // new generation of high IQ functions
    MUND bool doActISColMatch(u8 x1, u8 y1, u8 m, u8 n) const;
    MUND u8 doActISColMatchBatched(u8 x1, u8 y1, u8 m) const;
    MUND static double getDuplicateEstimateAtDepth(u32 depth);
    MUND u64 getRowColIntersections(u32 x, u32 y) const;

    MUND u32 getColorCount() const;


    MU void precomputeHash2();
    MU void precomputeHash3();
    MU void precomputeHash4();
    MUND HasherPtr getHashFunc() const;

    MUND u64 getScore1(const Board& other) const;

    MU static void appendBoardToString(std::string& str, const Board* board, c_i32 curY);
    MUND std::string toString() const;
    MUND std::string toString(const Board& other) const;
    MUND std::string toString(const Board* other) const;


    __forceinline bool operator==(const Board& other) const {
        return b1 == other.b1 && b2 == other.b2; }

    __forceinline bool operator<(const Board& other) const {
        return this->getHash() < other.getHash(); }

    __forceinline bool operator>(const Board& other) const {
        return this->getHash() > other.getHash(); }
};

typedef std::vector<Board> vecBoard_t;
typedef const Board c_Board;