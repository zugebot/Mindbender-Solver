#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include "memory.hpp"

#include <string>
#include <array>
#include <vector>


class Board;


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
    typedef void (HashMem::*HasherPtr)(u64, u64);

    explicit HashMem() = default;

    MUND Memory& getMemory() { return mem; }
    MUND const Memory& getMemoryConst() const { return mem; }

    MUND u64 getHash() const { return hash; }
    MU void setHash(c_u64 value) { hash = value; }

    MU void precomputeHash2(c_u64 b1, c_u64 b2);
    MU void precomputeHash3(c_u64 b1, c_u64 b2);
    MU void precomputeHash4(c_u64 b1, c_u64 b2);
    MUND static HasherPtr getHashFunc(const Board& board) ;


    __forceinline bool operator==(const HashMem& other) const {
        return this->getHash() == other.getHash(); }

    __forceinline bool operator<(const HashMem& other) const {
        return this->getHash() < other.getHash(); }

    __forceinline bool operator>(const HashMem& other) const {
        return this->getHash() > other.getHash(); }

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
    MU std::array<i8, 8> setStateAndRetColors(c_u8 values[36]);

    MU void setFatXY(u8 x, u8 y);
    MUND u8 getFatXY() const;
    MUND u8 getFatXYFast() const;

    MU void setFatBool(bool flag);
    MUND bool getFatBool() const;

    MU void setFatX(u64 x);
    MU void addFatX(u8 x);
    MUND u8 getFatX() const;

    MU void setFatY(u64 y);
    MU void addFatY(u8 y);
    MUND u8 getFatY() const;

    MUND u8 getColor(u8 x, u8 y) const;

    MUND u64 getHash() const { return hashMem.getHash(); }
    MUND Memory& getMemory() { return hashMem.getMemory(); }
    MUND const Memory& getMemory() const { return hashMem.getMemoryConst(); }

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

    MU static void appendBoardToString(std::string& str, const Board* board, c_i32 curY, std::array<i8, 8> trueColors={0,1,2,3,4,5,6,7}, bool printASCII = true);
    MUND std::string toBlandString() const;
    MUND std::string toString(const Board& other, bool printASCII = true, std::array<i8, 8> trueColors={0,1,2,3,4,5,6,7}) const;
    MUND std::string toString(const Board* other, bool printASCII = true, std::array<i8, 8> trueColors={0,1,2,3,4,5,6,7}) const;
    MUND std::string toStringSingle(bool printASCII = true, std::array<i8, 8> trueColors={0,1,2,3,4,5,6,7}) const;


    __forceinline bool operator==(const Board& other) const {
        return b1 == other.b1 && b2 == other.b2; }

    __forceinline bool operator<(const Board& other) const {
        return this->getHash() < other.getHash(); }

    __forceinline bool operator>(const Board& other) const {
        return this->getHash() > other.getHash(); }
};

typedef std::vector<Board> vecBoard_t;
typedef const Board c_Board;