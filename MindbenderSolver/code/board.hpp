#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include "memory.hpp"

#include <string>
#include <array>
#include <vector>


class Board {
public:
    typedef void (Board::*HasherPtr)();
    typedef std::array<i8, 8> ColorArray_t;
    static ColorArray_t ColorsDefault;
    struct PrintSettings {
        bool useAscii;
        ColorArray_t trueColors = ColorsDefault;
        PrintSettings() : useAscii(true) {}
        PrintSettings(bool useAscii, ColorArray_t colors)
            : useAscii(useAscii), trueColors(colors) {}
    };

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

    Memory memory;

    explicit Board() = default;
    explicit Board(c_u8 values[36]);
    explicit Board(c_u8 values[36], u8 x, u8 y);

    MU void setState(c_u8 values[36]);
    MU ColorArray_t setStateAndRetColors(c_u8 values[36]);

    MU void setFatXY(u64 x, u64 y);
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

    MUND u64 getHash() const { return memory.getHash(); }
    MUND Memory& getMemory() { return memory; }
    MUND const Memory& getMemory() const { return memory; }

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

    MU static void appendBoardToString(std::string &str, const Board *board, i32 curY, PrintSettings theSettings = PrintSettings());
    MUND std::string toString(const Board& other, PrintSettings theSettings = PrintSettings()) const;
    MUND std::string toString(const Board* other, PrintSettings theSettings = PrintSettings()) const;
    MUND std::string toStringSingle(PrintSettings theSettings) const;
    MUND std::string toBlandString() const;


    __forceinline bool operator==(const Board& other) const {
        return b1 == other.b1 && b2 == other.b2; }

    __forceinline bool operator<(const Board& other) const {
        return this->getHash() < other.getHash(); }

    __forceinline bool operator>(const Board& other) const {
        return this->getHash() > other.getHash(); }
};

typedef std::vector<Board> vecBoard_t;
typedef const Board c_Board;