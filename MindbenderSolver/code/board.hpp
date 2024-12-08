#pragma once

#include "MindbenderSolver/utils/processor.hpp"

#include "memory.hpp"

#include <string>
#include <array>

// extern int GET_SCORE_3_CALLS;


/**
 * Holds the chuzzle colors and other information.
 */
struct B1B2 {
    typedef std::array<i8, 8> ColorArray_t;

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

    B1B2() = default;
    HD B1B2(u64 theB1, u64 theB2) : b1(theB1), b2(theB2) {}

    MU void setState(C u8 values[36]);
    MU std::array<i8, 8> setStateAndRetColors(C u8 values[36]);

    MU HD void setFatXY(u64 x, u64 y);
    MUND HD u8 getFatXY() C;
    MUND HD u8 getFatXYFast() C;

    MU HD void setFatBool(bool flag);
    MUND HD bool getFatBool() C;

    MU HD void setFatX(u64 x);
    MU HD void addFatX(u8 x);
    MUND HD u8 getFatX() C;

    MU HD void setFatY(u64 y);
    MU HD void addFatY(u8 y);
    MUND HD u8 getFatY() C;

    MUND HD u8 getColor(u8 x, u8 y) C;
    MUND HD u32 getColorCount() C;

    MUND HD u64 getScore1(C B1B2& other) C;
    MUND HD int getScore3(B1B2 theOther) C;

    template<int MAX_DEPTH>
    MUND HD bool getScore3Till(B1B2 theOther) C;

    MUND HD bool canBeSolvedIn1Move(B1B2 theOther) C;


    __forceinline HD bool operator==(C B1B2& other) C {
        return b1 == other.b1 && b2 == other.b2; }
};


class Board : public B1B2 {
public:
    typedef void (Board::*HasherPtr)();

    static ColorArray_t ColorsDefault;
    struct PrintSettings {
        bool useAscii;
        ColorArray_t trueColors = ColorsDefault;
        PrintSettings() : useAscii(true) {}
        MU PrintSettings(C bool useAscii, C ColorArray_t colors)
            : useAscii(useAscii), trueColors(colors) {}
    };

    Memory memory;

    explicit Board() = default;
    explicit Board(C std::initializer_list<u8> values);
    explicit Board(C u8 values[36]);
    explicit Board(C u8 values[36], u8 x, u8 y);

    MU HD B1B2 asB1B2() { return {b1, b2}; }

    MUND HD u64 getHash() C { return memory.getHash(); }
    MUND HD Memory& getMemory() { return memory; }
    MUND HD C Memory& getMemory() C { return memory; }

    // new generation of high IQ functions
    MUND HD bool doActISColMatch(u8 x1, u8 y1, u8 m, u8 n) C;
    MUND HD u8 doActISColMatchBatched(u8 x1, u8 y1, u8 m) C;
    MUND HD static double getDuplicateEstimateAtDepth(u32 depth);
    MUND HD u64 getRowColIntersections(u32 x, u32 y) C;



    MU HD void precomputeHash2();
    MU HD void precomputeHash3();
    MU HD void precomputeHash4();
    MUND HD HasherPtr getHashFunc() C;


    MU static void appendBoardToString(std::string &str, C Board *board, i32 curY, PrintSettings theSettings = {});
    MUND std::string toString(C Board& other, PrintSettings theSettings = {}) C;
    MUND std::string toString(C Board* other, PrintSettings theSettings = {}) C;
    MUND std::string toStringSingle(PrintSettings theSettings) C;
    MUND std::string toBlandString() C;


    __forceinline HD bool operator==(C Board& other) C {
        return b1 == other.b1 && b2 == other.b2; }

    __forceinline HD bool operator<(C Board& other) C {
        return this->getHash() < other.getHash(); }

    __forceinline HD bool operator>(C Board& other) C {
        return this->getHash() > other.getHash(); }

};


extern template HD bool B1B2::getScore3Till<1>(C B1B2 theOther) C;
extern template HD bool B1B2::getScore3Till<2>(C B1B2 theOther) C;
extern template HD bool B1B2::getScore3Till<3>(C B1B2 theOther) C;
extern template HD bool B1B2::getScore3Till<4>(C B1B2 theOther) C;
extern template HD bool B1B2::getScore3Till<5>(C B1B2 theOther) C;



