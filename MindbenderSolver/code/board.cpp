#include "board.hpp"

#include "rotations.hpp"

#include <string>
#include <immintrin.h>

#include "MindbenderSolver/utils/colors.hpp"


u64 prime_func1(c_u64 b1, c_u64 b2) {
    static constexpr u64 MASK = 0'777777'777777'777777;
    static constexpr u64 prime = 31;
    u64 hash = 17;
    hash = hash * prime + (b1 & MASK ^ (b1 & MASK) >> 32);
    hash = hash * prime + (b2 & MASK ^ (b2 & MASK) >> 32);
    return hash;
}


// check commits before 10/16/24 for previous impl.
u64 getSegment2bits(c_u64 segment) {
    static constexpr u64 MASK_X0 = 0'111111'111111'111111;
    return _pext_u64(segment, MASK_X0);
}


// check commits before 10/16/24 for previous impl.
u64 getSegment3bits(c_u64 segment) {
    static constexpr u64 MASK_AS = 0'300300'300300'300300;
    static constexpr u64 MASK_BS = 0'030030'030030'030030;
    static constexpr u64 MASK_CS = 0'003003'003003'003003;
    c_u64 o1 = ((segment & MASK_AS) >> 6) * 9 |
               ((segment & MASK_BS) >> 3) * 3 |
               segment & MASK_CS;
    static constexpr u64 MASK_X23 = 0'037037'037037'037037;
    c_u64 x23 = _pext_u64(o1, MASK_X23);
    return x23;
}


// check commits before 10/16/24 for previous impl.
u64 getSegment4bits(c_u64 segment) {
    static constexpr u64 MASK_X0 = 0'333333'333333'333333;
    return _pext_u64(segment, MASK_X0);
}


void HashMem::precomputeHash2(c_u64 b1, c_u64 b2) {
    c_u64 above = getSegment2bits(b1);
    c_u64 below = getSegment2bits(b2);
    setHash(above << 18 | below);
}


void HashMem::precomputeHash3(c_u64 b1, c_u64 b2) {
    c_u64 above = getSegment3bits(b1);
    c_u64 below = getSegment3bits(b2);
    setHash(above << 30 | below);
}


void HashMem::precomputeHash4(c_u64 b1, c_u64 b2) {
    setHash(prime_func1(b2, b1));
}


MU HashMem::HasherPtr HashMem::getHashFunc(const Board& board) {
    c_u64 colorCount = board.getColorCount();
    if (board.getFatBool() || colorCount > 3) {
        return &HashMem::precomputeHash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return &HashMem::precomputeHash2;
    }
    return &HashMem::precomputeHash3;
}




Board::Board(const u8 values[36]) {
    setState(values);
}


Board::Board(const u8 values[36], c_u8 x, c_u8 y) {
    setState(values);
    setFatXY(x, y);
}


void Board::setState(c_u8 values[36]) {
    std::array<i8, 8> colors = {8, 8, 8, 8, 8, 8, 8, 8};
    for (int i = 0; i < 36; i++) {
        c_int val = values[i] & 0'7;
        colors[val] = 1;
    }
    u64 colorCount = 0;
    for (i8& color : colors) {
        if (color != 8) {
            color = static_cast<i8>(colorCount);
            colorCount++;
        }
    }
    u8 adjusted_values[36] = {};
    for (int i = 0; i < 36; i++) {
        adjusted_values[i] = colors[values[i]];
    }

    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = b1 << 3 | adjusted_values[i] & 0'7;
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = b2 << 3 | adjusted_values[i] & 0'7;
    }

    static constexpr u64 EVERYTHING_BUT_COLOR = 0xF0FF'FFFF'FFFF'FFFF;
    b1 = b1 & EVERYTHING_BUT_COLOR | colorCount << 56;
}


MU std::array<i8, 8> Board::setStateAndRetColors(c_u8 values[36]) {
    std::array<i8, 8> colors = {8, 8, 8, 8, 8, 8, 8, 8};
    std::array<i8, 8> trueColors = {8, 8, 8, 8, 8, 8, 8, 8};

    for (int i = 0; i < 36; i++) {
        c_int val = values[i] & 0'7;
        colors[val] = 1;
    }
    u64 colorCount = 0;
    i8 index = 0;
    for (i8& color : colors) {
        if (color != 8) {
            trueColors[colorCount] = index;
            color = static_cast<i8>(colorCount);
            colorCount++;
        }
        index++;
    }
    u8 adjusted_values[36] = {};
    for (int i = 0; i < 36; i++) {
        adjusted_values[i] = colors[values[i]];
    }

    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = b1 << 3 | adjusted_values[i] & 0'7;
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = b2 << 3 | adjusted_values[i] & 0'7;
    }


    static constexpr u64 EVERYTHING_BUT_COLOR = 0xF0FF'FFFF'FFFF'FFFF;
    b1 = b1 & EVERYTHING_BUT_COLOR | colorCount << 56;

    return trueColors;
}





u32 Board::getColorCount() const {
    c_u64 colorCount = b1 >> 56 & 0xF;
    return colorCount;
}







template<typename T>
u64 cast_u64(T var) {
    return static_cast<u64>(var);
}


static constexpr u64 MASK_FAT_POS = 0x1FFF'FFFF'FFFF'FFFF;

/**
 *
 * @param x value 0-4
 * @param y value 0-4
 */
void Board::setFatXY(c_u8 x, c_u8 y) {
    b1 = b1 & MASK_FAT_POS | cast_u64(x) << 61;
    b2 = b2 & MASK_FAT_POS | cast_u64(y) << 61;
    setFatBool(true);
}


MU void Board::setFatBool(c_bool flag) {
    static constexpr u64 MASK_FAT_FLAG = 0xEFFF'FFFF'FFFF'FFFF;
    b1 = b1 & MASK_FAT_FLAG | cast_u64(flag) << 60;

}

MU void Board::setFatX(c_u64 x) {
    b1 = b1 & MASK_FAT_POS | x << 61;
}


MU void Board::setFatY(c_u64 y) {
    b2 = b2 & MASK_FAT_POS | y << 61;
}


MU void Board::addFatX(c_u8 x) {
    u64 cur_x = getFatX() + x;
    cur_x -= 6 * (cur_x > 5);
    b1 = b1 & MASK_FAT_POS | cur_x << 61;
}


MU void Board::addFatY(c_u8 y) {
    u64 cur_y = getFatY() + y;
    cur_y -= 6 * (cur_y > 5);
    b2 = b2 & MASK_FAT_POS | cur_y << 61;
}


u8 Board::getFatX() const {
    return (b1 & ~MASK_FAT_POS) >> 61;
}


u8 Board::getFatY() const {
    return (b2 & ~MASK_FAT_POS) >> 61;
}


/// always returns a value between 0-24.
u8 Board::getFatXY() const {
    return (b1 >> 61) * 5 + (b2 >> 61);
}

u8 Board::getFatXYFast() const {
    return ((b1 >> 61) << 3) + (b2 >> 61);
}


bool Board::getFatBool() const {
    c_bool state = (b1 >> 60 & 1) != 0;
    return state;
}














u8 Board::getColor(c_u8 x, c_u8 y) const {
    c_i32 shift_amount = 51 - x * 3 - y % 3 * 18;
    return *(&b1 + (y >= 3)) >> shift_amount & 0'7;
}


/**
 * int x = (action1 % 30) / 5;
 * int y = (action2 % 30) / 5;
 * int m = 1 + action1 % 5;
 * int n = 1 + action2 % 5;
 */
bool Board::doActISColMatch(c_u8 x1, c_u8 y1, c_u8 m, c_u8 n) const {
    c_int y2 = (y1 - n + 6) % 6;
    c_int x2 = (x1 - m + 6) % 6;

    c_u8 x1_3 = x1 * 3;
    c_int offset_shared = 51 - (y1 % 3) * 18;
    c_int shift_amount1 = x1_3 + offset_shared;
    c_int shift_amount3 = x2 * 3 + offset_shared;

    c_u64 base = y1 < 3 ? b1 : b2;

    c_u8 color1 = base >> shift_amount1;
    c_u8 color3 = base >> shift_amount3;

    if ((color1 ^ color3) & 0'7) {
        return false;
    }
    c_int shift_amount2 = 51 - x1_3 - y2 % 3 * 18;
    c_u64 base2 = y2 < 3 ? b1 : b2;
    c_u8 color2 = base2 >> shift_amount2;

    return (color1 ^ color2) & 0'7;
}


/**
 *
 * @param x1 Sect: Column
 * @param y1 Sect: Row
 * @param m amount: Column
 *          amount: Row (finds true for all of these)
 * @return
 */
u8 Board::doActISColMatchBatched(c_u8 x1, c_u8 y1, c_u8 m) const {
    c_i32 x2 = (x1 - m + 6) % 6;
    c_u64 base = y1 < 3 ? b1 : b2;
    c_i32 offset_shared = 51 - y1 % 3 * 18;
    c_u8 color1 = base >> (x1 * 3 + offset_shared);
    c_u8 color3 = base >> (x2 * 3 + offset_shared);

    if ((color1 ^ color3) & 0'7) { return 0; }

    u8 results = 0;
    c_i32 offset_shared2 = 51 - x1 * 3;
    for (i32 i = -5; i < 1; i++) {
        c_i32 y2 = (y1 - i) % 6;
        c_i32 y3 = ((y1 - i) % 3) * 18;
        c_u64 base2 = y2 < 3 ? b1 : b2;
        c_u8 color2 = base2 >> (offset_shared2 - y3);
        results |= (((color1 ^ color2) & 07) != 0) << (i + 5);
    }

    return results;
}


double Board::getDuplicateEstimateAtDepth(MU u32 depth) {
    return 1.0;
}


/**
 * returns the ..100.000.100.000... of board1 compared to board2
 * ..001.. if cells are similar in value
 * ..000.. if cells differ in value
 * @param sect1 b1/b2 of 1st board
 * @param sect2 b1/b2 of 2nd board
 * @return
 */
inline u64 getSimilar54(c_u64& sect1, c_u64& sect2) {
    c_u64 s = sect1 ^ sect2;
    return ~(s | s >> 1 | s >> 2) & 0'111111'111111'111111;
}


u64 Board::getScore1(const Board &other) const {
    return __builtin_popcountll(getSimilar54(b1, other.b1))
         + __builtin_popcountll(getSimilar54(b2, other.b2));
}


u64 Board::getRowColIntersections(c_u32 x, c_u32 y) const {
    static constexpr u64 C_MAIN_MASK = 0'000007'000007'000007;
    static constexpr u32 C_CNTR_MASKS[8] = {
            0x00000000, 0x02108421, 0x04210842, 0x06318C63,
            0x08421084, 0x0A5294A5, 0x0C6318C6, 0x0E739CE7};
    c_u32 left = 15 - x * 3;
    c_u32 row = *(&b1 + (y >= 3)) >> (2 - y - 3 * (y >= 3)) * 18 & 0'777777;
    c_u32 cntr_p1_r = row >> left & 0'7;

    // find col_x5
    c_u64 col_mask = C_MAIN_MASK << left;
    c_u64 b1_c = (b1 & col_mask) >> left;
    c_u64 b2_c = (b2 & col_mask) >> left;
    c_u32 shifted_5 = (b2_c | b2_c >> 13 | b2_c >> 26) & 0x1CE7 |
                      (b1_c << 15 | b1_c << 2 | b1_c >> 11) & 0xE738000;
    c_u32 s = shifted_5 ^ C_CNTR_MASKS[cntr_p1_r];
    c_u32 sim = ((~(s | s >> 1 | s >> 2)) & C_CNTR_MASKS[1]) * 31;
    c_u32 col_x5 = (sim & (0x3FFFFFFF << (5 * (6 - y)))) >> 5
                   | sim & (0x1FFFFFF >> 5 * y);

    // find row_x5
    c_u32 s_ps = row ^ (cntr_p1_r * 0'111111);
    c_u32 sim_r = ~(s_ps | s_ps >> 1 | s_ps >> 2) & 0'111111;
    c_u32 p1_r = (sim_r & 0'101010) >> 2 | sim_r & 0'10101;
    c_u32 row_t1 = (p1_r >> 8 | p1_r >> 4 | p1_r) & 0'77;
    c_u32 row_x5 = ((row_t1 & (0'7700 >> x)) >> 1 | row_t1 & (0'37 >> x)) * 0x108421;

    return col_x5 & row_x5;
}


void Board::precomputeHash2() {
    c_u64 above = getSegment2bits(b1);
    c_u64 below = getSegment2bits(b2);
    hashMem.setHash(above << 18 | below);
}


void Board::precomputeHash3() {
    c_u64 above = getSegment3bits(b1);
    c_u64 below = getSegment3bits(b2);
    hashMem.setHash(above << 30 | below);
}


void Board::precomputeHash4() {
    hashMem.setHash(prime_func1(b2, b1));
}


Board::HasherPtr Board::getHashFunc() const {
    c_u64 colorCount = getColorCount();
    if (getFatBool() || colorCount > 3) {
        return &Board::precomputeHash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return &Board::precomputeHash2;
    }
    return &Board::precomputeHash3;
}


void Board::appendBoardToString(std::string& str, const Board* board, c_i32 curY,
                                std::array<i8, 8> trueColors, bool printASCII) {
    c_bool isFat = board->getFatBool();
    c_u8 curFatX = board->getFatX();
    c_u8 curFatY = board->getFatY();
    bool inMiddle = false;

    u64 board_b;
    if (curY == 0 || curY == 1 || curY == 2) {
        board_b = board->b1;
    } else if (curY == 3 || curY == 4 || curY == 5) {
        board_b = board->b2;
    } else {
        return;
    }

    for (int x = 0; x < 18; x += 3) {
        c_u8 value = trueColors[board_b >> (51 - x - (curY % 3) * 18) & 0'7];
        if (isFat) {
            c_u32 curX = x / 3;
            if (curFatX == curX || curFatX == curX - 1) {
                if (curFatY == curY || curFatY == curY - 1) {
                    if (printASCII)
                        str.append(Colors::getBgColor(value));
                    inMiddle = curFatX == curX;
                }
            }
        }
        if (printASCII)
            str.append(Colors::getColor(value));
        str.append(std::to_string(value));
        if (inMiddle) {
            if (x != 15) { str.append(" "); }
            if (printASCII)
                str.append(Colors::bgReset);
        } else {
            if (printASCII)
                str.append(Colors::bgReset);
            if (x != 15) { str.append(" "); }
        }
    }
}


MUND std::string Board::toBlandString() const {
    return toStringSingle(false);
}


MUND std::string Board::toString(const Board& other, bool printASCII, std::array<i8, 8> trueColors) const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, trueColors, printASCII);
        str.append("   ");
        appendBoardToString(str, &other, i, trueColors, printASCII);
        str.append("\n");
    }
    return str;
}


MUND std::string Board::toString(const Board* other, bool printASCII, std::array<i8, 8> trueColors) const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, trueColors, printASCII);
        str.append("   ");
        appendBoardToString(str, other, i, trueColors, printASCII);
        str.append("\n");
    }
    return str;
}


std::string Board::toStringSingle(bool printASCII, std::array<i8, 8> trueColors) const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, trueColors, printASCII);
        str.append("\n");
    }
    return str;
}


