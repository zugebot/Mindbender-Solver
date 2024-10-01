#include "board.hpp"

#include <string>
#include <iostream>

#include "MindbenderSolver/utils/colors.hpp"


Board::Board(const u8 values[36]) {
    setState(values);
}


Board::Board(const u8 values[36], c_u8 x, c_u8 y) {
    setState(values);
    setFat(x, y);
}


void Board::setState(c_u8 values[36]) {
    i8 colors[8] = {8, 8, 8, 8, 8, 8, 8, 8};
    for (int i = 0; i < 36; i++) {
        c_int val = values[i] & 0b111;
        colors[val] = 1;
    }
    u64 colorCount = 0;
    for (i8& color : colors) {
        if (color != 8) {
            color = static_cast<i8>(colorCount);
            colorCount++;
        }
    }
    u8 adjusted_values[36] = {0};
    for (int i = 0; i < 36; i++) {
        adjusted_values[i] = colors[values[i]];
    }

    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = b1 << 3 | (adjusted_values[i] & 0b111);
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = b2 << 3 | (adjusted_values[i] & 0b111);
    }


    static constexpr u64 EVERYTHING_BUT_COLOR = 0xF0FF'FFFF'FFFF'FFFF;
    b1 = (b1 & EVERYTHING_BUT_COLOR) | (colorCount << 56);
}


u32 Board::getColorCount() const {
    c_u64 colorCount = ((b1 >> 56) & 0xF);
    return colorCount;
}



/**
 *
 * @param x value 0-4
 * @param y value 0-4
 */
void Board::setFat(c_u8 x, c_u8 y) {
    static constexpr u64 MASK = 0x0FFF'FFFF'FFFF'FFFF;
    b1 = b1 & MASK | (static_cast<u64>(x) << 61) | (1LL << 60);
    b2 = b2 & MASK | (static_cast<u64>(y) << 61);
}


MU void Board::setFatX(c_u8 x) {
    static constexpr u64 MASK = 0x1FFF'FFFF'FFFF'FFFF;
    b1 = b1 & MASK | static_cast<u64>(x) << 61;
}


MU void Board::setFatY(c_u8 y) {
    static constexpr u64 MASK = 0x1FFF'FFFF'FFFF'FFFF;
    b2 = b2 & MASK | static_cast<u64>(y) << 61;
}


MU void Board::addFatX(c_u8 x) {
    static constexpr u64 MASK = ~0x1FFF'FFFF'FFFF'FFFF;
    uint64_t cur_x = (b1 & MASK) >> 61;
    cur_x += x;
    cur_x -= 6 * (cur_x > 5);
    b1 = (b1 & ~MASK) | (cur_x << 61);
}


MU void Board::addFatY(c_u8 y) {
    static constexpr u64 MASK = ~0x1FFF'FFFF'FFFF'FFFF;
    uint64_t cur_y = (b2 & MASK) >> 61;
    cur_y += y;
    cur_y -= 6 * (cur_y > 5);
    b2 = (b2 & ~MASK) | (cur_y << 61);
}


u8 Board::getFatX() const {
    static constexpr u64 MASK = ~0x1FFF'FFFF'FFFF'FFFF;
    return (b1 & MASK) >> 61;
}


u8 Board::getFatY() const {
    static constexpr u64 MASK = ~0x1FFF'FFFF'FFFF'FFFF;
    return (b2 & MASK) >> 61;
}


/// always returns a value between 0-24.
u8 Board::getFatXY() const {
    return (b1 >> 61) * 5 + (b2 >> 61);
}


bool Board::hasFat() const {
    c_bool state = ((b1 >> 60) & 1) != 0;
    return state;
}


u8 Board::getColor(c_u8 x, c_u8 y) const {
    c_i32 shift_amount = 51 - x * 3 - (y % 3) * 18;
    if (y < 3) {
        return (b1 >> shift_amount) & 0b111;
    }
    return (b2 >> shift_amount) & 0b111;
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

    if ((color1 ^ color3) & 0b111) {
        return false;
    }
    c_int shift_amount2 = 51 - x1_3 - (y2 % 3) * 18;
    c_u64 base2 = y2 < 3 ? b1 : b2;
    c_u8 color2 = base2 >> shift_amount2;

    return (color1 ^ color2) & 0b111;
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
    c_i32 offset_shared = 51 - (y1 % 3) * 18;
    c_u8 color1 = base >> (x1 * 3 + offset_shared);
    c_u8 color3 = base >> (x2 * 3 + offset_shared);

    if ((color1 ^ color3) & 0b111) { return 0; }

    u8 results = 0;
    c_i32 offset_shared2 = 51 - x1 * 3;
    for (i32 i = -5; i < 1; i++) {
        c_i32 y2 = (y1 - i) % 6;
        c_i32 y3 = ((y1 - i) % 3) * 18;
        c_u64 base2 = y2 < 3 ? b1 : b2;
        c_u8 color2 = base2 >> (offset_shared2 - y3);
        results |= (((color1 ^ color2) & 0b111) != 0) << (i + 5);
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
    return ~(s | s >> 1 | s >> 2) & 0x9'2492'4924'9249;
}


u64 score1Helper(c_u64& sect) {
    static constexpr u64 M3 = 0x0000'E070'381C'0E07;
    static constexpr u64 M4 = 0x0000'0000'7800'000F;
    static constexpr u64 M5 = 0x0000'0000'07FF'FFFF;

    c_u64 p1 = sect + (sect >> 3) + (sect >> 6) & M3;
    c_u64 p2 = p1 + (p1 >> 9) + (p1 >> 18) & M4;
    return (p2 & M5) + (p2 >> 27);
}

u64 Board::getScore1(const Board &other) const {
    return score1Helper(getSimilar54(b1, other.b1))
         + score1Helper(getSimilar54(b2, other.b2));
}

/**
 * WORKS (18 bits)
 */
u32 getRow(const Board *board, c_u64 y) {

    return *(&board->b1 + (y >= 3)) >> (2 - y % 3) * 18 & 0x3FFFF;

}

/**
 * does 5 - x
 * WORKS (18 bits)
 */
u32 getCol(const Board *board, c_u32 x) {
    c_u64 col_map = 0x70'001C'0007 << (5 - x) * 3;
    c_u32 xMult = 3 * x;
    c_u64 b1col = board->b1 & col_map;
    c_u64 b2col = board->b2 & col_map;
    c_u64 col = b1col >> (45 - xMult - 9)
              | b1col >> (30 - xMult - 9)
              | b1col >> (15 - xMult - 9)
              | b2col >> (45 - xMult)
              | b2col >> (30 - xMult)
              | b2col >> (15 - xMult)
    ;
    return col;

}

/**
 * does 5 - x
 * WORKS (18 bits)
 */
u32 constructMapCenter(const u32 row, c_u32 x) {
    c_u64 cntr_p1 = row >> (5 - x) * 3 & 0b111;
    c_u64 cntr_p2 = cntr_p1 | cntr_p1 << 3 | cntr_p1 << 6;
    return cntr_p2 | cntr_p2 << 9;
}


inline u32 getSimilar18(c_u32& sect1, c_u32& sect2) {
    c_u32 s = sect1 ^ sect2;
    return ~(s | s >> 1 | s >> 2) & 0x9249;
}


/**
 * WORKS (6 bits)
 * If an issue occurs, check this function first
 * it may stem from not using a mask on p1->p2 inputs
 *
 * @param sect either the row or column (18-bit value)
 * @param mapCent the map center row (18-bit value)
 * @return
 */
u32 getScore1ShiftComp(c_u32 sect, c_u32 mapCent) {
    c_u32 sim = getSimilar18(sect, mapCent);
    c_u32 p1 = (sim & 0x1041 << 3) >> 2 | sim & 0x1041;
    c_u32 p2 = (p1 >> 8 | p1 >> 4 | p1) & 0x3F;
    return p2;
}



/**
 * does 6 - index, 1 + index
 */
void shiftLeft3(u32 &sect, c_u32 index) {
    sect = (sect & 0x3FFFF << 3 * (6 - index)) >> 3 | sect & 0x3FFFF >> 3 * (1 + index);
}


/**
 *
 */
u64 Board::getRowColIntersections(c_u32 x, c_u32 y) const {
    c_u32 row = *(&b1 + (y >= 3)) >> (2 - y % 3) * 18 & 0x3FFFF;
    c_u64 cntr_p1_r = row >> (5 - x) * 3 & 0b111;
    c_u64 cntr_p2_r = cntr_p1_r | cntr_p1_r << 3 | cntr_p1_r << 6;
    c_u32 s_ps = row ^ (cntr_p2_r | cntr_p2_r << 9);          // v getScore1ShiftComp(row, mapCent);
    c_u32 sim_r = ~(s_ps | s_ps >> 1 | s_ps >> 2) & 0x9249;   // ^ getSimilar18
    c_u32 p1_r = (sim_r & 0x1041 << 3) >> 2 | sim_r & 0x1041; // |
    c_u32 row_t1 = (p1_r >> 8 | p1_r >> 4 | p1_r) & 0x3F;              // ^
    c_u32 row_t2 = (row_t1 & 0x3F << 6 - x) >> 1 | row_t1 & 0x3F >> 1 + x; // shiftLeft1
    c_u32 row_x5 = row_t2 | row_t2 << 5 | row_t2 << 10 | row_t2 << 15 | row_t2 << 20;



    // u32 col = getCol(this, x);
    c_u64 col_map = 0x70'001C'0007 << (5 - x) * 3;
    c_u64 b1col = b1 & col_map;
    c_u64 b2col = b2 & col_map;
    // these offsets will change
    c_u32 xMult = 3 * x;
    u64 col = b1col >> (45 - xMult - 9)
            | b1col >> (30 - xMult - 9)
            | b1col >> (15 - xMult - 9)
            | b2col >> (45 - xMult)
            | b2col >> (30 - xMult)
            | b2col >> (15 - xMult);

    // c_u32 col_center = constructMapCenter(col, y);
    c_u64 cntr_p1_c = row >> (5 - x) * 3 & 0b111;
    c_u64 cntr_p2_c = cntr_p1_c | cntr_p1_c << 3 | cntr_p1_c << 6; // dif
    c_u32 cntr_col = cntr_p2_c | cntr_p2_c << 9; // dif

    // shiftLeft3
    col = (col & 0x3FFFF << 3 * (6 - y)) >> 3 | col & 0x3FFFF >> 3 * (1 + y); // dif


    c_u32 sim_c = getSimilar18(col, cntr_col); // dif
    c_u32 p1_c = (sim_c & 0x1041 << 3) >> 2 | sim_c & 0x1041; // dif
    col = (p1_c >> 8 | p1_c >> 4 | p1_c) & 0x3F; // dif
    // these offsets will change
    c_u32 col_sg = (col & 0b00001) << 0
                 | (col & 0b00010) << 4
                 | (col & 0b00100) << 8
                 | (col & 0b01000) << 12
                 | (col & 0b10000) << 16;
    u32 col_x5 = col_sg << 0 | col_sg << 1 | col_sg << 2;
    col_x5 = col_x5 | col_x5 << 3;



    c_u64 final = col_x5 & row_x5;
    return final;
}














uint64_t prime_func1(c_u64 b1, c_u64 b2) {
    static constexpr u64 MASK = 0x003F'FFFF'FFFF'FFFF;
    static constexpr u64 prime = 31;
    uint64_t hash = 17;
    hash = hash * prime + ((b1 & MASK) ^ (b1 & MASK) >> 32);
    hash = hash * prime + ((b2 & MASK) ^ (b2 & MASK) >> 32);
    return hash;
}


uint64_t getSegment2bits(const uint64_t segment) {
    static constexpr uint64_t MASK_A1 = 0b001000'001000'001000'001000'001000'001000'001000'001000'001000;
    static constexpr uint64_t MASK_B1 = MASK_A1 >> 3;
    static constexpr uint64_t MASK_A2 = 0b000011'000000'000000'000011'000000'000000'000011'000000'000000;
    static constexpr uint64_t MASK_B2 = MASK_A2 >> 6;
    static constexpr uint64_t MASK_C2 = MASK_A2 >> 12;
    static constexpr uint64_t MASK_A3 = 0b000000'000000'111111'000000'000000'000000'000000'000000'000000;
    static constexpr uint64_t MASK_B3 = MASK_A3 >> 18;
    static constexpr uint64_t MASK_C3 = MASK_A3 >> 36;
    const uint64_t o1 = (segment & MASK_A1) >> 2 | (segment & MASK_B1);
    const uint64_t o2 = (o1 & MASK_A2) >> 8 | (o1 & MASK_B2) >> 4 | (o1 & MASK_C2);
    const uint64_t o3 = (o2 & MASK_A3) >> 24 | (o2 & MASK_B3) >> 12 | (o2 & MASK_C3);
    return o3;
}


uint64_t getSegment3bits(const uint64_t segment) {
    static constexpr uint64_t MASK_AS = 0b011000000'011000000'011000000'011000000'011000000'011000000;
    static constexpr uint64_t MASK_BS = MASK_AS >> 3;
    static constexpr uint64_t MASK_CS = MASK_AS >> 6;
    static constexpr uint64_t MASK_A1 = 0b000011111'000000000'000011111'000000000'000011111'000000000;
    static constexpr uint64_t MASK_B1 = MASK_A1 >> 9;
    static constexpr uint64_t MASK_A2 = 0b000000001'111111111'000000000'000000000'000000000'000000000;
    static constexpr uint64_t MASK_B2 = MASK_A2 >> 18;
    static constexpr uint64_t MASK_C2 = MASK_A2 >> 36;

    const uint64_t o1 = ((segment & MASK_AS) >> 6) * 9 | ((segment & MASK_BS) >> 3) * 3 | ((segment & MASK_CS));

    const uint64_t o2 = ((o1 & MASK_A1) >> 4) | (o1 & MASK_B1);

    const uint64_t o3 = ((o2 & MASK_A2) >> 16) | ((o2 & MASK_B2) >> 8) | (o2 & MASK_C2);
    return o3;
}




uint64_t getSegment4bits(const uint64_t segment) {
    static constexpr uint64_t MASK_A1 = 0b011000'011000'011000'011000'011000'011000'011000'011000'011000;
    static constexpr uint64_t MASK_B1 = MASK_A1 >> 3;
    static constexpr uint64_t MASK_A2 = 0b001111'000000'000000'001111'000000'000000'001111'000000'000000;
    static constexpr uint64_t MASK_B2 = MASK_A2 >> 6;
    static constexpr uint64_t MASK_C2 = MASK_A2 >> 12;
    static constexpr uint64_t MASK_A3 = 0b000000'111111'111111'000000'000000'000000'000000'000000'000000;
    static constexpr uint64_t MASK_B3 = MASK_A3 >> 18;
    static constexpr uint64_t MASK_C3 = MASK_A3 >> 36;

    const uint64_t o1 = (segment & MASK_A1) >> 1 | (segment & MASK_B1);
    const uint64_t o2 = (o1 & MASK_A2) >> 4 | (o1 & MASK_B2) >> 2 | (o1 & MASK_C2);
    const uint64_t o3 = (o2 & MASK_A3) >> 12 | (o2 & MASK_B3) >> 6 | (o2 & MASK_C3);
    return o3;
}




void Board::precomputeHash2() {
    c_u64 above = getSegment2bits(b1);
    c_u64 below = getSegment2bits(b2);
    hash = above << 18 | below;
}


void Board::precomputeHash3() {
    c_u64 above = getSegment3bits(b1);
    c_u64 below = getSegment3bits(b2);
    hash = above << 30 | below;
}

void Board::precomputeHash4() {
    hash = prime_func1(b2, b1);
}

Board::HasherPtr Board::getHashFunc() const {
    c_u64 colorCount = getColorCount();
    if (hasFat() || colorCount > 3) {
        return &Board::precomputeHash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return &Board::precomputeHash2;
    }
    return &Board::precomputeHash3;
}



void Board::appendBoardToString(std::string& str, const Board* board, c_i32 curY) {
    c_bool isFat = board->hasFat();
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
        c_u8 value = board_b >> (51 - x - (curY % 3) * 18) & 0b111;
        if (isFat) {
            c_u32 curX = x / 3;
            if (curFatX == curX || curFatX == curX - 1) {
                if (curFatY == curY || curFatY == curY - 1) {
                    str.append(Colors::getBgColor(value));
                    inMiddle = curFatX == curX;
                }
            }
        }

        str.append(Colors::getColor(value));
        str.append(std::to_string(value));
        if (inMiddle) {
            if (x != 15) {
                str.append(" ");
            }
            str.append(Colors::bgReset);
        } else {
            str.append(Colors::bgReset);
            if (x != 15) {
                str.append(" ");
            }
        }
    }
}


std::string Board::toString() const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i);
        str.append("\n");
    }
    return str;
}


MUND std::string Board::toString(const Board& other) const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i);
        str.append("   ");
        appendBoardToString(str, &other, i);
        str.append("\n");
    }
    return str;
}


MUND std::string Board::toString(const Board* other) const {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i);
        str.append("   ");
        appendBoardToString(str, other, i);
        str.append("\n");
    }
    return str;
}


u64 Board::getScore2(MU const Board& other) {
    /*
    u64 ROW = 0;
    auto *uncoveredRows = reinterpret_cast<u8 *>(&ROW);
    u64 COL = 0;
    auto *uncoveredCols = reinterpret_cast<u8 *>(&COL);


    // Find all differing cells and update the counts in rows and cols
    // for C++, I can instantly sum the rows, but IDK about the columns
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (this.myBoard[i][j] != theOther.myBoard[i][j]) {
                uncoveredRows[i]++;
                uncoveredCols[j]++;
                differingCells++;
            }
        }
    }


    // While there are still uncovered differing cells
    // worst case this should only occur 6 times?
    int lanes = 0;
    int maxCover = 0;
    bool isRow = false;
    int index = -1;
    while (differingCells > 0) {
        maxCover = 0;
        isRow = false;
        index = -1;
        // Find the row or column that covers the most uncovered differing cells
        // in C++, can probably reinterpret the bytes to see if either can be skipped,
        // base which level of checking I am doing off of getScore1?
        // can be recoded to find the index and value of the max in both?
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (uncoveredRows[i] > maxCover) {
                maxCover = uncoveredRows[i];
                isRow = true;
                index = i;
            }
            if (uncoveredCols[i] > maxCover) {
                maxCover = uncoveredCols[i];
                isRow = false;
                index = i;
            }
        }

        if (index == -1) {
            break;
        }

        // Cover the chosen row or column and update the counts in
        // uncoveredRows and uncoveredColumns
        // I could cache the results of getScore1 for this lol
        if (isRow) {
            differingCells -= uncoveredRows[index];
            uncoveredRows[index] = 0;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (this.myBoard[index][j] != theOther.myBoard[index][j] && uncoveredCols[j] > 0) {
                    uncoveredCols[j]--;
                }
            }
        } else {
            differingCells -= uncoveredCols[index];
            uncoveredCols[index] = 0;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (this.myBoard[j][index] != theOther.myBoard[j][index] && uncoveredRows[j] > 0) {
                    uncoveredRows[j]--;
                }
            }
        }

        lanes++;
    }
    return lanes;
    */
    return 0;

}