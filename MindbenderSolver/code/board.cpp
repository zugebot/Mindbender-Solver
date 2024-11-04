#include "board.hpp"


#include <string>

#include "MindbenderSolver/utils/colors.hpp"
#include "segments.hpp"


int GET_SCORE_3_CALLS = 0;


Board::ColorArray_t Board::ColorsDefault = {0, 1, 2, 3, 4, 5, 6, 7};


Board::Board(C u8 values[36]) {
    setState(values);
}


Board::Board(C u8 values[36], C u8 x, C u8 y) {
    setState(values);
    setFatXY(x, y);
}


void B1B2::setState(C u8 values[36]) {
    std::array<i8, 8> colors = {8, 8, 8, 8, 8, 8, 8, 8};
    for (int i = 0; i < 36; i++) {
        C int val = values[i] & 0'7;
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


MU Board::ColorArray_t B1B2::setStateAndRetColors(C u8 values[36]) {
    ColorArray_t colors = {8, 8, 8, 8, 8, 8, 8, 8};
    ColorArray_t trueColors = {8, 8, 8, 8, 8, 8, 8, 8};

    for (int i = 0; i < 36; i++) {
        C int val = values[i] & 0'7;
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



u32 B1B2::getColorCount() C {
    C u64 colorCount = b1 >> 56 & 0xF;
    return colorCount;
}


static constexpr u64 MASK_FAT_POS = 0x1FFF'FFFF'FFFF'FFFF;

/**
 *
 * @param x value 0-4
 * @param y value 0-4
 */
void B1B2::setFatXY(C u64 x, C u64 y) {
    b1 = b1 & MASK_FAT_POS | x << 61;
    b2 = b2 & MASK_FAT_POS | y << 61;
    setFatBool(true);
}


MU void B1B2::setFatBool(C bool flag) {
    static constexpr u64 MASK_FAT_FLAG = 0xEFFF'FFFF'FFFF'FFFF;
    b1 = b1 & MASK_FAT_FLAG | static_cast<u64>(flag) << 60;

}

MU void B1B2::setFatX(C u64 x) {
    b1 = b1 & MASK_FAT_POS | x << 61;
}


MU void B1B2::setFatY(C u64 y) {
    b2 = b2 & MASK_FAT_POS | y << 61;
}


MU void B1B2::addFatX(C u8 x) {
    u64 cur_x = getFatX() + x;
    cur_x -= 6 * (cur_x > 5);
    b1 = b1 & MASK_FAT_POS | cur_x << 61;
}


MU void B1B2::addFatY(C u8 y) {
    u64 cur_y = getFatY() + y;
    cur_y -= 6 * (cur_y > 5);
    b2 = b2 & MASK_FAT_POS | cur_y << 61;
}


u8 B1B2::getFatX() C {
    return (b1 & ~MASK_FAT_POS) >> 61;
}


u8 B1B2::getFatY() C {
    return (b2 & ~MASK_FAT_POS) >> 61;
}


/// always returns a value between 0-24.
u8 B1B2::getFatXY() C {
    return (b1 >> 61) * 5 + (b2 >> 61);
}

MU u8 B1B2::getFatXYFast() C {
    return ((b1 >> 61) << 3) + (b2 >> 61);
}


bool B1B2::getFatBool() C {
    C bool state = (b1 >> 60 & 1) != 0;
    return state;
}

MU u8 B1B2::getColor(C u8 x, C u8 y) C {
    C i32 shift_amount = 51 - x * 3 - y % 3 * 18;
    return *(&b1 + (y >= 3)) >> shift_amount & 0'7;
}




/**
 * int x = (action1 % 30) / 5;
 * int y = (action2 % 30) / 5;
 * int m = 1 + action1 % 5;
 * int n = 1 + action2 % 5;
 */
MU bool Board::doActISColMatch(C u8 x1, C u8 y1, C u8 m, C u8 n) C {
    C int y2 = (y1 - n + 6) % 6;
    C int x2 = (x1 - m + 6) % 6;

    C u8 x1_3 = x1 * 3;
    C int offset_shared = 51 - (y1 % 3) * 18;
    C int shift_amount1 = x1_3 + offset_shared;
    C int shift_amount3 = x2 * 3 + offset_shared;

    C u64 base = y1 < 3 ? b1 : b2;

    C u8 color1 = base >> shift_amount1;
    C u8 color3 = base >> shift_amount3;

    if ((color1 ^ color3) & 0'7) {
        return false;
    }
    C int shift_amount2 = 51 - x1_3 - y2 % 3 * 18;
    C u64 base2 = y2 < 3 ? b1 : b2;
    C u8 color2 = base2 >> shift_amount2;

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
u8 Board::doActISColMatchBatched(C u8 x1, C u8 y1, C u8 m) C {
    C i32 x2 = (x1 - m + 6) % 6;
    C u64 base = y1 < 3 ? b1 : b2;
    C i32 offset_shared = 51 - y1 % 3 * 18;
    C u8 color1 = base >> (x1 * 3 + offset_shared);
    C u8 color3 = base >> (x2 * 3 + offset_shared);

    if ((color1 ^ color3) & 0'7) { return 0; }

    u8 results = 0;
    C i32 offset_shared2 = 51 - x1 * 3;
    for (i32 i = -5; i < 1; i++) {
        C i32 y2 = (y1 - i) % 6;
        C i32 y3 = ((y1 - i) % 3) * 18;
        C u64 base2 = y2 < 3 ? b1 : b2;
        C u8 color2 = base2 >> (offset_shared2 - y3);
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
inline u64 getSimilar54(C u64& sect1, C u64& sect2) {
    C u64 s = sect1 ^ sect2;
    return ~(s | s >> 1 | s >> 2) & 0'111111'111111'111111;
}





MU u64 B1B2::getScore1(C B1B2 &other) C {
    return __builtin_popcountll(getSimilar54(b1, other.b1))
           + __builtin_popcountll(getSimilar54(b2, other.b2));
}




inline u64 getAntiSimilar54(C u64& sect1, C u64& sect2) {
    C u64 s = sect1 ^ sect2;
    return (s | s >> 1 | s >> 2) & 0'111111'111111'111111;
}


MUND int B1B2::getScore3(C B1B2 theOther) C {
    ++GET_SCORE_3_CALLS;

    // Find all differing cells and update the counts in uncoveredRows and uncoveredCols
    static constexpr u64 PENIS_MASK = 0'111'111'111'111'111'111;
    C u64 full = _pext_u64(getAntiSimilar54(b1, theOther.b1), PENIS_MASK) << 18
                 | _pext_u64(getAntiSimilar54(b2, theOther.b2), PENIS_MASK);

    u8 differingCells   = __builtin_popcountll(full);

    alignas(u64) u8 uncRows[8] = {0};
    uncRows[0] = __builtin_popcountll(full & 0'770000000000);
    uncRows[1] = __builtin_popcountll(full & 0'007700000000);
    uncRows[2] = __builtin_popcountll(full & 0'000077000000);
    uncRows[3] = __builtin_popcountll(full & 0'000000770000);
    uncRows[4] = __builtin_popcountll(full & 0'000000007700);
    uncRows[5] = __builtin_popcountll(full & 0'000000000077);

    alignas(u64) u8 uncCols[8] = {0};
    uncCols[0] = __builtin_popcountll(full & 0'404040404040);
    uncCols[1] = __builtin_popcountll(full & 0'202020202020);
    uncCols[2] = __builtin_popcountll(full & 0'101010101010);
    uncCols[3] = __builtin_popcountll(full & 0'040404040404);
    uncCols[4] = __builtin_popcountll(full & 0'020202020202);
    uncCols[5] = __builtin_popcountll(full & 0'010101010101);


    u8 lanes = 0;
    // While there are still uncovered differing cells (at most 6 loops)
    while (differingCells > 0) {



        // Find the row or column that covers the most uncovered differing cells
        // in C++, can probably reinterpret the bytes to see if either can be skipped,
        // base which level of checking I am doing off of getScore1?
        // can be recoded to find the index and value of the max in both?

        u64 promoteRowMask = 1 << uncRows[0] | 1 << uncRows[1]
                             | 1 << uncRows[2] | 1 << uncRows[3]
                             | 1 << uncRows[4] | 1 << uncRows[5];
        u8 highestRow = 31 - __builtin_clz(promoteRowMask);

        u64 promoteColMask = 1 << uncCols[0] | 1 << uncCols[1]
                             | 1 << uncCols[2] | 1 << uncCols[3]
                             | 1 << uncCols[4] | 1 << uncCols[5];
        u8 highestCol = 31 - __builtin_clz(promoteColMask);


        if (highestRow == 0 && highestCol == 0) {
            break;
        }

        int index;
        bool isRow = highestRow >= highestCol;
        if (isRow) {
            index = (uncRows[0] == highestRow) ? 0 :
                    (uncRows[1] == highestRow) ? 1 :
                    (uncRows[2] == highestRow) ? 2 :
                    (uncRows[3] == highestRow) ? 3 :
                    (uncRows[4] == highestRow) ? 4 : 5;
        } else {
            index = (uncCols[0] == highestCol) ? 0 :
                    (uncCols[1] == highestCol) ? 1 :
                    (uncCols[2] == highestCol) ? 2 :
                    (uncCols[3] == highestCol) ? 3 :
                    (uncCols[4] == highestCol) ? 4 : 5;
        }

        // Cover the chosen row or column and update the counts in
        // uncoveredRows and uncoveredColumns
        if (isRow) {
            differingCells -= uncRows[index];
            uncRows[index] = 0;
            for (int j = 0; j < 6; j++) {
                if (getColor(j, index) != theOther.getColor(j, index) && uncCols[j] > 0) {
                    uncCols[j]--;
                }
            }
        } else {
            differingCells -= uncCols[index];
            uncCols[index] = 0;
            for (int j = 0; j < 6; j++) {
                if (getColor(index, j) != theOther.getColor(index, j) && uncRows[j] > 0) {
                    uncRows[j]--;
                }
            }
        }

        lanes++;
    }

    return lanes;
}







template<int MAX_DEPTH>
bool B1B2::getScore3Till(C B1B2 theOther) C {
    ++GET_SCORE_3_CALLS;

    // Find all differing cells and update the counts in uncoveredRows and uncoveredCols
    static constexpr u64 PENIS_MASK = 0'111'111'111'111'111'111;
    C u64 full = _pext_u64(getAntiSimilar54(b1, theOther.b1), PENIS_MASK) << 18
                 | _pext_u64(getAntiSimilar54(b2, theOther.b2), PENIS_MASK);

    u8 diffCells = __builtin_popcountll(full);

    alignas(u64) u8 uncRows[8] = {0};
    uncRows[0] = __builtin_popcountll(full & 0'770000000000);
    uncRows[1] = __builtin_popcountll(full & 0'007700000000);
    uncRows[2] = __builtin_popcountll(full & 0'000077000000);
    uncRows[3] = __builtin_popcountll(full & 0'000000770000);
    uncRows[4] = __builtin_popcountll(full & 0'000000007700);
    uncRows[5] = __builtin_popcountll(full & 0'000000000077);

    alignas(u64) u8 uncCols[8] = {0};
    uncCols[0] = __builtin_popcountll(full & 0'404040404040);
    uncCols[1] = __builtin_popcountll(full & 0'202020202020);
    uncCols[2] = __builtin_popcountll(full & 0'101010101010);
    uncCols[3] = __builtin_popcountll(full & 0'040404040404);
    uncCols[4] = __builtin_popcountll(full & 0'020202020202);
    uncCols[5] = __builtin_popcountll(full & 0'010101010101);



    // While there are still uncovered differing cells (at most 6 loops)
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        ++depth;

        u64 promoteRowMask = 1 << uncRows[0] | 1 << uncRows[1]
                             | 1 << uncRows[2] | 1 << uncRows[3]
                             | 1 << uncRows[4] | 1 << uncRows[5];
        u8 highestRow = 31 - __builtin_clz(promoteRowMask);

        u64 promoteColMask = 1 << uncCols[0] | 1 << uncCols[1]
                             | 1 << uncCols[2] | 1 << uncCols[3]
                             | 1 << uncCols[4] | 1 << uncCols[5];
        u8 highestCol = 31 - __builtin_clz(promoteColMask);


        if (highestRow == 0 && highestCol == 0) {
            break;
        }

        int index;
        bool isRow = highestRow >= highestCol;
        if (isRow) {
            index = (uncRows[0] == highestRow) ? 0 :
                    (uncRows[1] == highestRow) ? 1 :
                    (uncRows[2] == highestRow) ? 2 :
                    (uncRows[3] == highestRow) ? 3 :
                    (uncRows[4] == highestRow) ? 4 : 5;
        } else {
            index = (uncCols[0] == highestCol) ? 0 :
                    (uncCols[1] == highestCol) ? 1 :
                    (uncCols[2] == highestCol) ? 2 :
                    (uncCols[3] == highestCol) ? 3 :
                    (uncCols[4] == highestCol) ? 4 : 5;
        }

        // Cover the chosen row or column and update the counts in
        // uncoveredRows and uncoveredColumns
        if (isRow) {
            diffCells -= uncRows[index];
            uncRows[index] = 0;
            for (int j = 0; j < 6; j++) {
                if (getColor(j, index) != theOther.getColor(j, index) && uncCols[j] > 0) {
                    uncCols[j]--;
                }
            }
        } else {
            diffCells -= uncCols[index];
            uncCols[index] = 0;
            for (int j = 0; j < 6; j++) {
                if (getColor(index, j) != theOther.getColor(index, j) && uncRows[j] > 0) {
                    uncRows[j]--;
                }
            }
        }

        if (diffCells == 0) {
            break;
        }
    }



    return diffCells != 0;
}



template bool B1B2::getScore3Till<1>(C B1B2 theOther) C;
template bool B1B2::getScore3Till<2>(C B1B2 theOther) C;
template bool B1B2::getScore3Till<3>(C B1B2 theOther) C;
template bool B1B2::getScore3Till<4>(C B1B2 theOther) C;
template bool B1B2::getScore3Till<5>(C B1B2 theOther) C;











MU u64 Board::getRowColIntersections(C u32 x, C u32 y) C {
    static constexpr u64 C_MAIN_MASK = 0'000007'000007'000007;
    static constexpr u32 C_CNTR_MASKS[8] = {
            0x00000000, 0x02108421, 0x04210842, 0x06318C63,
            0x08421084, 0x0A5294A5, 0x0C6318C6, 0x0E739CE7};
    C u32 left = 15 - x * 3;
    C u32 row = *(&b1 + (y >= 3)) >> (2 - y - 3 * (y >= 3)) * 18 & 0'777777;
    C u32 cntr_p1_r = row >> left & 0'7;

    // find col_x5
    C u64 col_mask = C_MAIN_MASK << left;
    C u64 b1_c = (b1 & col_mask) >> left;
    C u64 b2_c = (b2 & col_mask) >> left;
    C u32 shifted_5 = (b2_c | b2_c >> 13 | b2_c >> 26) & 0x1CE7 |
                      (b1_c << 15 | b1_c << 2 | b1_c >> 11) & 0xE738000;
    C u32 s = shifted_5 ^ C_CNTR_MASKS[cntr_p1_r];
    C u32 sim = ((~(s | s >> 1 | s >> 2)) & C_CNTR_MASKS[1]) * 31;
    C u32 col_x5 = (sim & (0x3FFFFFFF << (5 * (6 - y)))) >> 5
                   | sim & (0x1FFFFFF >> 5 * y);

    // find row_x5
    C u32 s_ps = row ^ (cntr_p1_r * 0'111111);

    // TODO: could this use _pext_u64?
    C u32 sim_r = ~(s_ps | s_ps >> 1 | s_ps >> 2) & 0'111111;
    C u32 p1_r = (sim_r & 0'101010) >> 2 | sim_r & 0'10101;
    C u32 row_t1 = (p1_r >> 8 | p1_r >> 4 | p1_r) & 0'77;

    C u32 row_x5 = ((row_t1 & (0'7700 >> x)) >> 1 | row_t1 & (0'37 >> x)) * 0x108421;

    return col_x5 & row_x5;
}


void Board::precomputeHash2() {
    C u64 above = getSegment2bits(b1);
    C u64 below = getSegment2bits(b2);
    memory.setHash(above << 18 | below);
}


void Board::precomputeHash3() {
    C u64 above = getSegment3bits(b1);
    C u64 below = getSegment3bits(b2);
    memory.setHash(above << 30 | below);
}


void Board::precomputeHash4() {
    memory.setHash(prime_func1(b2, b1));
}


MU Board::HasherPtr Board::getHashFunc() C {
    C u64 colorCount = getColorCount();
    if (getFatBool() || colorCount > 3) {
        return &Board::precomputeHash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return &Board::precomputeHash2;
    }
    return &Board::precomputeHash3;
}


void Board::appendBoardToString(std::string &str, C Board *board, C i32 curY, C PrintSettings theSettings) {
    C bool isFat = board->getFatBool();
    C u8 curFatX = board->getFatX();
    C u8 curFatY = board->getFatY();
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
        C u8 value = theSettings.trueColors[board_b >> (51 - x - (curY % 3) * 18) & 0'7];
        if (isFat) {
            C u32 curX = x / 3;
            if (curFatX == curX || curFatX == curX - 1) {
                if (curFatY == curY || curFatY == curY - 1) {
                    if (theSettings.useAscii)
                        str.append(Colors::getBgColor(value));
                    inMiddle = curFatX == curX;
                }
            }
        }
        if (theSettings.useAscii)
            str.append(Colors::getColor(value));
        str.append(std::to_string(value));
        if (inMiddle) {
            if (x != 15) { str.append(" "); }
            if (theSettings.useAscii)
                str.append(Colors::bgReset);
        } else {
            if (theSettings.useAscii)
                str.append(Colors::bgReset);
            if (x != 15) { str.append(" "); }
        }
    }
}


MUND std::string Board::toBlandString() C {
    return toStringSingle(PrintSettings());
}


MUND std::string Board::toString(C Board& other, C PrintSettings theSettings) C {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("   ");
        appendBoardToString(str, &other, i, theSettings);
        str.append("\n");
    }
    return str;
}


MUND std::string Board::toString(C Board* other, C PrintSettings theSettings) C {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("   ");
        appendBoardToString(str, other, i, theSettings);
        str.append("\n");
    }
    return str;
}


std::string Board::toStringSingle(C PrintSettings theSettings) C {
    std::string str;
    for (int i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("\n");
    }
    return str;
}