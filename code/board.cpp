// code/board.cpp
#include "board.hpp"


#include <sstream>
#include <vector>
#include <string>

#include "utils/processor.hpp"
#include "utils/colors.hpp"
#include "utils/intrinsics/clz.hpp"
#include "utils/intrinsics/pext_u64.hpp"
#include "utils/intrinsics/popcount.hpp"
#include "rotations.hpp"




namespace {
    static void setAllColors(B1B2 *b1b2, C u8 adjustedValues[36]) {
        b1b2->b1 = 0;
        for (i32 i = 0; i < 18; i++) {
            b1b2->b1 = (b1b2->b1 << 3) | (adjustedValues[i] & 0'7);
        }

        b1b2->b2 = 0;
        for (i32 i = 18; i < 36; i++) {
            b1b2->b2 = (b1b2->b2 << 3) | (adjustedValues[i] & 0'7);
        }
    }

    constexpr u64 MAKE_MASK(C u64 offset, C u64 bits) {
        return ~(((1ULL << bits) - 1ULL) << offset);
    }

    template<typename T1, typename T2>
    FORCEINLINE u64 getShiftAmount(C T1 x, C T2 y) {
        static constexpr u64 MAGIC = 0x33210F33210F;
        return (MAGIC >> (y * 8)) - (x * 3);
    }

    FORCEINLINE HD u64 getSimilar54(C u64 &sect1, C u64 &sect2) {
        C u64 s = sect1 ^ sect2;
        return ~(s | s >> 1 | s >> 2) & 0'111111'111111'111111;
    }

    FORCEINLINE HD u64 getAntiSimilar54(C u64 &sect1, C u64 &sect2) {
        C u64 s = sect1 ^ sect2;
        return (s | s >> 1 | s >> 2) & 0'111111'111111'111111;
    }
} // misc


Board::ColorArray_t Board::ColorsDefault = {0, 1, 2, 3, 4, 5, 6, 7};


Board::Board(C u8 values[36], C u8 x, C u8 y) {
    setState(values);
    setFatXY(x, y);
}

void B1B2::setState(C u8 values[36]) {
    ColorArray_t colors = {8, 8, 8, 8, 8, 8, 8, 8};

    for (i32 i = 0; i < 36; i++) {
        C i32 val = values[i] & 0'7;
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
    for (i32 i = 0; i < 36; i++) {
        adjusted_values[i] = colors[values[i]];
    }

    setAllColors(this, adjusted_values);
    setColorCount(colorCount);
}

MU B1B2::ColorArray_t B1B2::setStateAndRetColors(C u8 values[36]) {
    ColorArray_t colors = {8, 8, 8, 8, 8, 8, 8, 8};
    ColorArray_t trueColors = {8, 8, 8, 8, 8, 8, 8, 8};

    for (i32 i = 0; i < 36; i++) {
        C i32 val = values[i] & 0'7;
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
    for (i32 i = 0; i < 36; i++) {
        adjusted_values[i] = colors[values[i]];
    }

    setAllColors(this, adjusted_values);
    setColorCount(colorCount);
    return trueColors;
}





// ############################################################
// #                        Board                            #
// ############################################################


// ============================================================
// Packed-state metadata
// ============================================================

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

MU HD void B1B2::setColorCount(C u64 colorCount) {
    b1 = (b1 & COLOR_COUNT_MASK) | ((colorCount - 1) << COLOR_COUNT_OFFSET);
}

u32 HD B1B2::getColorCount() C {
    C u64 colorCount = (b1 & ~COLOR_COUNT_MASK) >> COLOR_COUNT_OFFSET;
    return colorCount + 1;
}

MU HD void B1B2::setFatBool(C bool flag) {
    b1 = (b1 & FAT_BOOL_MASK) | (static_cast<u64>(flag) << FAT_BOOL_OFFSET);
}

bool HD B1B2::getFatBool() C {
    return (b1 >> FAT_BOOL_OFFSET) & 1;
}

MU HD void B1B2::setFatX(C u64 x) {
    b1 = (b1 & FAT_X_MASK) | (x << FAT_X_OFFSET);
}

MU HD void B1B2::setFatY(C u64 y) {
    b1 = (b1 & FAT_Y_MASK) | (y << FAT_Y_OFFSET);
}

u8 HD B1B2::getFatX() C {
    return (b1 & ~FAT_X_MASK) >> FAT_X_OFFSET;
}

u8 HD B1B2::getFatY() C {
    return (b1 & ~FAT_Y_MASK) >> FAT_Y_OFFSET;
}

static constexpr u64 ADD_FAT_MAGIC = 0x8D116344;

MU HD void B1B2::addFatX(C u64 x) {
    b1 = (b1 & FAT_X_MASK)
         | ((((ADD_FAT_MAGIC >> (3 * (getFatX() + x) - 1)) & 0b111)) << FAT_X_OFFSET);
}

MU HD void B1B2::addFatY(C u64 y) {
    b1 = (b1 & FAT_Y_MASK)
         | ((((ADD_FAT_MAGIC >> (3 * (getFatY() + y) - 1)) & 0b111)) << FAT_Y_OFFSET);
}

HD void B1B2::setFatXY(C u64 x, C u64 y) {
    setFatX(x);
    setFatY(y);
    setFatBool(true);
}

MU u8 HD B1B2::getColor(C u8 x, C u8 y) C {
    C u64 shift_amount = getShiftAmount<u8, u8>(x, y);
    return (*(&b1 + (y >= 3)) >> shift_amount) & 0'7;
}

// ============================================================
// Board-only helpers
// ============================================================

MU HD bool Board::doActISColMatch(C u8 x1, C u8 y1, C u8 m, C u8 n) C {
    C i32 y2 = (y1 - n + 6) % 6;
    C i32 x2 = (x1 - m + 6) % 6;

    C i32 offset_shared = 51 - (y1 % 3) * 18;
    C i32 shift_amount1 = x1 * 3 + offset_shared;
    C i32 shift_amount3 = x2 * 3 + offset_shared;

    C u64 base = y1 < 3 ? b1 : b2;

    C u8 color1 = base >> shift_amount1;
    C u8 color3 = base >> shift_amount3;

    if ((color1 ^ color3) & 0'7) {
        return false;
    }

    C u64 shift_amount2 = getShiftAmount<u8, i32>(x1, y2);
    C u64 base2 = y2 < 3 ? b1 : b2;
    C u8 color2 = base2 >> shift_amount2;

    return (color1 ^ color2) & 0'7;
}

u8 HD Board::doActISColMatchBatched(C u8 x1, C u8 y1, C u8 m) C {
    C i32 x2 = (x1 - m + 6) % 6;
    C u64 base = y1 < 3 ? b1 : b2;
    C i32 offset_shared = 51 - y1 % 3 * 18;
    C u8 color1 = base >> (x1 * 3 + offset_shared);
    C u8 color3 = base >> (x2 * 3 + offset_shared);

    if ((color1 ^ color3) & 0'7) {
        return 0;
    }

    u8 results = 0;
    C i32 offset_shared2 = 51 - x1 * 3;
    for (i32 i = -5; i < 1; i++) {
        C i32 y2 = (y1 - i) % 6;
        C i32 y3 = ((y1 - i) % 3) * 18;
        C u64 base2 = y2 < 3 ? b1 : b2;
        C u8 color2 = base2 >> (offset_shared2 - y3);
        results |= ((((color1 ^ color2) & 07) != 0) << (i + 5));
    }

    return results;
}

double HD Board::getDuplicateEstimateAtDepth(MU u32 depth) {
    (void)depth;
    return 1.0;
}

MU HD u64 B1B2::getScore1(C B1B2& other) C {
    return my_popcount(getSimilar54(b1, other.b1))
           + my_popcount(getSimilar54(b2, other.b2));
}



namespace {
    constexpr u64 ROW_DIFF_PEXT_MASK = 0'111111;
    
    FORCEINLINE HD u32 extractPackedRow18(C u64 half, C u8 rowInHalf) {
        return static_cast<u32>((half >> ((2u - rowInHalf) * 18u)) & 0x3FFFFu);
    }

    FORCEINLINE HD u8 buildPackedRowDiffMask(C u32 lhsRow, C u32 rhsRow) {
        C u64 s = static_cast<u64>(lhsRow ^ rhsRow);
        C u64 diffBits = (s | (s >> 1) | (s >> 2)) & ROW_DIFF_PEXT_MASK;
        return static_cast<u8>(my_pext_u64(diffBits, ROW_DIFF_PEXT_MASK));
    }

    FORCEINLINE HD void buildRowDifferenceMasksFast(C B1B2& lhs,
                                                    C B1B2& rhs,
                                                    u8 (&rowMasks)[6]) {
        C u32 lhsRow0 = extractPackedRow18(lhs.b1, 0);
        C u32 lhsRow1 = extractPackedRow18(lhs.b1, 1);
        C u32 lhsRow2 = extractPackedRow18(lhs.b1, 2);
        C u32 lhsRow3 = extractPackedRow18(lhs.b2, 0);
        C u32 lhsRow4 = extractPackedRow18(lhs.b2, 1);
        C u32 lhsRow5 = extractPackedRow18(lhs.b2, 2);

        C u32 rhsRow0 = extractPackedRow18(rhs.b1, 0);
        C u32 rhsRow1 = extractPackedRow18(rhs.b1, 1);
        C u32 rhsRow2 = extractPackedRow18(rhs.b1, 2);
        C u32 rhsRow3 = extractPackedRow18(rhs.b2, 0);
        C u32 rhsRow4 = extractPackedRow18(rhs.b2, 1);
        C u32 rhsRow5 = extractPackedRow18(rhs.b2, 2);

        rowMasks[0] = buildPackedRowDiffMask(lhsRow0, rhsRow0);
        rowMasks[1] = buildPackedRowDiffMask(lhsRow1, rhsRow1);
        rowMasks[2] = buildPackedRowDiffMask(lhsRow2, rhsRow2);
        rowMasks[3] = buildPackedRowDiffMask(lhsRow3, rhsRow3);
        rowMasks[4] = buildPackedRowDiffMask(lhsRow4, rhsRow4);
        rowMasks[5] = buildPackedRowDiffMask(lhsRow5, rhsRow5);
    }

    FORCEINLINE HD i32 popcount6(C u8 v) {
        return static_cast<i32>(my_popcount(static_cast<u32>(v)));
    }

    FORCEINLINE HD u8 computeUncoveredCols(C u8 rowsChosen, C u8 (&rowMasks)[6]) {
        u8 uncoveredCols = 0;

        if ((rowsChosen & (1u << 0)) == 0) { uncoveredCols |= rowMasks[0]; }
        if ((rowsChosen & (1u << 1)) == 0) { uncoveredCols |= rowMasks[1]; }
        if ((rowsChosen & (1u << 2)) == 0) { uncoveredCols |= rowMasks[2]; }
        if ((rowsChosen & (1u << 3)) == 0) { uncoveredCols |= rowMasks[3]; }
        if ((rowsChosen & (1u << 4)) == 0) { uncoveredCols |= rowMasks[4]; }
        if ((rowsChosen & (1u << 5)) == 0) { uncoveredCols |= rowMasks[5]; }

        return uncoveredCols;
    }

    FORCEINLINE HD i32 exactMinRowColCoverFromMasks(C u8 (&rowMasks)[6]) {
        i32 best = 6;

        for (u8 rowsChosen = 0; rowsChosen < 64; ++rowsChosen) {
            C i32 rowCost = popcount6(rowsChosen);
            if (rowCost >= best) {
                continue;
            }

            C u8 uncoveredCols = computeUncoveredCols(rowsChosen, rowMasks);
            C i32 coverCost = rowCost + popcount6(uncoveredCols);

            if (coverCost < best) {
                best = coverCost;
                if (best == 0) {
                    break;
                }
            }
        }

        return best;
    }

    template<i32 MAX_DEPTH>
    FORCEINLINE HD bool exactMinRowColCoverExceedsFromMasks(C u8 (&rowMasks)[6]) {
        for (u8 rowsChosen = 0; rowsChosen < 64; ++rowsChosen) {
            C i32 rowCost = popcount6(rowsChosen);
            if (rowCost > MAX_DEPTH) {
                continue;
            }

            C u8 uncoveredCols = computeUncoveredCols(rowsChosen, rowMasks);
            C i32 coverCost = rowCost + popcount6(uncoveredCols);

            if (coverCost <= MAX_DEPTH) {
                return false;
            }
        }

        return true;
    }
}

MUND HD i32 B1B2::getExactRowColLowerBound(C B1B2& other) C {
    u8 rowMasks[6];
    buildRowDifferenceMasksFast(*this, other, rowMasks);
    return exactMinRowColCoverFromMasks(rowMasks);
}

template<i32 MAX_DEPTH>
MUND HD bool B1B2::getExactRowColLowerBoundTill(C B1B2& other) C {
    u8 rowMasks[6];
    buildRowDifferenceMasksFast(*this, other, rowMasks);
    return exactMinRowColCoverExceedsFromMasks<MAX_DEPTH>(rowMasks);
}

template HD bool B1B2::getExactRowColLowerBoundTill<1>(C B1B2& other) C;
template HD bool B1B2::getExactRowColLowerBoundTill<2>(C B1B2& other) C;
template HD bool B1B2::getExactRowColLowerBoundTill<3>(C B1B2& other) C;
template HD bool B1B2::getExactRowColLowerBoundTill<4>(C B1B2& other) C;
template HD bool B1B2::getExactRowColLowerBoundTill<5>(C B1B2& other) C;










namespace {
    constexpr u64 DIFF_CELL_MASK = 0'111'111'111'111'111'111;
    
    FORCEINLINE HD u64 buildDiffFull(C B1B2& lhs, C B1B2& rhs) {
        return (my_pext_u64(getAntiSimilar54(lhs.b1, rhs.b1), DIFF_CELL_MASK) << 18)
               | my_pext_u64(getAntiSimilar54(lhs.b2, rhs.b2), DIFF_CELL_MASK);
    }

    FORCEINLINE HD void buildUncoveredCounts(C u64 full, u8 (&uncRows)[8], u8 (&uncCols)[8]) {
        uncRows[0] = my_popcount(full & 0'770000000000);
        uncRows[1] = my_popcount(full & 0'007700000000);
        uncRows[2] = my_popcount(full & 0'000077000000);
        uncRows[3] = my_popcount(full & 0'000000770000);
        uncRows[4] = my_popcount(full & 0'000000007700);
        uncRows[5] = my_popcount(full & 0'000000000077);

        uncCols[0] = my_popcount(full & 0'404040404040);
        uncCols[1] = my_popcount(full & 0'202020202020);
        uncCols[2] = my_popcount(full & 0'101010101010);
        uncCols[3] = my_popcount(full & 0'040404040404);
        uncCols[4] = my_popcount(full & 0'020202020202);
        uncCols[5] = my_popcount(full & 0'010101010101);
    }
}


#define countNonZeroBytes(x) \
(((((x)&~0ULL/255*127)+~0ULL/255*(127)|(x))&~0ULL/255*128)/128%255)

/**
 * This blindly wipes a single row or column to check if doing so makes the states equal.
 * It doesn't do real permutations, so it can have false positives.
 */
MUND HD bool B1B2::couldBeSolvedIn1Move(const B1B2 theOther) C {
    C u64 full = buildDiffFull(*this, theOther);

    if EXPECT_FALSE(my_popcount(full) == 0) {
        return true;
    }

    alignas(u64) u8 uncRows[8] = {};
    alignas(u64) u8 uncCols[8] = {};
    buildUncoveredCounts(full, uncRows, uncCols);

    C u32 countRow = countNonZeroBytes(*reinterpret_cast<u64*>(uncRows));
    C u32 countCol = countNonZeroBytes(*reinterpret_cast<u64*>(uncCols));
    C u32 minimum = std::min(countRow, countCol);
    return minimum < 2;
}
#undef countNonZeroBytes

MU __host__ void B1B2::doMoves(C std::initializer_list<Action> theInitList) {
    for (Action func : theInitList) {
        func(*this);
    }
}

MU HD u64 Board::getRowColIntersections(C u32 x, C u32 y) C {
    static constexpr u64 C_MAIN_MASK = 0'000007'000007'000007;
    static constexpr u32 C_CNTR_MASKS[8] = {
            0x00000000, 0x02108421, 0x04210842, 0x06318C63,
            0x08421084, 0x0A5294A5, 0x0C6318C6, 0x0E739CE7
    };

    C u32 left = 15 - x * 3;
    C u32 row = *(&b1 + (y >= 3)) >> ((2 - y - 3 * (y >= 3)) * 18) & 0'777777;
    C u32 cntr_p1_r = row >> left & 0'7;

    C u64 col_mask = C_MAIN_MASK << left;
    C u64 b1_c = (b1 & col_mask) >> left;
    C u64 b2_c = (b2 & col_mask) >> left;
    C u32 shifted_5 = (b2_c | b2_c >> 13 | b2_c >> 26) & 0x1CE7
                      | (b1_c << 15 | b1_c << 2 | b1_c >> 11) & 0xE738000;

    C u32 s = shifted_5 ^ C_CNTR_MASKS[cntr_p1_r];
    C u32 sim = ((~(s | s >> 1 | s >> 2)) & C_CNTR_MASKS[1]) * 31;
    C u32 col_x5 = (sim & (0x3FFFFFFF << (5 * (6 - y)))) >> 5
                   | sim & (0x1FFFFFF >> (5 * y));

    C u32 s_ps = row ^ (cntr_p1_r * 0'111111);
    C u32 sim_r = ~(s_ps | s_ps >> 1 | s_ps >> 2) & 0'111111;
    C u32 p1_r = ((sim_r & 0'101010) >> 2) | (sim_r & 0'10101);
    C u32 row_t1 = (p1_r >> 8 | p1_r >> 4 | p1_r) & 0'77;

    C u32 row_x5 = (((row_t1 & (0'7700 >> x)) >> 1) | (row_t1 & (0'37 >> x))) * 0x108421;

    return col_x5 & row_x5;
}

MUND HD u32 Board::getRowCC() C {
    if EXPECT_FALSE(memory.getMoveCount() == 0) {
        return 30;
    }

    C u8 lastMove = memory.getLastMove();
    if (lastMove < 32) {
        return 25 - ((lastMove & 0b11111) / 5) * 5;
    }
    return 30;
}

MUND HD u32 Board::getColCC() C {
    if EXPECT_FALSE(memory.getMoveCount() == 0) {
        return 30;
    }

    C u8 lastMove = memory.getLastMove();
    if (lastMove < 32) {
        return 30;
    }
    return 25 - ((lastMove & 0b11111) / 5) * 5;
}

namespace my_cuda {
    MU __constant__ u8 ROW_COL_OFFSETS[30] = {
            25, 25, 25, 25, 25,
            20, 20, 20, 20, 20,
            15, 15, 15, 15, 15,
            10, 10, 10, 10, 10,
            5, 5, 5, 5, 5,
            0, 0, 0, 0, 0
    };
}

MU __device__ void Board::setRowColCC(u32* ptr) C {
    C u8 lastMove = memory.getLastMove();
    if (lastMove < 32) {
        ptr[0] = my_cuda::ROW_COL_OFFSETS[lastMove & 0b11111];
        ptr[1] = 30;
    } else {
        ptr[0] = 30;
        ptr[1] = my_cuda::ROW_COL_OFFSETS[lastMove & 0b11111];
    }
}

void Board::appendBoardToString(std::string& str, C Board* board, C i32 curY, C PrintSettings theSettings) {
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

    for (i32 x = 0; x < 18; x += 3) {
        C u8 value = theSettings.trueColors[(board_b >> (51 - x - (curY % 3) * 18)) & 0'7];

        if (isFat) {
            C u32 curX = x / 3;
            if (curFatX == curX || curFatX == curX - 1) {
                if (curFatY == curY || curFatY == curY - 1) {
                    if (theSettings.useAscii) {
                        str.append(Colors::getBgColor(value));
                    }
                    inMiddle = curFatX == curX;
                }
            }
        }

        if (theSettings.useAscii) {
            str.append(Colors::getColor(value));
        }

        str.append(std::to_string(value));

        if (inMiddle) {
            if (x != 15) {
                str.append(" ");
            }
            if (theSettings.useAscii) {
                str.append(Colors::bgReset);
            }
        } else {
            if (theSettings.useAscii) {
                str.append(Colors::bgReset);
            }
            if (x != 15) {
                str.append(" ");
            }
        }
    }
}

MUND std::string Board::toBlandString() C {
    return toStringSingle(PrintSettings());
}

MUND std::string Board::toString(C Board& other, C PrintSettings theSettings) C {
    std::string str;
    for (i32 i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("   ");
        appendBoardToString(str, &other, i, theSettings);
        str.append("\n");
    }
    return str;
}

MUND std::string Board::toString(C Board* other, C PrintSettings theSettings) C {
    std::string str;
    for (i32 i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("   ");
        appendBoardToString(str, other, i, theSettings);
        str.append("\n");
    }
    return str;
}

std::string Board::toStringSingle(C PrintSettings theSettings) C {
    std::string str;
    for (i32 i = 0; i < 6; i++) {
        appendBoardToString(str, this, i, theSettings);
        str.append("\n");
    }
    return str;
}



// ############################################################
// #                        MEMORY                            #
// ############################################################


MU HD void Memory::setNextMoves(C std::initializer_list<u64> moveValues) {
    for (C auto& moveValue : moveValues) {
        setNextNMove<1>(moveValue);
    }
}

// ############################################################
// #            To String -Similar- Functions                 #
// ############################################################

namespace {
    std::string removeTrailingSpace(std::string& str) {
        if (!str.empty() && str.back() == ' ') {
            str.pop_back();
        }
        return str;
    }
}

std::string Memory::asmString(C Memory* other) C {
    std::string start = asmStringForwards();
    std::string end = other->asmStringBackwards();
    return start.empty() ? end : end.empty() ? start : start + " " + end;
}

std::string Memory::formatMoveString(C u8 move, C bool isForwards) {
    char temp[5] = {};
    if (isForwards) {
        memcpy(temp, allActStructList[move].name.data(), 4);
    } else {
        C u32 index = move + allActStructList[move].tillNext - 1 - allActStructList[move].tillLast;
        memcpy(temp, allActStructList[index].name.data(), 4);
    }
    return temp;
}

std::string Memory::asmStringForwards() C {
    C u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(3 * count);

    for (u32 i = 0; i < count; i++) {
        C u8 move = getMove(i);
        moves_str += formatMoveString(move, true) + " ";
    }

    removeTrailingSpace(moves_str);
    return moves_str;
}

std::string Memory::asmStringBackwards() C {
    C u32 count = getMoveCount();

    std::string moves_str;
    moves_str.reserve(3 * count);

    for (u32 i = count; i != 0; i--) {
        C u8 move = getMove(i - 1);
        moves_str += formatMoveString(move, false) + " ";
    }

    removeTrailingSpace(moves_str);
    return moves_str;
}

std::string Memory::asmFatString(C u8 fatPos, C Memory* other, C u8 fatPosOther) C {
    std::string start = asmFatStringForwards(fatPos);
    if (other == nullptr) {
        return start;
    }

    std::string end = other->asmFatStringBackwards(fatPosOther);
    if (start.empty()) {
        return end;
    }
    if (end.empty()) {
        return start;
    }
    return start + " " + end;
}

std::string Memory::asmFatStringForwards(C u8 fatPos) C {
    std::string moves_str;
    C u32 count = getMoveCount();
    i32 x = fatPos / 5;
    i32 y = fatPos % 5;

    for (u32 i = 0; i < count; i++) {
        char temp[5] = {};
        memcpy(temp, allActStructList[
                             fatActionsIndexes[x * 5 + y][getMove(i)]].name.data(), 4);

        C u32 back = 2 + (temp[3] != '\0');
        moves_str += temp;

        if (back == 3) { // if it is a fat move
            if (temp[0] == 'R') {
                if (temp[1] - '0' == y) {
                    x += temp[back] - '0';
                    x -= 6 * (x > 5);
                }
            } else if EXPECT_TRUE(temp[0] == 'C') {
                if (temp[1] - '0' == x) {
                    y += temp[back] - '0';
                    y -= 6 * (y > 5);
                }
            }
        }

        if (i != count - 1) {
            moves_str += " ";
        }
    }

    return moves_str;
}

std::string Memory::asmFatStringBackwards(C u8 fatPos) C {
    C u32 count = getMoveCount();

    std::vector<std::string> moves_vec;
    moves_vec.resize(count);

    i32 x = fatPos / 5;
    i32 y = fatPos % 5;

    for (u32 i = 0; i < count; i++) {
        char temp[5] = {};
        memcpy(temp, allActStructList[
                             fatActionsIndexes[x * 5 + y][getMove(i)]].name.data(), 4);

        C u32 back = 2 + (temp[3] != '\0');

        if (back == 3) { // if it is a fat move
            if (temp[0] == 'R') {
                if (temp[1] - '0' == y) {
                    x += temp[back] - '0';
                    x -= 6 * (x > 5);
                }
            } else if EXPECT_TRUE(temp[0] == 'C') {
                if (temp[1] - '0' == x) {
                    y += temp[back] - '0';
                    y -= 6 * (y > 5);
                }
            }
        }

        temp[back] = static_cast<char>('f' - temp[back]);
        moves_vec[i] = temp;
    }

    std::string moves_str;
    for (i32 i = static_cast<i32>(moves_vec.size()) - 1; i >= 0; i--) {
        moves_str.append(moves_vec[i]);
        if (i != 0) {
            moves_str += " ";
        }
    }

    return moves_str;
}

MU std::string Memory::toString() C {
    std::string str = "Move[";

    C i32 moveCount = getMoveCount();
    for (i32 i = 0; i < moveCount; i++) {
        str.append(std::to_string(getMove(i)));
        if (i != moveCount - 1) {
            str.append(", ");
        }
    }

    str.append("]");
    return str;
}

template<bool HAS_FAT>
static std::vector<u8> parseMoveStringTemplated(C std::string& input) {
    (void)HAS_FAT;

    std::vector<u8> result;
    std::istringstream iss(input);
    std::string seg;

    while (iss >> seg) {
        if (seg.length() == 3) {
            C u8 baseValue = seg[0] == 'R' ? 0 : 32;
            C u32 value = baseValue + (seg[1] - '0') * 5 + (seg[2] - '0') - 1;
            result.push_back(value);
        } else if (seg.length() == 4) {
            C u8 baseValue = seg[0] == 'R' ? 64 : 89;
            C u32 value = baseValue + (seg[1] - '0') * 5 + (seg[3] - '0') - 1;
            result.push_back(value);
        }
    }

    return result;
}

MU std::vector<u8> Memory::parseNormMoveString(C std::string& input) {
    return parseMoveStringTemplated<false>(input);
}

MU std::vector<u8> Memory::parseFatMoveString(C std::string& input) {
    return parseMoveStringTemplated<true>(input);
}





















// ############################################################
// #                     STATE HASH                           #
// ############################################################



namespace {
    
    // MU HD static u64 prime_func1(C u64 b1, C u64 b2) {
    //     return ((b1 << 4) + b1) ^ b2;
    // }
    
    MU HD FORCEINLINE u64 mix64(u64 x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }
    
    MU HD FORCEINLINE u64 prime_func1(C u64 b1, C u64 b2) {
        u64 x = b1 ^ 0x9e3779b97f4a7c15ULL;
        u64 y = b2 ^ 0xc2b2ae3d27d4eb4fULL;
    
        x = mix64(x);
        y = mix64(y);
    
        C u64 h = x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
        return mix64(h);
    }
    
    // check commits before 10/16/24 for previous impl.
    MU HD FORCEINLINE u64 getSegment2bits(C u64 segment) {
#ifndef __CUDA_ARCH__
        static constexpr u64 MASK_X0 = 0'111111'111111'111111;
        return my_pext_u64(segment, MASK_X0);
#else
        static constexpr u64 MASK_A1 = 0'101010'101010'101010;
        static constexpr u64 MASK_B1 = MASK_A1 >> 3;
        static constexpr u64 MASK_A2 = 0'030000'030000'030000;
        static constexpr u64 MASK_B2 = MASK_A2 >> 6;
        static constexpr u64 MASK_C2 = MASK_A2 >> 12;
        static constexpr u64 MASK_A3 = 0'000077'000000'000000;
        static constexpr u64 MASK_B3 = MASK_A3 >> 18;
        static constexpr u64 MASK_C3 = MASK_A3 >> 36;
        C u64 o1 = (segment & MASK_A1) >> 2 | segment & MASK_B1;
        C u64 o2 = (o1 & MASK_A2) >> 8 | (o1 & MASK_B2) >> 4 | o1 & MASK_C2;
        C u64 o3 = (o2 & MASK_A3) >> 24 | (o2 & MASK_B3) >> 12 | o2 & MASK_C3;
        return o3;
#endif
    }
    
    // check commits before 10/16/24 for previous impl.
    MU HD FORCEINLINE u64 getSegment3bits(C u64 segment) {
        static constexpr u64 MASK_CS = 0'003003'003003'003003;
        C u64 o1 = (segment >> 6 & MASK_CS) * 9 /* 0b1001 */
                   |
                   (segment >> 3 & MASK_CS) * 3 /* 0b1001 */
                   |
                   segment & MASK_CS
                ;
#ifndef __CUDA_ARCH__
        static constexpr u64 MASK_X23 = 0'037037'037037'037037;
        C u64 x23 = my_pext_u64(o1, MASK_X23);
        return x23;
#else
        static constexpr u64 MASK_A1 = 0'037000'037000'037000;
        static constexpr u64 MASK_B1 = MASK_A1 >> 9;
        static constexpr u64 MASK_A2 = 0'001777'000000'000000;
        static constexpr u64 MASK_B2 = MASK_A2 >> 18;
        static constexpr u64 MASK_C2 = MASK_A2 >> 36;
        C u64 o2 = (o1 & MASK_A1) >> 4 | o1 & MASK_B1;
        C u64 o3 = (o2 & MASK_A2) >> 16 | (o2 & MASK_B2) >> 8 | o2 & MASK_C2;
        return o3;
#endif
    }
    
    // check commits before 10/16/24 for previous impl.
    MU HD FORCEINLINE u64 getSegment4bits(C u64 segment) {
#ifndef __CUDA_ARCH__
        static constexpr u64 MASK_X0 = 0'333333'333333'333333;
        return my_pext_u64(segment, MASK_X0);
#else
        static constexpr u64 MASK_A1 = 0'303030'303030'303030;
        static constexpr u64 MASK_B1 = MASK_A1 >> 3;
        static constexpr u64 MASK_A2 = 0'170000'170000'170000;
        static constexpr u64 MASK_B2 = MASK_A2 >> 6;
        static constexpr u64 MASK_C2 = MASK_A2 >> 12;
        static constexpr u64 MASK_A3 = 0'007777'000000'000000;
        static constexpr u64 MASK_B3 = MASK_A3 >> 18;
        static constexpr u64 MASK_C3 = MASK_A3 >> 36;
        C u64 o1 = (segment & MASK_A1) >> 1 | segment & MASK_B1;
        C u64 o2 = (o1 & MASK_A2) >> 4 | (o1 & MASK_B2) >> 2 | o1 & MASK_C2;
        C u64 o3 = (o2 & MASK_A3) >> 12 | (o2 & MASK_B3) >> 6 | o2 & MASK_C3;
        return o3;
#endif
    }
} // hashing


StateHash::HashFuncPtr StateHash::gHashFunc_ = &StateHash::computeHash4;
StateHash::HashKind StateHash::gHashKind_ = StateHash::HashKind::Hash4;

u64 StateHash::computeHash2(C B1B2& state) {
    C u64 above = getSegment2bits(state.b1);
    C u64 below = getSegment2bits(state.b2);
    return (above << 18) | below;
}

u64 StateHash::computeHash3(C B1B2& state) {
    C u64 above = getSegment3bits(state.b1);
    C u64 below = getSegment3bits(state.b2);
    return (above << 30) | below;
}

u64 StateHash::computeHash4(C B1B2& state) {
    return prime_func1(state.b2, state.b1);
}

StateHash::HashKind StateHash::chooseHashKind(C B1B2& state) {
    C u64 colorCount = state.getColorCount();
    if (state.getFatBool() || colorCount > 3) {
        return HashKind::Hash4;
    }
    if (colorCount == 1 || colorCount == 2) {
        return HashKind::Hash2;
    }
    return HashKind::Hash3;
}

StateHash::HashFuncPtr StateHash::getHashFunc() {
    return gHashFunc_;
}

void StateHash::setHashKind(C HashKind kind) {
    gHashKind_ = kind;

    switch (kind) {
        case HashKind::Hash2:
            gHashFunc_ = &StateHash::computeHash2;
            break;
        case HashKind::Hash3:
            gHashFunc_ = &StateHash::computeHash3;
            break;
        case HashKind::Hash4:
        default:
            gHashFunc_ = &StateHash::computeHash4;
            break;
    }
}

void StateHash::refreshHashFunc(C B1B2& state) {
    setHashKind(chooseHashKind(state));
}

u64 StateHash::computeHash(C B1B2& state) {
    return gHashFunc_(state);
}




