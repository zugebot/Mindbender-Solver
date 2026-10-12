#pragma once
// src/chuzzles/lane_rotations.hpp
//
// Inline "all five shifts of one lane at once" for normal boards, used by the
// canonical stream's leaf level instead of five indirect action calls.
// Layout (board.cpp setAllColors): cell (row r, col c) of each 3-row half sits
// at bit 51 - 18 * (r % 3) - 3 * c, rows 0-2 in b1 and rows 3-5 in b2.
// Equivalence with allActStructList is checked in testExactSolverRegression.

#include "src/chuzzles/board.hpp"

#include <immintrin.h>

namespace lane_rotations {

    constexpr u64 LANE_BITS = 0x3FFFFULL;   // six 3-bit cells

    // A lane read as 18 bits with its first cell in the top 3 bits; shifting
    // the lane "forward" by n cells is a right rotation by 3n within 18 bits.
    FORCEINLINE u64 rotateLane(const u64 lane, const u32 cells) {
        const u32 s = 3 * cells;
        return ((lane >> s) | (lane << (18 - s))) & LANE_BITS;
    }

    // out[n - 1] = row `row` shifted right by n (R{row}{n}), n = 1..5.
    FORCEINLINE void rowShifts(const B1B2& board, const u32 row, B1B2 out[5]) {
        const u32 shift = 36 - 18 * (row % 3);
        const bool lower = row >= 3;
        const u64 half = lower ? board.b2 : board.b1;
        const u64 lane = (half >> shift) & LANE_BITS;
        const u64 rest = half & ~(LANE_BITS << shift);
        for (u32 n = 1; n <= 5; ++n) {
            const u64 moved = rest | (rotateLane(lane, n) << shift);
            out[n - 1] = lower ? B1B2(board.b1, moved) : B1B2(moved, board.b2);
        }
    }

    // out[m - 1] = column `col` shifted down by m (C{col}{m}), m = 1..5.
    FORCEINLINE void colShifts(const B1B2& board, const u32 col, B1B2 out[5]) {
        const u64 mask = (7ULL << (51 - 3 * col)) | (7ULL << (33 - 3 * col)) | (7ULL << (15 - 3 * col));
        const u64 lane = (_pext_u64(board.b1, mask) << 9) | _pext_u64(board.b2, mask);
        const u64 keep1 = board.b1 & ~mask;
        const u64 keep2 = board.b2 & ~mask;
        for (u32 m = 1; m <= 5; ++m) {
            const u64 moved = rotateLane(lane, m);
            out[m - 1] = B1B2(keep1 | _pdep_u64(moved >> 9, mask), keep2 | _pdep_u64(moved & 0x1FF, mask));
        }
    }

    FORCEINLINE u64 readRow(const B1B2& board, const u32 row) {
        const u64 half = row >= 3 ? board.b2 : board.b1;
        return (half >> (36 - 18 * (row % 3))) & LANE_BITS;
    }

    FORCEINLINE void writeRow(B1B2& board, const u32 row, const u64 lane) {
        const u32 shift = 36 - 18 * (row % 3);
        u64& half = row >= 3 ? board.b2 : board.b1;
        half = (half & ~(LANE_BITS << shift)) | (lane << shift);
    }

    FORCEINLINE u64 columnMask(const u32 col) {
        return (7ULL << (51 - 3 * col)) | (7ULL << (33 - 3 * col)) | (7ULL << (15 - 3 * col));
    }

    // Fat lanes move the two rows (or columns) under the 2x2 piece together
    // and carry the piece with them. out[n - 1] is the shift by n; the caller
    // skips the one amount that would put the piece across the board edge,
    // (5 - fatX) for rows and (5 - fatY) for columns, which the game forbids.
    FORCEINLINE void fatRowShifts(const B1B2& board, const u32 row, const u32 fatX, B1B2 out[5]) {
        const u64 laneA = readRow(board, row);
        const u64 laneB = readRow(board, row + 1);
        for (u32 n = 1; n <= 5; ++n) {
            B1B2 moved = board;
            writeRow(moved, row, rotateLane(laneA, n));
            writeRow(moved, row + 1, rotateLane(laneB, n));
            moved.b1 = (moved.b1 & b1b2::FAT_X_MASK) | (static_cast<u64>((fatX + n) % 6) << b1b2::FAT_X_OFFSET);
            out[n - 1] = moved;
        }
    }

    FORCEINLINE void fatColShifts(const B1B2& board, const u32 col, const u32 fatY, B1B2 out[5]) {
        const u64 maskA = columnMask(col);
        const u64 maskB = columnMask(col + 1);
        const u64 laneA = (_pext_u64(board.b1, maskA) << 9) | _pext_u64(board.b2, maskA);
        const u64 laneB = (_pext_u64(board.b1, maskB) << 9) | _pext_u64(board.b2, maskB);
        const u64 keep1 = board.b1 & ~(maskA | maskB);
        const u64 keep2 = board.b2 & ~(maskA | maskB);
        for (u32 m = 1; m <= 5; ++m) {
            const u64 movedA = rotateLane(laneA, m);
            const u64 movedB = rotateLane(laneB, m);
            u64 b1 = keep1 | _pdep_u64(movedA >> 9, maskA) | _pdep_u64(movedB >> 9, maskB);
            const u64 b2 = keep2 | _pdep_u64(movedA & 0x1FF, maskA) | _pdep_u64(movedB & 0x1FF, maskB);
            b1 = (b1 & b1b2::FAT_Y_MASK) | (static_cast<u64>((fatY + m) % 6) << b1b2::FAT_Y_OFFSET);
            out[m - 1] = B1B2(b1, b2);
        }
    }

} // namespace lane_rotations
