#pragma once
// src/solver/ssd/class_key.hpp
//
// Row and column class keys (the free-block join, docs/changes-2026-11-04.md section 15).
//
// A row move only rotates its row, so the row class of a board (every row turned to its
// smallest rotation; on 2x2 boards the piece's row pair turned so the piece is at column 0,
// with the piece's row kept) is unchanged by row moves, and two boards share it exactly when
// one block of row moves on distinct lanes turns one into the other. Column classes likewise.
// A board one move from another therefore shares a row or a column class with it, which lets
// the left side stop one move early: a depth-(L - 8) leaf is matched against the depth-7 goal
// side by class, and a match whose boards differ in exactly one lane is a middle.
//
// Plain C++ shared by the solver and the CUDA module (mbgpu.cu), so both compute identical
// keys. Layout (b1b2): cell (column x, row y) at bit 51 - 18 * (y % 3) - 3 * x of b1 (rows
// 0-2) or b2 (rows 3-5); the 2x2 piece: b1 bit 57 = present, bits 58-60 = its row, 61-63 =
// its column (0-4: it never wraps).

#include <cstdint>

#if defined(__CUDACC__)
#define MB_CK_HD __host__ __device__ __forceinline__
#else
#define MB_CK_HD inline
#endif

namespace class_key {

    using ck64 = std::uint64_t;
    using ck32 = std::uint32_t;

    constexpr ck64 LANE = 0x3FFFFULL;                    // one lane: 6 cells x 3 bits
    constexpr ck64 COLUMN_SALT = 0x9E3779B97F4A7C15ULL;  // keeps row and column keys apart

    // Rotate an 18-bit lane code left by `cells` cells (cell 0 is the top 3 bits).
    MB_CK_HD ck64 rotateCells(const ck64 v, const ck32 cells) {
        const ck32 s = 3 * (cells % 6);
        return s == 0 ? v : (((v << s) | (v >> (18 - s))) & LANE);
    }

    MB_CK_HD ck64 minRotation(const ck64 v) {
        ck64 m = v;
        for (ck32 k = 1; k < 6; ++k) {
            const ck64 r = rotateCells(v, k);
            m = r < m ? r : m;
        }
        return m;
    }

    // Row y as an 18-bit code, column 0 in the top bits.
    MB_CK_HD ck64 rowCode(const ck64 b1, const ck64 b2, const ck32 y) {
        return ((y < 3 ? b1 : b2) >> (36 - 18 * (y % 3))) & LANE;
    }

    // Column x as an 18-bit code, row 0 in the top bits.
    MB_CK_HD ck64 colCode(const ck64 b1, const ck64 b2, const ck32 x) {
        ck64 v = 0;
        for (ck32 y = 0; y < 6; ++y) {
            v = (v << 3) | (((y < 3 ? b1 : b2) >> (51 - 18 * (y % 3) - 3 * x)) & 7ULL);
        }
        return v;
    }

    MB_CK_HD bool fat(const ck64 b1) { return ((b1 >> 57) & 1ULL) != 0; }
    MB_CK_HD ck32 fatRow(const ck64 b1) { return static_cast<ck32>((b1 >> 58) & 7ULL); }
    MB_CK_HD ck32 fatCol(const ck64 b1) { return static_cast<ck32>((b1 >> 61) & 7ULL); }

    // b2 bit 63 (unused by boards): a left-side survivor that stands for its leaves (key mode 2:
    // a parent's row class is every row-move child's, so the parent is probed once for all of them).
    constexpr ck64 PARENT_FLAG = 1ULL << 63;

    struct LoopMinRotation {
        MB_CK_HD ck64 operator()(const ck64 v) const { return minRotation(v); }
    };

    // The same values from a 2^18-entry table (the GPU's; class_key::minRotation fills it).
    struct TableMinRotation {
        const ck32* table;
        MB_CK_HD ck64 operator()(const ck64 v) const { return table[v]; }
    };

    // The class of a board as 108 bits of lane codes plus the piece's fixed coordinate.
    template<typename MinRot>
    MB_CK_HD void classCodeT(const ck64 b1, const ck64 b2, const bool cols, ck64& lo, ck64& hi, const MinRot& minRot) {
        const bool f = fat(b1);
        const ck32 px = f ? fatCol(b1) : 0, py = f ? fatRow(b1) : 0;
        ck64 code[6];
        for (ck32 i = 0; i < 6; ++i) {
            const ck64 raw = cols ? colCode(b1, b2, i) : rowCode(b1, b2, i);
            // the piece's lane pair turns together: canonical = the piece at cell 0
            const bool pair = f && (cols ? (i == px || i == px + 1) : (i == py || i == py + 1));
            code[i] = pair ? rotateCells(raw, cols ? py : px) : minRot(raw);
        }
        lo = code[0] | (code[1] << 18) | (code[2] << 36);
        hi = code[3] | (code[4] << 18) | (code[5] << 36);
        // row moves keep the piece's row, column moves its column
        hi |= static_cast<ck64>(f ? 1 + (cols ? px : py) : 0) << 56;
    }

    MB_CK_HD void classCode(const ck64 b1, const ck64 b2, const bool cols, ck64& lo, ck64& hi) {
        classCodeT(b1, b2, cols, lo, hi, LoopMinRotation{});
    }

    MB_CK_HD ck64 mix(ck64 x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    // 64-bit key of the row class (cols = false) or column class; top 8 bits = partition.
    template<typename MinRot>
    MB_CK_HD ck64 classKeyT(const ck64 b1, const ck64 b2, const bool cols, const MinRot& minRot) {
        ck64 lo = 0, hi = 0;
        classCodeT(b1, b2, cols, lo, hi, minRot);
        const ck64 h = lo ^ ((hi << 1) | (hi >> 63)) ^ (lo >> 17) ^ (hi << 13);
        return mix(h ^ (cols ? COLUMN_SALT : 0));
    }

    MB_CK_HD ck64 classKey(const ck64 b1, const ck64 b2, const bool cols) {
        return classKeyT(b1, b2, cols, LoopMinRotation{});
    }

    // Lanes of one direction where two boards of the same class differ (the piece's pair is
    // one lane, and differs when its cells or the piece's position along it differ).
    MB_CK_HD ck32 differingLanes(const ck64 a1, const ck64 a2, const ck64 z1, const ck64 z2, const bool cols) {
        const bool f = fat(a1);
        const ck32 p = f ? (cols ? fatCol(a1) : fatRow(a1)) : 9;
        ck32 n = 0;
        bool pairDiffers = f && (cols ? fatRow(a1) != fatRow(z1) : fatCol(a1) != fatCol(z1));
        for (ck32 i = 0; i < 6; ++i) {
            const bool d = cols ? colCode(a1, a2, i) != colCode(z1, z2, i) : rowCode(a1, a2, i) != rowCode(z1, z2, i);
            if (i == p || i == p + 1) {
                pairDiffers = pairDiffers || d;
            } else if (d) {
                ++n;
            }
        }
        return n + (pairDiffers ? 1 : 0);
    }

} // namespace class_key
