// LHeuristic.hpp
#pragma once
#include <cstdint>
#include <array>
#include <algorithm>

// Assumptions about your Board:
// struct Board {
//     int getColor(int x, int y) const; // x,y in [0..5]
// };

namespace lband_heuristic {

// Small helper: rotate a 6-bit mask "m" circularly right by k in [0..5].
static inline uint8_t rot6(uint8_t m, int k) {
    k &= 5; // mod 6
    if (k == 0) return uint8_t(m & 0x3F);
    uint8_t lo = uint8_t(m & ((1u << k) - 1u));
    return uint8_t(((m >> k) | (lo << (6 - k))) & 0x3F);
}

// For a 6-bit mask m, return:
//  - 0 if (m ⊆ allowed) already,
//  - 1 if ∃ rotation k ∈ [1..5] such that rot6(m,k) ⊆ allowed,
//  - 0 otherwise (admissible but weaker: columns/rows could fix the rest).
static inline uint8_t min_rot_to_subset6(uint8_t m, uint8_t allowed) {
    m &= 0x3F; allowed &= 0x3F;
    if ((m & ~allowed) == 0) return 0; // already subset
    for (int k = 1; k <= 5; ++k) {
        if ((rot6(m, k) & ~allowed) == 0) return 1;
    }
    // Can't be fixed by row/col rotation alone (we return 0 as a safe lower bound).
    return 0;
}

// Extract a 6-bit mask for a given COLOR on a specific row y: bit x = 1 iff board(x,y)==color.
template <class Board>
static inline uint8_t row_mask_color(const Board& b, int y, int color) {
    uint8_t m = 0;
    // x from 0..5 -> bit x
    for (int x = 0; x < 6; ++x) {
        // Expect getColor(x,y) is cheap; if not, you can inline a raw pointer read later.
        m |= (b.getColor(x, y) == color) ? uint8_t(1u << x) : uint8_t(0);
    }
    return m;
}

// Extract a 6-bit mask for a given COLOR on a specific column x: bit y = 1 iff board(x,y)==color.
template <class Board>
static inline uint8_t col_mask_color(const Board& b, int x, int color) {
    uint8_t m = 0;
    for (int y = 0; y < 6; ++y) {
        m |= (b.getColor(x, y) == color) ? uint8_t(1u << y) : uint8_t(0);
    }
    return m;
}

// Find the modal/majority color in a 3×3 block (ties broken by smallest color id).
template <class Board>
static inline int mode_color_3x3(const Board& goal, int x0, int y0) {
    // colors might be small ints; we avoid map to keep this tiny: track up to, say, 16 colors.
    // If your palette can exceed that, replace with an unordered_map<int,int>.
    // Here we use a tiny linear list (since 9 samples only).
    int colors[9]; int freq[9]; int used = 0;

    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
            int c = goal.getColor(x0 + dx, y0 + dy);
            int k = 0;
            for (; k < used; ++k) if (colors[k] == c) break;
            if (k == used) { colors[used] = c; freq[used] = 1; ++used; }
            else           { ++freq[k]; }
        }
    }
    int bestC = colors[0], bestF = freq[0];
    for (int i = 1; i < used; ++i) {
        if (freq[i] > bestF || (freq[i] == bestF && colors[i] < bestC)) {
            bestC = colors[i]; bestF = freq[i];
        }
    }
    return bestC;
}

// Holds four anchor colors (TL, TR, BL, BR) and provides the L-band heuristic.
struct LHeuristic {
    // Anchor colors (detected from goal or set manually).
    int cTL = 0, cTR = 1, cBL = 2, cBR = 3;

    // Allowed sets for columns & rows (6 bits across X or Y axis).
    static constexpr uint8_t COLS_L = 0b000111; // columns {0,1,2}
    static constexpr uint8_t COLS_R = 0b111000; // columns {3,4,5}
    static constexpr uint8_t ROWS_T = 0b000111; // rows    {0,1,2}
    static constexpr uint8_t ROWS_B = 0b111000; // rows    {3,4,5}

    LHeuristic() = default;

    template <class Board>
    static LHeuristic fromGoalBoard(const Board& goal) {
        LHeuristic H;
        H.cTL = mode_color_3x3(goal, /*x0=*/0, /*y0=*/0);
        H.cTR = mode_color_3x3(goal, /*x0=*/3, /*y0=*/0);
        H.cBL = mode_color_3x3(goal, /*x0=*/0, /*y0=*/3);
        H.cBR = mode_color_3x3(goal, /*x0=*/3, /*y0=*/3);
        return H;
    }

    // ----- Horizontal L (row-charged) per anchor -----
    // We only charge **row rotations** in the 3 rows that pass through the anchor.
    // Cost per row is 0 if its 6-bit mask is already subset of allowed columns,
    // or 1 if some rotation makes it so; else 0 (admissible lower bound).
    template <class Board>
    int HL_TL(const Board& b) const { // top band, anchor color cTL, allowed columns {0,1,2}
        int cost = 0;
        for (int y = 0; y < 3; ++y) {
            uint8_t m = row_mask_color(b, y, cTL);
            cost += min_rot_to_subset6(m, COLS_L);
        }
        return cost;
    }
    template <class Board>
    int HL_TR(const Board& b) const { // top band, anchor color cTR, allowed columns {3,4,5}
        int cost = 0;
        for (int y = 0; y < 3; ++y) {
            uint8_t m = row_mask_color(b, y, cTR);
            cost += min_rot_to_subset6(m, COLS_R);
        }
        return cost;
    }
    template <class Board>
    int HL_BL(const Board& b) const { // bottom band, anchor cBL, allowed columns {0,1,2}
        int cost = 0;
        for (int y = 3; y < 6; ++y) {
            uint8_t m = row_mask_color(b, y, cBL);
            cost += min_rot_to_subset6(m, COLS_L);
        }
        return cost;
    }
    template <class Board>
    int HL_BR(const Board& b) const { // bottom band, anchor cBR, allowed columns {3,4,5}
        int cost = 0;
        for (int y = 3; y < 6; ++y) {
            uint8_t m = row_mask_color(b, y, cBR);
            cost += min_rot_to_subset6(m, COLS_R);
        }
        return cost;
    }

    // ----- Vertical L (column-charged) per anchor -----
    // We only charge **column rotations** in the 3 columns that pass through the anchor.
    // Cost per column is 0 if its 6-bit mask is already subset of allowed rows,
    // or 1 if some rotation makes it so; else 0 (admissible).
    template <class Board>
    int VL_TL(const Board& b) const { // left band, anchor cTL, allowed rows {0,1,2}
        int cost = 0;
        for (int x = 0; x < 3; ++x) {
            uint8_t m = col_mask_color(b, x, cTL);
            cost += min_rot_to_subset6(m, ROWS_T);
        }
        return cost;
    }
    template <class Board>
    int VL_TR(const Board& b) const { // right band, anchor cTR, allowed rows {0,1,2}
        int cost = 0;
        for (int x = 3; x < 6; ++x) {
            uint8_t m = col_mask_color(b, x, cTR);
            cost += min_rot_to_subset6(m, ROWS_T);
        }
        return cost;
    }
    template <class Board>
    int VL_BL(const Board& b) const { // left band, anchor cBL, allowed rows {3,4,5}
        int cost = 0;
        for (int x = 0; x < 3; ++x) {
            uint8_t m = col_mask_color(b, x, cBL);
            cost += min_rot_to_subset6(m, ROWS_B);
        }
        return cost;
    }
    template <class Board>
    int VL_BR(const Board& b) const { // right band, anchor cBR, allowed rows {3,4,5}
        int cost = 0;
        for (int x = 3; x < 6; ++x) {
            uint8_t m = col_mask_color(b, x, cBR);
            cost += min_rot_to_subset6(m, ROWS_B);
        }
        return cost;
    }

    // Combine bands **without** double counting:
    //  - h_rows uses row-only costs, grouped by top/bottom (take max within each, then sum).
    //  - h_cols uses column-only costs, grouped by left/right (take max within each, then sum).
    //  - final = max(h_rows, h_cols).
    template <class Board>
    int eval(const Board& b) const {
        const int h_rows =
            std::max(HL_TL(b), HL_TR(b)) +
            std::max(HL_BL(b), HL_BR(b));

        const int h_cols =
            std::max(VL_TL(b), VL_BL(b)) +
            std::max(VL_TR(b), VL_BR(b));

        return std::max(h_rows, h_cols);
    }
};

} // namespace lband_heuristic
