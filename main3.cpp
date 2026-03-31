// main_bench_intersections.cpp
#include "MindbenderSolver/include.hpp"

#include <bitset>
#include <x86gprintrin.h>
#include <iomanip>
#include <iostream>
#include <string>

#define USE_PDEP_ROW

// keep signature exactly as-is
// keep signature exactly as-is
MU u64 getRowColIntersections(const Board& board, const u32 x, const u32 y) {
    // ====================== LUTs (64B aligned) ======================
    alignas(64) static constexpr u8  LEFT_BY_X[6]      = {15, 12,  9,  6,  3,  0};       // 15 - 3*x
    alignas(64) static constexpr u8  WORD_IDX_BY_Y[6]  = { 0,  0,  0,  1,  1,  1};       // 0->b1, 1->b2
    // top rows in b1 at 36/18/0, bottom rows in b2 at 56/38/20
    alignas(64) static constexpr u8  ROW_SHIFT_BY_Y[6] = {36, 18,  0, 56, 38, 20};

    // Column crop masks (pre-shifted to avoid variable shifts)
    alignas(64) static constexpr u32 COL_TOP_MASK[6] = {
        0x3FFFFFFFu << 30, 0x3FFFFFFFu << 25, 0x3FFFFFFFu << 20,
        0x3FFFFFFFu << 15, 0x3FFFFFFFu << 10, 0x3FFFFFFFu << 5
    };
    alignas(64) static constexpr u32 COL_BOT_MASK[6] = {
        0x1FFFFFFu >> 0,  0x1FFFFFFu >> 5,  0x1FFFFFFu >> 10,
        0x1FFFFFFu >> 15, 0x1FFFFFFu >> 20, 0x1FFFFFFu >> 25
    };

    // Row window masks (octal preserved) — for exact original behavior
    alignas(64) static constexpr u32 ROW_HI_MASKS[6] = {0'7700, 0'7700>>1, 0'7700>>2, 0'7700>>3, 0'7700>>4, 0'7700>>5};
    alignas(64) static constexpr u32 ROW_LO_MASKS[6] = {  0'37,   0'37>>1,   0'37>>2,   0'37>>3,   0'37>>4,   0'37>>5};

    // Center-line masks (unchanged)
    alignas(64) static constexpr u32 C_CNTR_MASKS[8] = {
        0x00000000u, 0x02108421u, 0x04210842u, 0x06318C63u,
        0x08421084u, 0x0A5294A5u, 0x0C6318C6u, 0x0E739CE7u
    };

    // Spaced bit positions {0,5,10,15,20,25}
    static constexpr u32 ROW_SPACED_MASK = 0x02108421u;   // six spaced slots
    static constexpr u32 ROW_SPACED_5    = 0x00108421u;   // five spaced slots (0,5,10,15,20)

    // 3-octal-groups mask for extracting column slices
    static constexpr u64 C_MAIN_MASK = 0'000007'000007'000007ULL;

    // ====================== Common subexpressions ======================
    const u32 left = LEFT_BY_X[x];
    const u64 b1v  = board.b1;
    const u64 b2v  = board.b2;

    // ---- Row extract (no UB shifts) ----
    const u64 row_word = WORD_IDX_BY_Y[y] ? b2v : b1v;
    const u32 row      = static_cast<u32>((row_word >> ROW_SHIFT_BY_Y[y]) & 0x3FFFFu); // 18 bits
    const u32 cntr_p1_r = (row >> left) & 0x7u;

    // ====================== Column pathway ======================
    const u64 col_mask = (C_MAIN_MASK << left);
    const u64 b1_c64   = (b1v & col_mask) >> left;
    const u64 b2_c64   = (b2v & col_mask) >> left;

    const u32 shifted_5 = static_cast<u32>(
          ((b2_c64 | (b2_c64 >> 13) | (b2_c64 >> 26)) & 0x1CE7ULL)
        | (((b1_c64 << 15) | (b1_c64 <<  2) | (b1_c64 >> 11)) & 0xE738000ULL)
    );

    const u32 s = shifted_5 ^ C_CNTR_MASKS[cntr_p1_r];

#if defined(__BMI__)
    const u32 merged = s | (s >> 1) | (s >> 2);
    const u32 n      = _andn_u32(merged, 0x02108421u); // mask & ~merged
#else
    const u32 n      = (~(s | (s >> 1) | (s >> 2))) & 0x02108421u;
#endif
    const u32 sim    = n * 31u;
    const u32 col_x5 = ((sim & COL_TOP_MASK[y]) >> 5) | (sim & COL_BOT_MASK[y]);

    // ====================== Row pathway (exact original math) ======================
    const u32 s_ps   = row ^ (cntr_p1_r * 0'111111);
#if defined(__BMI__)
    const u32 n_r    = _andn_u32((s_ps | (s_ps >> 1) | (s_ps >> 2)), 0'111111);
#else
    const u32 n_r    = ~(s_ps | (s_ps >> 1) | (s_ps >> 2)) & 0'111111;
#endif
    const u32 p1_r   = ((n_r & 0'101010) >> 2) | (n_r & 0'10101);
    const u32 row_t1 = ((p1_r >> 8) | (p1_r >> 4) | p1_r) & 0'77;

    // Delete column x, compact to 5 contiguous bits (original logic)
    const u32 row_bits5 = ((row_t1 & ROW_HI_MASKS[x]) >> 1) | (row_t1 & ROW_LO_MASKS[x]);

    // Map 5 contiguous bits into spaced slots {0,5,10,15,20,25} with a hole at 5*x
    u32 row_x5;
#if defined(__BMI2__) && defined(USE_PDEP_ROW)
    // Try both bit orders; fall back to multiply if neither matches
    const u32 mul = row_bits5 * 0x108421u; // ground truth (5 slots)
    const u32 deposit_mask = ROW_SPACED_MASK & ~(1u << (5 * x));

    auto rev5 = [](u32 v) -> u32 {
        v &= 0x1Fu;
        return ((v & 0x01u) << 4)
             | ((v & 0x02u) << 2)
             | ((v & 0x04u) << 0)
             | ((v & 0x08u) >> 2)
             | ((v & 0x10u) >> 4);
    };

    const u32 p0 = _pdep_u32(row_bits5, deposit_mask);
    if (p0 == mul) {
        row_x5 = p0;
    } else {
        const u32 p1 = _pdep_u32(rev5(row_bits5), deposit_mask);
        row_x5 = (p1 == mul) ? p1 : mul;  // guaranteed equal to original
    }
#else
    row_x5 = row_bits5 * 0x108421u;       // exact original mapping
#endif

    return static_cast<u64>(col_x5 & row_x5);
}




// ---------- Simple micro-benchmark harness ----------
template <class F>
static void bench(const char* name, const Board& b, F&& f,
                  const int iters, int rounds = 3) {
    // Each iteration calls all 36 (x,y) pairs.
    constexpr int CALLS_PER_ITER = 36;

    // warm-up
    {
        volatile u64 sink = 0;
        for (int i = 0; i < 1000; ++i)
            for (int y = 0; y < 6; ++y)
                for (int x = 0; x < 6; ++x)
                    sink ^= f(b, x, y);
    }

    double best_s = 1e100;
    u64 last_sink = 0;

    for (int r = 1; r <= rounds; ++r) {
        volatile u64 sink = 0;
        const Timer t;

        for (int i = 0; i < iters; ++i) {
            // fixed order for fairness and cache stability
            for (int y = 0; y < 6; ++y)
                for (int x = 0; x < 6; ++x)
                    sink ^= f(b, x, y);
        }

        const double s = t.getSeconds();
        const double calls = static_cast<double>(iters) * CALLS_PER_ITER;
        const double ops_per_sec = calls / s;

        std::cout << std::left << std::setw(28) << name
                  << " | round " << r
                  << " | time " << std::fixed << std::setprecision(3) << s << " s"
                  << " | calls " << static_cast<u64>(calls)
                  << " | ops/s " << static_cast<u64>(ops_per_sec)
                  << " | sink 0x" << std::hex << (u64)sink << std::dec
                  << "\n";

        if (s < best_s) { best_s = s; last_sink = sink; }
    }

    const double best_ops = (static_cast<double>(iters) * CALLS_PER_ITER) / best_s;
    std::cout << std::left << std::setw(28) << name
              << " | BEST   "
              << " | time " << std::fixed << std::setprecision(3) << best_s << " s"
              << " | ops/s " << static_cast<u64>(best_ops)
              << " | sink 0x" << std::hex << last_sink << std::dec
              << "\n\n";
}

int main(const int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    const char* id = (argc >= 2 ? argv[1] : "7-1");
    const auto pair  = BoardLookup::getBoardPair(id);
    const Board start = pair->getStartState();
    const Board goal  = pair->getEndState();

    // Optional: show the pair
    std::cout << pair->toString() << "\n";

    // Correctness check (all 36 positions)
    for (int x = 0; x < 6; ++x) {
        for (int y = 0; y < 6; ++y) {
            const u64 a = start.getRowColIntersections(x, y);
            const u64 b = getRowColIntersections(start, x, y);
            if (a != b) {
                std::cout << "HEURISTIC1 and HEURISTIC2 do not match!\n";
                std::cout << "[Offenders] x=" << x << " y=" << y << "\n";
                std::cout << "Original: " << std::bitset<64>(a) << "\n";
                std::cout << "Modified: " << std::bitset<64>(b) << "\n";
                return -1;
            }
        }
    }
    std::cout << "Equality check: OK (all 36 positions)\n";

    // ---------- Benchmarks ----------
    // Tweak these to taste; each "iter" calls the function 36 times (all x,y).
    constexpr int ITERS  = 1'000'000; // total calls = ITERS * 36
    constexpr int ROUNDS = 10;

    std::cout << "\n=== Bench getRowColIntersections ===\n";

    bench("Optimized (free)", start,
          [](const Board& b, u32 x, u32 y) -> u64 {
              return getRowColIntersections(b, x, y);
          },
          ITERS, ROUNDS);

    bench("Original (member)", start,
          [](const Board& b, u32 x, u32 y) -> u64 {
              return b.getRowColIntersections(x, y);
          },
          ITERS, ROUNDS);


    return 0;
}
