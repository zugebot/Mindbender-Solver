// src/programs/experiments/experimentBoundSample.cpp
//
// Measures an admissible lower bound on the number of moves between two
// boards, to see how much of a meet-in-the-middle search it could skip.
//
// Bound ("colour mix"): a row move rotates its row, so every row keeps its
// colour multiset; a column move changes exactly one cell of every row. So
// the column moves still needed are at least R = the largest, over rows, of
// how many cells of that row's colour multiset differ from the target row's;
// likewise the row moves needed are at least C (same over columns).
// h = R + C. On 2x2 boards one move can shift two lanes (two cells per
// row/column), so h = ceil(R/2) + ceil(C/2). Either way h changes by at most
// 1 per move, so a whole subtree can be skipped once h exceeds the moves left.
//
//   experimentBoundSample --check
//       Replays every solution in build/levels_final (run from the repo root)
//       and verifies h never exceeds the true distance at any step, to the
//       goal or back to the start. Prints how strong h is per puzzle.
//   experimentBoundSample <puzzle> --seed-depth S --table-depth T [--samples N] [--threads N]
//       For the search <S,3,T>: the share of left states with h(goal) > T
//       (lookups that could be skipped), of seeds with h(goal) > T + 3 (whole
//       3-move streams), and of goal-side depth-T states with h(start) > S + 3
//       (index entries that could be dropped).
//   experimentBoundSample --bench <puzzle>
//       Fast (src/solver/colour_mix_bound.hpp) vs reference on 20 M real left
//       states: identical values, and ns/state on 1 and all threads.

#include "src/chuzzles/include.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/utils/intrinsics/pext_u64.hpp"
#include "src/utils/intrinsics/ctz.hpp"
#include "src/utils/intrinsics/popcount.hpp"
#include "src/solver/colour_mix_bound.hpp"

#include <array>
#include <functional>
#include <immintrin.h>
#include <bitset>
#include <cstring>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {
    using colour_mix::COL_MASK;
    using LaneMix = colour_mix::LaneMixRef;
    using colour_mix::LaneTables;
    using colour_mix::buildLaneTables;

    inline LaneMix laneMix(const B1B2& b) { return colour_mix::laneMixRef(b); }
    inline int colourMixBound(const B1B2& s, const LaneMix& t, const bool fat) { return colour_mix::referenceBound(s, t, fat); }
    FORCEINLINE int colourMixBoundTables(const B1B2& s, const LaneTables& t) { return colour_mix::bound(s, t); }


    // ------------------------------------------------------------------
    // Fast version on the packed board. b1 holds rows 0-2 and b2 rows 3-5,
    // each row 18 bits (row i of a half at bit (2 - i) * 18), cell x of a row
    // at bit 15 - 3x within it (see b1b2::getShiftAmount).
    // ------------------------------------------------------------------

    // Colour counts of 3 cells (9 bits): 8 colours x 4-bit counters. 2 KB, stays in L1.
    alignas(64) u32 TRIPLE_COUNTS[512];

    struct TripleCountsInit {
        TripleCountsInit() {
            for (u32 v = 0; v < 512; ++v) {
                TRIPLE_COUNTS[v] = (1U << (4 * (v & 7))) + (1U << (4 * ((v >> 3) & 7))) + (1U << (4 * (v >> 6)));
            }
        }
    } tripleCountsInit;

    FORCEINLINE u32 rowCounts(const u64 half, const u32 i) {
        const u32 v = static_cast<u32>(half >> ((2 - i) * 18)) & 0x3FFFFU;
        return TRIPLE_COUNTS[v & 511U] + TRIPLE_COUNTS[v >> 9];
    }

    FORCEINLINE u32 colCounts(const B1B2& b, const u32 x) {
        return TRIPLE_COUNTS[my_pext_u64(b.b1, COL_MASK[x])] + TRIPLE_COUNTS[my_pext_u64(b.b2, COL_MASK[x])];
    }

    // Sum over the 8 counters of min(a, b): the cells two lanes have in
    // common. Counters are <= 6, so (a | 8) - b never borrows across
    // counters, and its bit 3 is set exactly where a >= b.
    FORCEINLINE u32 commonCells(const u32 a, const u32 b) {
        const u32 t = (a | 0x88888888U) - b;
        const u32 ge = ((t >> 3) & 0x11111111U) * 0xFU;     // 0xF where a >= b
        const u32 m = (b & ge) | (a & ~ge);                 // per-counter min
        return (m * 0x11111111U) >> 28;                     // sum of 8 counters (<= 6)
    }

    struct GoalMix {
        u32 rows[6];
        u32 cols[6];
        bool fat;
    };

    GoalMix goalMix(const B1B2& goal) {
        GoalMix g{};
        for (u32 i = 0; i < 3; ++i) {
            g.rows[i] = rowCounts(goal.b1, i);
            g.rows[3 + i] = rowCounts(goal.b2, i);
        }
        for (u32 x = 0; x < 6; ++x) {
            g.cols[x] = colCounts(goal, x);
        }
        g.fat = goal.getFatBool();
        return g;
    }

    FORCEINLINE int colourMixBoundFast(const B1B2& s, const GoalMix& g) {
        // R = 6 - fewest common cells over the rows, C likewise over the columns.
        u32 rowCommon = commonCells(rowCounts(s.b1, 0), g.rows[0]);
        rowCommon = std::min(rowCommon, commonCells(rowCounts(s.b1, 1), g.rows[1]));
        rowCommon = std::min(rowCommon, commonCells(rowCounts(s.b1, 2), g.rows[2]));
        rowCommon = std::min(rowCommon, commonCells(rowCounts(s.b2, 0), g.rows[3]));
        rowCommon = std::min(rowCommon, commonCells(rowCounts(s.b2, 1), g.rows[4]));
        rowCommon = std::min(rowCommon, commonCells(rowCounts(s.b2, 2), g.rows[5]));
        u32 colCommon = commonCells(colCounts(s, 0), g.cols[0]);
        for (u32 x = 1; x < 6; ++x) {
            colCommon = std::min(colCommon, commonCells(colCounts(s, x), g.cols[x]));
        }
        const int r = 6 - static_cast<int>(rowCommon), c = 6 - static_cast<int>(colCommon);
        return g.fat ? (r + 1) / 2 + (c + 1) / 2 : r + c;
    }

    template<typename Fn>
    void parallelFor(const u32 threads, const std::size_t count, Fn&& fn) {
        std::atomic<std::size_t> next = 0;
        std::vector<std::thread> pool;
        for (u32 t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                for (std::size_t i; (i = next.fetch_add(1)) < count;) {
                    fn(t, i);
                }
            });
        }
        for (auto& th : pool) {
            th.join();
        }
    }

    template<typename Fn>
    struct CallbackSink {
        Fn& fn;
        void operator()(JVec<B1B2>& states, JVec<u64>&, const u32 count) {
            for (u32 i = 0; i < count; ++i) {
                fn(states[i]);
            }
        }
    };

    template<typename Fn>
    void streamFrom(const B1B2& root, const u32 depth, Fn&& fn) {
        CallbackSink<Fn> sink{fn};
        PermStream<B1B2>::streamDepthRuntime<eSequenceDir::ASCENDING>(makeBoardFromState(root), sink, depth, 4096);
    }

    void buildFrontier(const Board& root, const u32 depth, JVec<B1B2>& out) {
        JVec<u64> hashes;
        switch (depth) {
            case 1: buildUniqueNoneDepthFrontierB1B2<1>(root, out, hashes, false); return;
            case 2: buildUniqueNoneDepthFrontierB1B2<2>(root, out, hashes, false); return;
            case 3: buildUniqueNoneDepthFrontierB1B2<3>(root, out, hashes, false); return;
            case 4: buildUniqueNoneDepthFrontierB1B2<4>(root, out, hashes, false); return;
            case 5: buildUniqueNoneDepthFrontierB1B2<5>(root, out, hashes, false); return;
            default: throw std::runtime_error("depth must be 1..5");
        }
    }

    struct Histogram {
        std::array<std::atomic<u64>, 16> bins{};
        void add(const int h) { bins[std::min(h, 15)].fetch_add(1, std::memory_order_relaxed); }
        u64 total() const {
            u64 n = 0;
            for (const auto& b : bins) n += b.load();
            return n;
        }
        u64 above(const int limit) const {
            u64 n = 0;
            for (int h = limit + 1; h < 16; ++h) n += bins[h].load();
            return n;
        }
        void print(const std::string& label, const int limit) const {
            const u64 n = std::max<u64>(1, total());
            tcout << label << ": " << total() << " sampled, h > " << limit << " for "
                  << 100.0 * static_cast<double>(above(limit)) / static_cast<double>(n) << "%\n    h:";
            for (int h = 0; h < 16; ++h) {
                if (bins[h].load() != 0) {
                    tcout << "  " << h << "=" << 100.0 * static_cast<double>(bins[h].load()) / static_cast<double>(n) << "%";
                }
            }
            tcout << "\n";
        }
    };

    int runCheck() {
        const std::regex name(R"((\d+-\d+)_c(\d+)_(\d+)\.txt)");
        std::vector<fs::path> files;
        for (const auto& e : fs::directory_iterator("build/levels_final")) {
            files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        u64 states = 0, violations = 0;
        i64 rotGain = 0;
        for (const fs::path& f : files) {
            std::smatch m;
            const std::string fn = f.filename().string();
            if (!std::regex_match(fn, m, name)) {
                continue;
            }
            const BoardPair* pair = BoardLookup::getBoardPair(m[1].str());
            if (pair == nullptr) {
                continue;
            }
            const Board start = pair->getStartState();
            const Board goal = pair->getEndState();
            const bool fat = start.getFatBool();
            const LaneMix toGoal = laneMix(goal.asB1B2());
            const LaneMix toStart = laneMix(start.asB1B2());
            std::ifstream in(f);
            std::string line;
            int lines = 0, worstGap = 99;
            while (std::getline(in, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }
                const std::vector<u8> moves = fat ? Memory::parseFatMoveString(line) : Memory::parseNormMoveString(line);
                const int n = static_cast<int>(moves.size());
                Board b = start;
                for (int i = 0; i <= n; ++i) {
                    const int hg = colourMixBound(b.asB1B2(), toGoal, fat);
                    const int hs = colourMixBound(b.asB1B2(), toStart, fat);
                    const int rg = colour_mix::referenceBound(b.asB1B2(), goal.asB1B2(), colour_mix::Kind::Rotation);
                    const int rs = colour_mix::referenceBound(b.asB1B2(), start.asB1B2(), colour_mix::Kind::Rotation);
                    ++states;
                    if (hg > n - i || hs > i || rg > n - i || rs > i || rg < hg) {
                        if (violations < 20) {
                            tcout << "VIOLATION " << fn << " step " << i << ": h(goal)=" << hg << " rot " << rg
                                  << " vs " << n - i << ", h(start)=" << hs << " rot " << rs << " vs " << i << " in "
                                  << line << "\n";
                        }
                        ++violations;
                    }
                    rotGain += rg - hg;
                    if (i == 0) {
                        worstGap = std::min(worstGap, n - hg);
                    }
                    if (i < n) {
                        allActStructList[moves[i]].action(b);
                    }
                }
                ++lines;
            }
            const int h0 = colourMixBound(start.asB1B2(), toGoal, fat);
            tcout << m[1].str() << (fat ? " (2x2)" : "") << ": optimal " << m[2].str() << ", h(start->goal) = " << h0
                  << ", " << lines << " solutions checked\n";
        }
        tcout << "rotation bound: mean " << static_cast<double>(rotGain) / static_cast<double>(std::max<u64>(1, states))
              << " above colour-mix\n";
        tcout << "CHECK (colour-mix and rotation, to goal and to start): " << states << " states on known optimal paths, "
              << violations << " violations -> "
              << (violations == 0 ? "ADMISSIBLE" : "NOT ADMISSIBLE") << "\n";
        return violations == 0 ? 0 : 1;
    }

    int runSample(const std::string& puzzle, const u32 seedDepth, const u32 tableDepth, const u32 samples,
                  const u32 threads) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        const bool fat = start.getFatBool();
        const LaneMix toGoal = laneMix(goal.asB1B2());
        const LaneMix toStart = laneMix(start.asB1B2());
        const int leftDepth = static_cast<int>(seedDepth) + 3;
        tcout << puzzle << (fat ? " (2x2)" : "") << " <" << seedDepth << ",3," << tableDepth << "> = "
              << leftDepth + static_cast<int>(tableDepth) << " moves; h(start->goal) = "
              << colourMixBound(start.asB1B2(), toGoal, fat) << "\n";

        Timer timer;
        JVec<B1B2> seeds;
        buildFrontier(start, seedDepth, seeds);
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / samples);
        Histogram seedH, leftH;
        std::atomic<u64> streamedFromKept{0};
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            const B1B2& seed = seeds[k * step];
            const int hs = colourMixBound(seed, toGoal, fat);
            seedH.add(hs);
            u64 n = 0;
            streamFrom(seed, 3, [&](const B1B2& s) {
                leftH.add(colourMixBound(s, toGoal, fat));
                ++n;
            });
            if (hs <= static_cast<int>(tableDepth) + 3) {
                streamedFromKept.fetch_add(n);
            }
        });
        tcout << "left seeds (exact depth " << seedDepth << "): " << seeds.size() << ", sampled every " << step
              << " (" << timer.getSeconds() << "s)\n";
        seedH.print("  seeds, h(goal)", static_cast<int>(tableDepth) + 3);
        leftH.print("  left states, h(goal)", static_cast<int>(tableDepth));
        tcout << "  left states under seeds that can't be skipped: "
              << 100.0 * static_cast<double>(streamedFromKept.load()) / static_cast<double>(std::max<u64>(1, leftH.total()))
              << "%\n";

        // One level above the leaves: a node with h > T + 1 has only children
        // with h > T (h changes by <= 1 per move), so its whole last level of
        // the stream could be skipped without generating it.
        Histogram parentH;
        std::atomic<u64> parentsSkippable{0}, leavesUnderSkippable{0}, leavesAll{0};
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            streamFrom(seeds[k * step], 2, [&](const B1B2& node) {
                const int h = colourMixBound(node, toGoal, fat);
                parentH.add(h);
                u64 n = 0;
                streamFrom(node, 1, [&](const B1B2&) { ++n; });
                leavesAll.fetch_add(n, std::memory_order_relaxed);
                if (h > static_cast<int>(tableDepth) + 1) {
                    parentsSkippable.fetch_add(1, std::memory_order_relaxed);
                    leavesUnderSkippable.fetch_add(n, std::memory_order_relaxed);
                }
            });
        });
        parentH.print("  depth-" + std::to_string(seedDepth + 2) + " nodes, h(goal)", static_cast<int>(tableDepth) + 1);
        tcout << "  (approx.) leaves under those nodes: "
              << 100.0 * static_cast<double>(leavesUnderSkippable.load()) / static_cast<double>(std::max<u64>(1, leavesAll.load()))
              << "% of a one-move expansion of every node\n";

        Timer rightTimer;
        JVec<B1B2> depth5;
        buildFrontier(goal, 5, depth5);
        const u32 extra = tableDepth - 5;
        const std::size_t rstep = std::max<std::size_t>(1, depth5.size() / samples);
        Histogram rightH;
        parallelFor(threads, (depth5.size() + rstep - 1) / rstep, [&](u32, const std::size_t k) {
            streamFrom(depth5[k * rstep], extra, [&](const B1B2& s) { rightH.add(colourMixBound(s, toStart, fat)); });
        });
        tcout << "goal-side depth-5 frontier: " << depth5.size() << ", sampled every " << rstep << ", +" << extra
              << " stream moves (" << rightTimer.getSeconds() << "s)\n";
        rightH.print("  goal-side depth-" + std::to_string(tableDepth) + " states, h(start)", leftDepth);
        return 0;
    }

    // Fast vs reference bound on real left states: identical values, and ns/state.
    int runBench(const std::string& puzzle, const u32 seedDepth, const u32 threads) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        const bool fat = start.getFatBool();
        const LaneMix ref = laneMix(goal.asB1B2());
        const GoalMix fast = goalMix(goal.asB1B2());

        JVec<B1B2> seeds;
        buildFrontier(start, seedDepth, seeds);
        constexpr std::size_t WANT = 20'000'000;
        std::vector<B1B2> states;
        states.reserve(WANT + 1'000'000);
        for (std::size_t i = 0; i < seeds.size() && states.size() < WANT; i += std::max<std::size_t>(1, seeds.size() / 400)) {
            streamFrom(seeds[i], 3, [&](const B1B2& s) { states.push_back(s); });
        }
        tcout << puzzle << ": " << states.size() << " left states from " << seeds.size() << " depth-" << seedDepth
              << " seeds\n";

        u64 mismatches = 0;
        for (const B1B2& s : states) {
            mismatches += colourMixBound(s, ref, fat) != colourMixBoundFast(s, fast);
        }
        tcout << "fast vs reference: " << mismatches << " mismatches of " << states.size() << "\n";

        auto timeIt = [&](const char* label, auto&& fn) {
            u64 sink = 0;
            Timer t;
            for (int rep = 0; rep < 3; ++rep) {
                for (const B1B2& s : states) {
                    sink += static_cast<u64>(fn(s));
                }
            }
            const double ns = t.getSeconds() * 1e9 / (3.0 * static_cast<double>(states.size()));
            tcout << "  " << label << ": " << ns << " ns/state (1 thread, sink " << sink << ")\n";
            return ns;
        };
        const double base = timeIt("loop only (read state)", [](const B1B2& s) { return (s.b1 ^ s.b2) & 1; });
        timeIt("one row move (R1 +1) for reference", [](const B1B2& s) {
            B1B2 b = s;
            allActStructList[0].action(b);
            return (b.b1 ^ b.b2) & 1;
        });
        timeIt("reference bound", [&](const B1B2& s) { return colourMixBound(s, ref, fat); });
        const double f = timeIt("SWAR bound", [&](const B1B2& s) { return colourMixBoundFast(s, fast); });
        tcout << "SWAR bound net of the loop: " << f - base << " ns/state\n";

        LaneTables tabs;
        buildLaneTables(goal.asB1B2(), tabs);
        u64 tableMismatches = 0;
        for (const B1B2& s : states) {
            tableMismatches += colourMixBound(s, ref, fat) != colourMixBoundTables(s, tabs);
        }
        tcout << "lane tables: " << tabs.bytes << " bytes; vs reference: " << tableMismatches << " mismatches of "
              << states.size() << "\n";
        const double tt = timeIt("lane-table bound", [&](const B1B2& s) { return colourMixBoundTables(s, tabs); });
        tcout << "lane-table bound net of the loop: " << tt - base << " ns/state\n";

        // All threads at once (shares the core with hyperthreads, like the search).
        auto allThreads = [&](const char* label, auto&& fn) {
            std::atomic<u64> sink{0};
            Timer t;
            parallelFor(threads, threads, [&](u32, const std::size_t k) {
                // Each thread a contiguous block, small enough to stay in cache
                // (the search bounds states straight out of its 4096-state chunks).
                u64 local = 0;
                const std::size_t lo = states.size() * k / threads, hi = states.size() * (k + 1) / threads;
                constexpr std::size_t CHUNK = 4096;
                for (std::size_t c = lo; c < hi; c += CHUNK) {
                    const std::size_t e = std::min(hi, c + CHUNK);
                    for (int rep = 0; rep < 3; ++rep) {
                        for (std::size_t i = c; i < e; ++i) {
                            local += static_cast<u64>(fn(states[i]));
                        }
                    }
                }
                sink += local;
            });
            const double total = 3.0 * static_cast<double>(states.size());
            tcout << label << ", " << threads << " threads: " << total / t.getSeconds() / 1e6 << " M states/s ("
                  << t.getSeconds() * 1e9 * threads / total << " ns/state per thread)\n";
        };
        allThreads("SWAR bound", [&](const B1B2& s) { return colourMixBoundFast(s, fast); });
        allThreads("lane-table bound", [&](const B1B2& s) { return colourMixBoundTables(s, tabs); });
        return mismatches == 0 && tableMismatches == 0 ? 0 : 1;
    }

    // ------------------------------------------------------------------
    // Pattern databases: one colour at a time, solved exactly.
    //
    // Abstract board for colour c: the same B1B2 with every cell of colour c
    // set to 1 and every other cell 0 (the 2x2 piece's position bits kept).
    // Moves only permute cells, so the real move code runs on it unchanged.
    // Any real solution is also a solution of the abstract puzzle, so the
    // exact abstract distance (BFS over every placement of c's k cells, times
    // the 25 piece positions on 2x2 boards) is an admissible lower bound.
    // ------------------------------------------------------------------
    constexpr u64 LOW_BITS = 0111111111111111111ULL;   // low bit of each of 18 3-bit cells

    u64 BINOM[37][37];

    struct BinomInit {
        BinomInit() {
            for (u32 n = 0; n <= 36; ++n) {
                BINOM[n][0] = 1;
                for (u32 k = 1; k <= 36; ++k) {
                    BINOM[n][k] = n == 0 ? 0 : BINOM[n - 1][k - 1] + BINOM[n - 1][k];
                }
            }
        }
    } binomInit;

    FORCEINLINE u64 colourFlags(const u64 half, const u32 c) {
        const u64 x = ~(half ^ (static_cast<u64>(c) * LOW_BITS));
        return x & (x >> 1) & (x >> 2) & LOW_BITS;
    }

    FORCEINLINE B1B2 abstractBoard(const B1B2& s, const u32 c) {
        return {(s.b1 & ~((1ULL << 54) - 1)) | colourFlags(s.b1, c), colourFlags(s.b2, c)};
    }

    struct ColourPdb {
        u32 colour = 0, cells = 0;
        bool fat = false;
        std::vector<u8> dist;   // 255 = unreachable
        u32 maxDist = 0;
        double buildSeconds = 0.0;

        FORCEINLINE u64 indexOfAbstract(const B1B2& a) const {
            u64 m = (my_pext_u64(a.b1, LOW_BITS) << 18) | my_pext_u64(a.b2, LOW_BITS);
            u64 r = 0;
            for (u32 i = 1; m != 0; ++i, m &= m - 1) {
                r += BINOM[my_ctz64(m)][i];
            }
            return fat ? r * 25 + a.getFatXY() : r;
        }

        FORCEINLINE u32 lookup(const B1B2& s) const {
            return dist[indexOfAbstract(abstractBoard(s, colour))];
        }
    };

    ColourPdb buildColourPdb(const B1B2& target, const u32 colour, const u32 threads) {
        Timer timer;
        ColourPdb p;
        p.colour = colour;
        p.fat = target.getFatBool();
        const B1B2 goal = abstractBoard(target, colour);
        p.cells = static_cast<u32>(my_popcount(goal.b1 & LOW_BITS) + my_popcount(goal.b2 & LOW_BITS));
        p.dist.assign(BINOM[36][p.cells] * (p.fat ? 25 : 1), 255);
        p.dist[p.indexOfAbstract(goal)] = 0;
        std::vector<B1B2> frontier{goal};
        for (u32 d = 0; !frontier.empty(); ++d) {
            std::vector<std::vector<B1B2>> next(threads);
            constexpr std::size_t CHUNK = 4096;
            parallelFor(threads, (frontier.size() + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t ch) {
                B1B2 children[60];
                u64 hashes[60];
                const std::size_t end = std::min(frontier.size(), (ch + 1) * CHUNK);
                for (std::size_t i = ch * CHUNK; i < end; ++i) {
                    const std::size_t n = p.fat
                            ? frontier_recovery_detail::emitNoneChildren<true>(frontier[i], children, hashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(frontier[i], children, hashes);
                    for (std::size_t k = 0; k < n; ++k) {
                        u8* slot = &p.dist[p.indexOfAbstract(children[k])];
                        u8 expected = 255;
                        if (*slot == 255 && __atomic_compare_exchange_n(slot, &expected, static_cast<u8>(d + 1), false,
                                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            next[t].push_back(children[k]);
                        }
                    }
                }
            });
            frontier.clear();
            for (auto& v : next) {
                frontier.insert(frontier.end(), v.begin(), v.end());
            }
            if (!frontier.empty()) {
                p.maxDist = d + 1;
            }
        }
        p.buildSeconds = timer.getSeconds();
        return p;
    }

    // Colour-mix vs pattern databases vs both, on real left states and on
    // nodes one move above the leaves (whole last level skippable).
    int runPdb(const std::string& puzzle, const u32 seedDepth, const u32 tableDepth, const u32 samples,
               const u32 threads, const u32 kMax) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        const bool fat = start.getFatBool();
        const int T = static_cast<int>(tableDepth);
        tcout << puzzle << (fat ? " (2x2)" : "") << " <" << seedDepth << ",3," << tableDepth << ">, threads " << threads
              << "\n";

        LaneTables mix;
        buildLaneTables(goal.asB1B2(), mix);
        u32 counts[8] = {};
        for (u64 y = 0; y < 6; ++y) {
            for (u64 x = 0; x < 6; ++x) {
                ++counts[goal.getColor(x, y)];
            }
        }
        std::vector<ColourPdb> pdbs;
        for (u32 c = 0; c < 8; ++c) {
            if (counts[c] == 0 || counts[c] > kMax) {
                continue;
            }
            pdbs.push_back(buildColourPdb(goal.asB1B2(), c, threads));
            const ColourPdb& p = pdbs.back();
            const B1B2 sa = start.asB1B2();
            tcout << "  pdb colour " << c << ": " << p.cells << " cells, " << p.dist.size() << " entries, max distance "
                  << p.maxDist << ", h(start) = " << p.lookup(sa) << ", built in " << p.buildSeconds << "s\n";
        }
        tcout << "  colour-mix h(start) = " << colour_mix::bound(start.asB1B2(), mix) << "\n";
        if (pdbs.empty()) {
            tcout << "  no colour with <= " << kMax << " cells\n";
        }
        auto pdbMax = [&](const B1B2& s) {
            u32 h = 0;
            for (const ColourPdb& p : pdbs) {
                h = std::max(h, p.lookup(s));
            }
            return static_cast<int>(h);
        };

        JVec<B1B2> seeds;
        buildFrontier(start, seedDepth, seeds);
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / samples);
        std::atomic<u64> leaves{0}, lMix{0}, lPdb{0}, lAny{0};
        std::atomic<u64> nodes{0}, nMix{0}, nPdb{0}, nAny{0};
        std::atomic<u64> pdbWins{0};
        Histogram pdbH, mixH;
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            streamFrom(seeds[k * step], 2, [&](const B1B2& node) {
                const int nm = colour_mix::bound(node, mix), np = pdbMax(node);
                nodes.fetch_add(1, std::memory_order_relaxed);
                nMix.fetch_add(nm > T + 1, std::memory_order_relaxed);
                nPdb.fetch_add(np > T + 1, std::memory_order_relaxed);
                nAny.fetch_add(std::max(nm, np) > T + 1, std::memory_order_relaxed);
                streamFrom(node, 1, [&](const B1B2& leaf) {
                    const int hm = colour_mix::bound(leaf, mix), hp = pdbMax(leaf);
                    leaves.fetch_add(1, std::memory_order_relaxed);
                    lMix.fetch_add(hm > T, std::memory_order_relaxed);
                    lPdb.fetch_add(hp > T, std::memory_order_relaxed);
                    lAny.fetch_add(std::max(hm, hp) > T, std::memory_order_relaxed);
                    pdbWins.fetch_add(hp > hm, std::memory_order_relaxed);
                    pdbH.add(hp);
                    mixH.add(hm);
                });
            });
        });
        auto pct = [](const std::atomic<u64>& a, const std::atomic<u64>& n) {
            return 100.0 * static_cast<double>(a.load()) / static_cast<double>(std::max<u64>(1, n.load()));
        };
        tcout << "  leaves (" << leaves.load() << "): skip colour-mix " << pct(lMix, leaves) << "%, pdb " << pct(lPdb, leaves)
              << "%, either " << pct(lAny, leaves) << "% (pdb higher on " << pct(pdbWins, leaves) << "%)\n";
        tcout << "  nodes (" << nodes.load() << "): whole last level skip colour-mix " << pct(nMix, nodes) << "%, pdb "
              << pct(nPdb, nodes) << "%, either " << pct(nAny, nodes) << "%\n";
        pdbH.print("  pdb h on leaves", T);
        mixH.print("  colour-mix h on leaves", T);
        return 0;
    }

    // Every known optimal path: the pattern-database bound never exceeds the
    // true distance to the goal (a sanity check of the BFS and the ranking).
    int runPdbCheck(const u32 threads, const u32 kMax) {
        const std::regex name(R"((\d+-\d+)_c(\d+)_(\d+)\.txt)");
        u64 states = 0, violations = 0;
        for (const auto& e : fs::directory_iterator("build/levels_final")) {
            std::smatch m;
            const std::string fn = e.path().filename().string();
            if (!std::regex_match(fn, m, name)) {
                continue;
            }
            const BoardPair* pair = BoardLookup::getBoardPair(m[1].str());
            if (pair == nullptr) {
                continue;
            }
            const Board start = pair->getStartState();
            const Board goal = pair->getEndState();
            const bool fat = start.getFatBool();
            u32 counts[8] = {};
            for (u64 y = 0; y < 6; ++y) {
                for (u64 x = 0; x < 6; ++x) {
                    ++counts[goal.getColor(x, y)];
                }
            }
            std::vector<ColourPdb> pdbs;
            for (u32 c = 0; c < 8; ++c) {
                if (counts[c] != 0 && counts[c] <= kMax) {
                    pdbs.push_back(buildColourPdb(goal.asB1B2(), c, threads));
                }
            }
            std::ifstream in(e.path());
            std::string line;
            int best = 0;
            while (std::getline(in, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }
                const std::vector<u8> moves = fat ? Memory::parseFatMoveString(line) : Memory::parseNormMoveString(line);
                const int n = static_cast<int>(moves.size());
                Board b = start;
                for (int i = 0; i <= n; ++i) {
                    for (const ColourPdb& p : pdbs) {
                        const int h = static_cast<int>(p.lookup(b.asB1B2()));
                        if (i == 0) {
                            best = std::max(best, h);
                        }
                        ++states;
                        if (h > n - i) {
                            if (violations < 10) {
                                tcout << "VIOLATION " << fn << " step " << i << " colour " << p.colour << ": h=" << h
                                      << " > " << n - i << "\n";
                            }
                            ++violations;
                        }
                    }
                    if (i < n) {
                        allActStructList[moves[i]].action(b);
                    }
                }
            }
            tcout << m[1].str() << (fat ? " (2x2)" : "") << ": optimal " << m[2].str() << ", pdb h(start) = " << best
                  << " (" << pdbs.size() << " colours)\n";
        }
        tcout << "PDB CHECK: " << states << " lookups on known optimal paths, " << violations << " violations -> "
              << (violations == 0 ? "ADMISSIBLE" : "NOT ADMISSIBLE") << "\n";
        return violations == 0 ? 0 : 1;
    }

    // ------------------------------------------------------------------
    // Grouped pattern databases: colours mapped to classes A (value 1),
    // B (value 2) and blank (0). One class = several colours merged; A and B =
    // two colour groups told apart. Index = rank of A's cells, then rank of
    // B's cells among the remaining positions (x 25 piece positions on 2x2
    // boards). BFS layer by layer over the distance array (no frontier kept).
    // ------------------------------------------------------------------
    FORCEINLINE u64 colexRank(u64 m) {
        u64 r = 0;
        for (u32 i = 1; m != 0; ++i, m &= m - 1) {
            r += BINOM[my_ctz64(m)][i];
        }
        return r;
    }

    // k-subset of n positions with colex rank r
    FORCEINLINE u64 colexUnrank(u64 r, const u32 k, const u32 n) {
        u64 m = 0;
        u32 p = n;
        for (u32 i = k; i >= 1; --i) {
            do {
                --p;
            } while (BINOM[p][i] > r);
            r -= BINOM[p][i];
            m |= 1ULL << p;
        }
        return m;
    }

    struct GroupPdb {
        u8 classOf[8] = {};       // 0 blank, 1 A, 2 B
        u32 a = 0, b = 0;
        bool fat = false;
        u64 bStates = 1, entries = 0;
        std::vector<u8> dist;
        u32 maxDist = 0;
        double buildSeconds = 0.0;
        std::string name;

        FORCEINLINE B1B2 abstractOf(const B1B2& s) const {
            u64 h[2] = {s.b1, s.b2}, out[2] = {0, 0};
            for (u32 c = 0; c < 8; ++c) {
                if (classOf[c] == 0) {
                    continue;
                }
                for (u32 k = 0; k < 2; ++k) {
                    out[k] |= colourFlags(h[k], c) << (classOf[c] - 1);
                }
            }
            return {(s.b1 & ~((1ULL << 54) - 1)) | out[0], out[1]};
        }

        FORCEINLINE u64 indexOfAbstract(const B1B2& x) const {
            const u64 mA = (my_pext_u64(x.b1, LOW_BITS) << 18) | my_pext_u64(x.b2, LOW_BITS);
            const u64 mB = (my_pext_u64(x.b1 >> 1, LOW_BITS) << 18) | my_pext_u64(x.b2 >> 1, LOW_BITS);
            const u64 free = ~mA & ((1ULL << 36) - 1);
            const u64 r = colexRank(mA) * bStates + colexRank(my_pext_u64(mB, free));
            return fat ? r * 25 + x.getFatXY() : r;
        }

        B1B2 boardOfIndex(u64 idx, const B1B2& meta) const {
            u32 fatXY = 0;
            if (fat) {
                fatXY = static_cast<u32>(idx % 25);
                idx /= 25;
            }
            const u64 mA = colexUnrank(idx / bStates, a, 36);
            const u64 free = ~mA & ((1ULL << 36) - 1);
            const u64 mB = _pdep_u64(colexUnrank(idx % bStates, b, 36 - a), free);
            B1B2 x((meta.b1 & ~((1ULL << 54) - 1)) | _pdep_u64(mA >> 18, LOW_BITS) | (_pdep_u64(mB >> 18, LOW_BITS) << 1),
                   _pdep_u64(mA & ((1ULL << 18) - 1), LOW_BITS) | (_pdep_u64(mB & ((1ULL << 18) - 1), LOW_BITS) << 1));
            if (fat) {
                x.setFatXY(fatXY / 5, fatXY % 5);
            }
            return x;
        }

        FORCEINLINE u32 lookup(const B1B2& s) const {
            return dist[indexOfAbstract(abstractOf(s))];
        }
    };

    GroupPdb buildGroupPdb(const B1B2& target, const std::vector<u32>& groupA, const std::vector<u32>& groupB,
                           const u32 threads) {
        Timer timer;
        GroupPdb p;
        p.fat = target.getFatBool();
        for (const u32 c : groupA) p.classOf[c] = 1;
        for (const u32 c : groupB) p.classOf[c] = 2;
        p.name = "{";
        for (const u32 c : groupA) p.name += std::to_string(c);
        p.name += "}";
        if (!groupB.empty()) {
            p.name += "+{";
            for (const u32 c : groupB) p.name += std::to_string(c);
            p.name += "}";
        }
        const B1B2 goal = p.abstractOf(target);
        const u64 mA = (my_pext_u64(goal.b1, LOW_BITS) << 18) | my_pext_u64(goal.b2, LOW_BITS);
        const u64 mB = (my_pext_u64(goal.b1 >> 1, LOW_BITS) << 18) | my_pext_u64(goal.b2 >> 1, LOW_BITS);
        p.a = static_cast<u32>(my_popcount(mA));
        p.b = static_cast<u32>(my_popcount(mB));
        p.bStates = BINOM[36 - p.a][p.b];
        p.entries = BINOM[36][p.a] * p.bStates * (p.fat ? 25 : 1);
        p.dist.assign(p.entries, 255);
        p.dist[p.indexOfAbstract(goal)] = 0;
        for (u32 d = 0;; ++d) {
            std::atomic<u64> added{0};
            constexpr u64 CHUNK = 1ULL << 16;
            parallelFor(threads, (p.entries + CHUNK - 1) / CHUNK, [&](u32, const std::size_t ch) {
                B1B2 children[60];
                u64 hashes[60];
                const u64 end = std::min<u64>(p.entries, (ch + 1) * CHUNK);
                u64 local = 0;
                for (u64 i = ch * CHUNK; i < end; ++i) {
                    if (p.dist[i] != d) {
                        continue;
                    }
                    const B1B2 x = p.boardOfIndex(i, goal);
                    const std::size_t n = p.fat
                            ? frontier_recovery_detail::emitNoneChildren<true>(x, children, hashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(x, children, hashes);
                    for (std::size_t k = 0; k < n; ++k) {
                        u8* slot = &p.dist[p.indexOfAbstract(children[k])];
                        u8 expected = 255;
                        if (*slot == 255 && __atomic_compare_exchange_n(slot, &expected, static_cast<u8>(d + 1), false,
                                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            ++local;
                        }
                    }
                }
                added.fetch_add(local);
            });
            if (added.load() == 0) {
                break;
            }
            p.maxDist = d + 1;
        }
        p.buildSeconds = timer.getSeconds();
        return p;
    }

    int runPdb2(const std::string& puzzle, const u32 seedDepth, const u32 tableDepth, const u32 samples,
                const u32 threads, const u64 maxEntries) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        const bool fat = start.getFatBool();
        const int T = static_cast<int>(tableDepth);
        tcout << puzzle << (fat ? " (2x2)" : "") << " <" << seedDepth << ",3," << tableDepth << ">, threads " << threads
              << ", tables up to " << maxEntries / 1000000 << " M entries\n";
        u32 counts[8] = {};
        for (u64 y = 0; y < 6; ++y) {
            for (u64 x = 0; x < 6; ++x) {
                ++counts[goal.getColor(x, y)];
            }
        }
        std::vector<u32> colours;
        for (u32 c = 0; c < 8; ++c) {
            if (counts[c] != 0) {
                colours.push_back(c);
                tcout << "  colour " << c << ": " << counts[c] << " cells\n";
            }
        }
        const u64 mul = fat ? 25 : 1;
        // Candidates: every merged pair / triple (one class), every pair told
        // apart (two classes), that fits; keep the 6 largest.
        struct Cand { std::vector<u32> A, B; u64 entries; };
        std::vector<Cand> cands;
        const std::size_t nc = colours.size();
        for (std::size_t i = 0; i < nc; ++i) {
            for (std::size_t j = i + 1; j < nc; ++j) {
                const u32 a2 = counts[colours[i]] + counts[colours[j]];
                if (BINOM[36][a2] * mul <= maxEntries) {
                    cands.push_back({{colours[i], colours[j]}, {}, BINOM[36][a2] * mul});
                }
                for (std::size_t k = j + 1; k < nc; ++k) {
                    const u32 a3 = a2 + counts[colours[k]];
                    if (BINOM[36][a3] * mul <= maxEntries) {
                        cands.push_back({{colours[i], colours[j], colours[k]}, {}, BINOM[36][a3] * mul});
                    }
                }
                for (const auto& [x, y] : {std::pair{i, j}, std::pair{j, i}}) {
                    const u32 ca = counts[colours[x]], cb = counts[colours[y]];
                    const u64 e = BINOM[36][ca] * BINOM[36 - ca][cb] * mul;
                    if (e <= maxEntries && x < y) {
                        cands.push_back({{colours[x]}, {colours[y]}, e});
                    }
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& p, const Cand& q) { return p.entries > q.entries; });
        if (cands.size() > 3) {
            cands.resize(3);
        }
        std::vector<GroupPdb> pdbs;
        for (const Cand& c : cands) {
            pdbs.push_back(buildGroupPdb(goal.asB1B2(), c.A, c.B, threads));
            const GroupPdb& p = pdbs.back();
            tcout << "  pdb " << p.name << ": " << p.a << "+" << p.b << " cells, " << p.entries / 1000000.0
                  << " M entries, max distance " << p.maxDist << ", h(start) = " << p.lookup(start.asB1B2())
                  << ", built in " << p.buildSeconds << "s\n";
        }
        LaneTables mix;
        buildLaneTables(goal.asB1B2(), mix);
        tcout << "  colour-mix h(start) = " << colour_mix::bound(start.asB1B2(), mix) << "\n";

        JVec<B1B2> seeds;
        buildFrontier(start, seedDepth, seeds);
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / samples);
        const std::size_t np = pdbs.size();
        std::vector<std::atomic<u64>> lSkip(np + 2), nSkip(np + 2);   // per pdb, then colour-mix, then all
        std::atomic<u64> leaves{0}, nodes{0};
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            streamFrom(seeds[k * step], 2, [&](const B1B2& node) {
                nodes.fetch_add(1, std::memory_order_relaxed);
                int best = colour_mix::bound(node, mix);
                nSkip[np].fetch_add(best > T + 1, std::memory_order_relaxed);
                for (std::size_t q = 0; q < np; ++q) {
                    const int h = static_cast<int>(pdbs[q].lookup(node));
                    nSkip[q].fetch_add(h > T + 1, std::memory_order_relaxed);
                    best = std::max(best, h);
                }
                nSkip[np + 1].fetch_add(best > T + 1, std::memory_order_relaxed);
                streamFrom(node, 1, [&](const B1B2& leaf) {
                    leaves.fetch_add(1, std::memory_order_relaxed);
                    int lb = colour_mix::bound(leaf, mix);
                    lSkip[np].fetch_add(lb > T, std::memory_order_relaxed);
                    for (std::size_t q = 0; q < np; ++q) {
                        const int h = static_cast<int>(pdbs[q].lookup(leaf));
                        lSkip[q].fetch_add(h > T, std::memory_order_relaxed);
                        lb = std::max(lb, h);
                    }
                    lSkip[np + 1].fetch_add(lb > T, std::memory_order_relaxed);
                });
            });
        });
        auto pct = [](const u64 a, const u64 n) { return 100.0 * static_cast<double>(a) / static_cast<double>(std::max<u64>(1, n)); };
        tcout << "  skip (leaves " << leaves.load() << " | whole last level, nodes " << nodes.load() << "):\n";
        for (std::size_t q = 0; q < np; ++q) {
            tcout << "    pdb " << pdbs[q].name << ": " << pct(lSkip[q].load(), leaves.load()) << "% | "
                  << pct(nSkip[q].load(), nodes.load()) << "%\n";
        }
        tcout << "    colour-mix: " << pct(lSkip[np].load(), leaves.load()) << "% | " << pct(nSkip[np].load(), nodes.load()) << "%\n";
        tcout << "    all combined: " << pct(lSkip[np + 1].load(), leaves.load()) << "% | "
              << pct(nSkip[np + 1].load(), nodes.load()) << "%\n";
        return 0;
    }

    // A deeper left stream (seed depth S + `streamDepth` moves) pruned at
    // every level: a node with r moves still to go below it is dropped when
    // h > T + r (h changes by at most 1 per move). Walks one move at a time
    // (1-move streams, so a little more than the canonical stream) and
    // compares states generated and lookups left against no pruning.
    int runDeep(const std::string& puzzle, const u32 seedDepth, const u32 tableDepth, const u32 streamDepth,
                const u32 samples, const u32 threads, const colour_mix::Kind kind) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        const int T = static_cast<int>(tableDepth);
        LaneTables mix;
        buildLaneTables(goal.asB1B2(), mix, kind);
        tcout << puzzle << (start.getFatBool() ? " (2x2)" : "") << " <" << seedDepth << "," << streamDepth << ","
              << tableDepth << "> = " << seedDepth + streamDepth + tableDepth << " moves, " << colour_mix::kindName(kind)
              << " bound pruned at every level (tables " << mix.bytes << " B)\n";
        JVec<B1B2> seeds;
        buildFrontier(start, seedDepth, seeds);
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / samples);
        constexpr u32 MAXD = 6;
        std::array<std::atomic<u64>, MAXD + 1> visited{}, pruned{}, full{};
        std::atomic<u64> lookups{0};
        // full[L]: nodes at level L with no pruning (a pruned node's subtree is
        // counted with the fan-out measured at that level).
        std::function<void(const B1B2&, u32)> walk = [&](const B1B2& node, const u32 level) {
            visited[level].fetch_add(1, std::memory_order_relaxed);
            const int h = colour_mix::bound(node, mix);
            const int left = static_cast<int>(streamDepth - level);
            if (level == streamDepth) {
                lookups.fetch_add(h <= T, std::memory_order_relaxed);
                return;
            }
            if (level > 0 && h > T + left) {
                pruned[level].fetch_add(1, std::memory_order_relaxed);
                return;
            }
            streamFrom(node, 1, [&](const B1B2& child) { walk(child, level + 1); });
        };
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) { walk(seeds[k * step], 0); });
        // Unpruned counts: expand without pruning on a smaller sample.
        std::array<std::atomic<u64>, MAXD + 1> plain{};
        std::function<void(const B1B2&, u32)> walkPlain = [&](const B1B2& node, const u32 level) {
            plain[level].fetch_add(1, std::memory_order_relaxed);
            if (level == streamDepth) {
                return;
            }
            streamFrom(node, 1, [&](const B1B2& child) { walkPlain(child, level + 1); });
        };
        const std::size_t plainStep = step * 8;
        parallelFor(threads, (seeds.size() + plainStep - 1) / plainStep, [&](u32, const std::size_t k) {
            walkPlain(seeds[k * plainStep], 0);
        });
        const double scale = static_cast<double>(visited[0].load()) / static_cast<double>(std::max<u64>(1, plain[0].load()));
        u64 genPruned = 0;
        double genPlain = 0.0;
        for (u32 L = 1; L <= streamDepth; ++L) {
            const double pl = static_cast<double>(plain[L].load()) * scale;
            genPruned += visited[L].load();
            genPlain += pl;
            tcout << "  level " << L << " (" << streamDepth - L << " to go): visited " << visited[L].load() << " of ~"
                  << static_cast<u64>(pl) << " (" << 100.0 * static_cast<double>(visited[L].load()) / std::max(1.0, pl)
                  << "%), pruned here " << pruned[L].load() << "\n";
        }
        const double leavesPlain = static_cast<double>(plain[streamDepth].load()) * scale;
        tcout << "  states generated: " << 100.0 * static_cast<double>(genPruned) / std::max(1.0, genPlain)
              << "% of unpruned; lookups: " << 100.0 * static_cast<double>(lookups.load()) / std::max(1.0, leavesPlain)
              << "% of unpruned leaves\n";
        return 0;
    }
}

int main(const int argc, char** argv) {
    tcout << std::unitbuf;
    if (argc >= 2 && std::string(argv[1]) == "--check") {
        return runCheck();
    }
    if (argc >= 3 && std::string(argv[1]) == "--deep") {
        u32 seedDepth = 5, tableDepth = 6, streamDepth = 4, samples = 200, threads = 4;
        colour_mix::Kind kind = colour_mix::Kind::Mix;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--kind") {
                kind = std::string(argv[i + 1]) == "rotation" ? colour_mix::Kind::Rotation : colour_mix::Kind::Mix;
                continue;
            }
            const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
            if (a == "--seed-depth") seedDepth = v;
            else if (a == "--table-depth") tableDepth = v;
            else if (a == "--stream") streamDepth = v;
            else if (a == "--samples") samples = v;
            else if (a == "--threads") threads = v;
        }
        return runDeep(argv[2], seedDepth, tableDepth, streamDepth, samples, threads, kind);
    }
    if (argc >= 3 && std::string(argv[1]) == "--pdb2") {
        u32 seedDepth = 5, tableDepth = 6, samples = 150, threads = 4;
        u64 maxEntries = 300'000'000ULL;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            const u64 v = std::stoull(argv[i + 1]);
            if (a == "--seed-depth") seedDepth = static_cast<u32>(v);
            else if (a == "--table-depth") tableDepth = static_cast<u32>(v);
            else if (a == "--samples") samples = static_cast<u32>(v);
            else if (a == "--threads") threads = static_cast<u32>(v);
            else if (a == "--max-mentries") maxEntries = v * 1000000ULL;
        }
        return runPdb2(argv[2], seedDepth, tableDepth, samples, threads, maxEntries);
    }
    if (argc >= 2 && (std::string(argv[1]) == "--pdb" || std::string(argv[1]) == "--pdb-check")) {
        u32 seedDepth = 5, tableDepth = 6, samples = 300, threads = 4, kMax = 8;
        const bool check = std::string(argv[1]) == "--pdb-check";
        const int first = check ? 2 : 3;
        for (int i = first; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
            if (a == "--seed-depth") seedDepth = v;
            else if (a == "--table-depth") tableDepth = v;
            else if (a == "--samples") samples = v;
            else if (a == "--threads") threads = v;
            else if (a == "--kmax") kMax = v;
        }
        return check ? runPdbCheck(threads, kMax) : runPdb(argv[2], seedDepth, tableDepth, samples, threads, kMax);
    }
    if (argc >= 3 && std::string(argv[1]) == "--colors") {
        for (int i = 2; i < argc; ++i) {
            const BoardPair* pair = BoardLookup::getBoardPair(argv[i]);
            if (pair == nullptr) {
                continue;
            }
            const Board s = pair->getStartState();
            const Board g = pair->getEndState();
            u32 used = 0;
            for (u64 y = 0; y < 6; ++y) {
                for (u64 x = 0; x < 6; ++x) {
                    used |= 1U << s.getColor(x, y);
                    used |= 1U << g.getColor(x, y);
                }
            }
            tcout << argv[i] << (s.getFatBool() ? " (2x2)" : "") << ": colorCount " << s.getColorCount()
                  << ", codes used 0b" << std::bitset<8>(used).to_string() << "\n";
        }
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        return runBench(argv[2], 5, std::max(1U, std::thread::hardware_concurrency()));
    }
    if (argc < 2) {
        tcout << "usage: experimentBoundSample --check | <puzzle> --seed-depth S --table-depth T [--samples N] [--threads N]\n";
        return 1;
    }
    u32 seedDepth = 4, tableDepth = 6, samples = 4000;
    u32 threads = std::max(1U, std::thread::hardware_concurrency());
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
        if (a == "--seed-depth") seedDepth = v;
        else if (a == "--table-depth") tableDepth = v;
        else if (a == "--samples") samples = v;
        else if (a == "--threads") threads = v;
    }
    return runSample(argv[1], seedDepth, tableDepth, samples, threads);
}
