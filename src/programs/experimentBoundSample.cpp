// src/programs/experimentBoundSample.cpp
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

#include "src/chuzzles/include.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/utils/intrinsics/pext_u64.hpp"

#include <array>
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

    // Colour multiset of each row and column, as 8 counts of 4 bits packed in a u32.
    struct LaneMix {
        u32 rows[6];
        u32 cols[6];
    };

    LaneMix laneMix(const B1B2& b) {
        LaneMix m{};
        for (u32 y = 0; y < 6; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const u32 c = b.getColor(static_cast<u64>(x), static_cast<u64>(y));
                m.rows[y] += 1U << (4 * c);
                m.cols[x] += 1U << (4 * c);
            }
        }
        return m;
    }

    // Cells (of 6) that must change for one multiset to become the other.
    int mixDistance(const u32 a, const u32 b) {
        int common = 0;
        for (u32 c = 0; c < 8; ++c) {
            common += static_cast<int>(std::min((a >> (4 * c)) & 15U, (b >> (4 * c)) & 15U));
        }
        return 6 - common;
    }

    int colourMixBound(const B1B2& state, const LaneMix& target, const bool fat) {
        const LaneMix s = laneMix(state);
        int r = 0, c = 0;
        for (u32 i = 0; i < 6; ++i) {
            r = std::max(r, mixDistance(s.rows[i], target.rows[i]));
            c = std::max(c, mixDistance(s.cols[i], target.cols[i]));
        }
        return fat ? (r + 1) / 2 + (c + 1) / 2 : r + c;
    }

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

    // Column x within a half: cell x of each of its three rows (9 bits after pext).
    constexpr u64 colMask(const u32 x) {
        return (7ULL << (36 + 15 - 3 * x)) | (7ULL << (18 + 15 - 3 * x)) | (7ULL << (15 - 3 * x));
    }
    constexpr u64 COL_MASK[6] = {colMask(0), colMask(1), colMask(2), colMask(3), colMask(4), colMask(5)};

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

    // ------------------------------------------------------------------
    // Per-puzzle lane tables: no counting at all at search time.
    //
    // A lane's distance is 6 - sum over colours c of min(count_c, goal_c),
    // where goal_c is how often c appears in that goal lane. A lane's 6 cells
    // are two triples of 3 cells (9 bits each), so for the pair of triples
    // only each colour's count up to goal_c matters (more is never shared),
    // and colours absent from the goal lane don't matter at all. Per lane L:
    //   first[L][triple]  -> offset of L's table + id * ids_L   (u16)
    //   second[L][triple] -> id                                 (u8)
    //   dist[first + second] -> 1 << the lane's distance         (one flat u8 array)
    // where id numbers the triple's capped colour counts. A lane is 3 loads
    // and an add: no multiply or shift (pext already uses port 1). Distances
    // are one-hot, so the max over 6 lanes is 5 ORs and one bit scan. Rows come
    // straight from the board (two 9-bit fields per row); a column's two
    // triples are one pext of b1 and one of b2.
    // ------------------------------------------------------------------
    struct LaneTables {
        alignas(64) u16 first[12][512];     // lanes 0-5 rows, 6-11 columns
        alignas(64) u8 second[12][512];
        std::vector<u8> dist;
        u32 ids[12] = {};
        bool fat = false;
        u64 bytes = 0;
    };

    void buildLaneTables(const B1B2& goal, LaneTables& t) {
        t.fat = goal.getFatBool();
        u8 cells[12][6];   // goal lane L, cell k
        for (u32 y = 0; y < 6; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const u8 c = goal.getColor(static_cast<u64>(x), static_cast<u64>(y));
                cells[y][x] = c;
                cells[6 + x][y] = c;
            }
        }
        t.dist.clear();
        for (u32 lane = 0; lane < 12; ++lane) {
            int classOf[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            u32 goalCount[6] = {};
            u32 m = 0;
            for (u32 k = 0; k < 6; ++k) {
                if (classOf[cells[lane][k]] < 0) {
                    classOf[cells[lane][k]] = static_cast<int>(m++);
                }
                ++goalCount[classOf[cells[lane][k]]];
            }
            std::map<std::vector<u32>, u32> idOf;
            std::vector<std::vector<u32>> countsOf;
            u32 idOfTriple[512];
            for (u32 v = 0; v < 512; ++v) {
                std::vector<u32> counts(m, 0);
                for (u32 k = 0; k < 3; ++k) {
                    const int c = classOf[(v >> (3 * k)) & 7];
                    if (c >= 0) {
                        counts[c] = std::min(counts[c] + 1, goalCount[c]);
                    }
                }
                auto it = idOf.find(counts);
                if (it == idOf.end()) {
                    it = idOf.emplace(counts, static_cast<u32>(countsOf.size())).first;
                    countsOf.push_back(counts);
                }
                idOfTriple[v] = it->second;
            }
            const u32 ids = static_cast<u32>(countsOf.size());
            const std::size_t base = t.dist.size();
            if (base + static_cast<std::size_t>(ids) * ids > 65535) {
                throw std::runtime_error("lane tables exceed 16-bit offsets");
            }
            t.ids[lane] = ids;
            t.dist.resize(base + static_cast<std::size_t>(ids) * ids);
            for (u32 a = 0; a < ids; ++a) {
                for (u32 b = 0; b < ids; ++b) {
                    u32 common = 0;
                    for (u32 c = 0; c < m; ++c) {
                        common += std::min(countsOf[a][c] + countsOf[b][c], goalCount[c]);
                    }
                    t.dist[base + a * ids + b] = static_cast<u8>(1U << (6 - common));
                }
            }
            for (u32 v = 0; v < 512; ++v) {
                t.first[lane][v] = static_cast<u16>(base + idOfTriple[v] * ids);
                t.second[lane][v] = static_cast<u8>(idOfTriple[v]);
            }
        }
        t.bytes = sizeof(t.first) + sizeof(t.second) + t.dist.size();
    }

    FORCEINLINE int colourMixBoundTables(const B1B2& s, const LaneTables& t) {
        const u8* d = t.dist.data();
        // Unsigned 64-bit indices (no sign extension); each lane finished and
        // OR-ed in before the next starts, so few values are live at once.
        auto lane = [&](const u32 L, const u64 a9, const u64 b9) -> u32 {
            return d[static_cast<u64>(t.first[L][a9]) + static_cast<u64>(t.second[L][b9])];
        };
        u64 b1 = s.b1, b2 = s.b2;
        u32 rowBits = lane(0, (b1 >> 36) & 511U, (b1 >> 45) & 511U);
        rowBits |= lane(1, (b1 >> 18) & 511U, (b1 >> 27) & 511U);
        rowBits |= lane(2, b1 & 511U, (b1 >> 9) & 511U);
        rowBits |= lane(3, (b2 >> 36) & 511U, (b2 >> 45) & 511U);
        rowBits |= lane(4, (b2 >> 18) & 511U, (b2 >> 27) & 511U);
        rowBits |= lane(5, b2 & 511U, (b2 >> 9) & 511U);
        u32 colBits = 0;
        for (u32 x = 0; x < 6; ++x) {
            // Empty asm: b1/b2 "change" here, so GCC cannot hoist all twelve
            // pexts to the top (it did, and spilled them to the stack).
            asm volatile("" : "+r"(b1), "+r"(b2));
            colBits |= lane(6 + x, my_pext_u64(b1, COL_MASK[x]), my_pext_u64(b2, COL_MASK[x]));
        }
        const u32 r = 31U - static_cast<u32>(__builtin_clz(rowBits)), c = 31U - static_cast<u32>(__builtin_clz(colBits));
        return t.fat ? static_cast<int>((r + 1) / 2 + (c + 1) / 2) : static_cast<int>(r + c);
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
                    ++states;
                    if (hg > n - i || hs > i) {
                        if (violations < 20) {
                            tcout << "VIOLATION " << fn << " step " << i << ": h(goal)=" << hg << " vs " << n - i
                                  << ", h(start)=" << hs << " vs " << i << " in " << line << "\n";
                        }
                        ++violations;
                    }
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
        tcout << "CHECK: " << states << " states on known optimal paths, " << violations << " violations -> "
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
}

int main(const int argc, char** argv) {
    tcout << std::unitbuf;
    if (argc >= 2 && std::string(argv[1]) == "--check") {
        return runCheck();
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
