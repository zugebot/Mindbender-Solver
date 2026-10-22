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
//       Both kinds of lane tables (src/solver/bounds/lane_bound.hpp: colour mix
//       and rotation) vs their reference on 20 M real left states: identical
//       values, and ns/state on 1 and all threads.
//   experimentBoundSample --deep <puzzle> [--seed-depth S] [--table-depth T] [--stream N]
//                         [--samples N] [--threads N] [--kind mix|rotation] [--reverse 1]
//       A left stream pruned at every level: states generated and lookups left,
//       against no pruning.
//   experimentBoundSample --goal-side <puzzle> [--table-depth T] [--left-depth L] [--samples N] [--threads N]
//       The goal side of <S,T2,T> with S + T2 = L: how many goal-side depth-T states (from
//       sampled depth-5 states, streamed T - 5 moves) have h(start) <= L, i.e. could still be
//       meeting points, and how many depth-5 states have h(start) <= L + T - 5 (could still have
//       such descendants). Rotation and colour-mix bounds. Normal boards only.
//   experimentBoundSample --colors <puzzle>...
//       Colour codes used by each puzzle.
// Older research modes (SWAR bound, single-colour and prototype grouped pattern
// databases) are in old/src/programs/experiments/experimentBoundSample_research.cpp.

#include "src/chuzzles/include.hpp"
#include "src/utils/parallel_for.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/solver/bounds/lane_bound.hpp"

#include <array>
#include <functional>
#include <bitset>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {
    using LaneMix = lane_bound::LaneMixRef;
    using lane_bound::LaneTables;
    using lane_bound::buildLaneTables;

    inline LaneMix laneMix(const B1B2& b) { return lane_bound::laneMixRef(b); }
    inline int colourMixBound(const B1B2& s, const LaneMix& t, const bool fat) { return lane_bound::referenceBound(s, t, fat); }
    FORCEINLINE int colourMixBoundTables(const B1B2& s, const LaneTables& t) { return lane_bound::bound(s, t); }


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
                    const int rg = lane_bound::referenceBound(b.asB1B2(), goal.asB1B2(), lane_bound::Kind::Rotation);
                    const int rs = lane_bound::referenceBound(b.asB1B2(), start.asB1B2(), lane_bound::Kind::Rotation);
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

        auto timeIt = [&](const std::string& label, auto&& fn) {
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
        // All threads at once (shares the core with hyperthreads, like the search).
        auto allThreads = [&](const std::string& label, auto&& fn) {
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
            tcout << "  " << label << ", " << threads << " threads: " << total / t.getSeconds() / 1e6 << " M states/s ("
                  << t.getSeconds() * 1e9 * threads / total << " ns/state per thread)\n";
        };
        const double base = timeIt("loop only (read state)", [](const B1B2& s) { return (s.b1 ^ s.b2) & 1; });
        timeIt("one row move (R1 +1) for reference", [](const B1B2& s) {
            B1B2 b = s;
            allActStructList[0].action(b);
            return (b.b1 ^ b.b2) & 1;
        });

        u64 mismatches = 0;
        for (const lane_bound::Kind kind : {lane_bound::Kind::Mix, lane_bound::Kind::Rotation}) {
            const std::string name = lane_bound::kindName(kind);
            LaneTables tabs;
            buildLaneTables(goal.asB1B2(), tabs, kind);
            std::atomic<u64> bad{0};
            parallelFor(threads, (states.size() + 65535) / 65536, [&](u32, const std::size_t c) {
                for (std::size_t i = c * 65536; i < std::min(states.size(), (c + 1) * 65536); ++i) {
                    if (lane_bound::referenceBound(states[i], goal.asB1B2(), kind) != lane_bound::bound(states[i], tabs)) {
                        bad.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
            mismatches += bad.load();
            tcout << name << " lane tables: " << tabs.bytes << " bytes; vs reference: " << bad.load()
                  << " mismatches of " << states.size() << "\n";
            const double tt = timeIt(name + " lane-table bound", [&](const B1B2& s) { return lane_bound::bound(s, tabs); });
            tcout << "  net of the loop: " << tt - base << " ns/state\n";
            allThreads(name + " lane-table bound", [&](const B1B2& s) { return lane_bound::bound(s, tabs); });
        }
        return mismatches == 0 ? 0 : 1;
    }

    // A deeper left stream (seed depth S + `streamDepth` moves) pruned at
    // every level: a node with r moves still to go below it is dropped when
    // h > T + r (h changes by at most 1 per move). Walks one move at a time
    // (1-move streams, so a little more than the canonical stream) and
    // compares states generated and lookups left against no pruning.
    int runDeep(const std::string& puzzle, const u32 seedDepth, const u32 tableDepth, const u32 streamDepth,
                const u32 samples, const u32 threads, const lane_bound::Kind kind, const bool reverse = false) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << "\n";
            return 1;
        }
        // --reverse 1: search from the goal towards the start (the bound then measures to the start)
        const Board start = reverse ? pair->getEndState() : pair->getStartState();
        const Board goal = reverse ? pair->getStartState() : pair->getEndState();
        const int T = static_cast<int>(tableDepth);
        if (reverse) tcout << "reversed: left side from the goal, bound to the start\n";
        LaneTables mix;
        buildLaneTables(goal.asB1B2(), mix, kind);
        tcout << puzzle << (start.getFatBool() ? " (2x2)" : "") << " <" << seedDepth << "," << streamDepth << ","
              << tableDepth << "> = " << seedDepth + streamDepth + tableDepth << " moves, " << lane_bound::kindName(kind)
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
            const int h = lane_bound::bound(node, mix);
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

namespace {
    int runGoalSide(const std::string& puzzle, const u32 tableDepth, const u32 leftDepth, const u32 samples,
                    const u32 threads) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "unknown puzzle " << puzzle << "\n";
            return 1;
        }
        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        if (start.getFatBool()) {
            tcout << puzzle << " has a 2x2 piece: the halved bound prunes nothing there\n";
            return 1;
        }
        LaneTables rot, mix;
        buildLaneTables(start.asB1B2(), rot, lane_bound::Kind::Rotation);
        buildLaneTables(start.asB1B2(), mix, lane_bound::Kind::Mix);
        const int L = static_cast<int>(leftDepth);
        const u32 extra = tableDepth - 5;
        Timer timer;
        JVec<B1B2> depth5;
        buildFrontier(goal, 5, depth5);
        tcout << puzzle << ": goal-side depth-5 frontier " << depth5.size() << " (" << timer.getSeconds()
              << "s); h(start->goal) rotation " << lane_bound::bound(goal.asB1B2(), rot) << "\n";
        Histogram parentRot, parentMix, rot7, mix7, rot7Kept;
        const std::size_t step = std::max<std::size_t>(1, depth5.size() / samples);
        parallelFor(threads, (depth5.size() + step - 1) / step, [&](u32, const std::size_t k) {
            const B1B2& g = depth5[k * step];
            const int hr = lane_bound::bound(g, rot);
            parentRot.add(hr);
            parentMix.add(lane_bound::bound(g, mix));
            streamFrom(g, extra, [&](const B1B2& s) {
                const int r = lane_bound::bound(s, rot);
                rot7.add(r);
                mix7.add(lane_bound::bound(s, mix));
                if (hr <= L + static_cast<int>(extra)) rot7Kept.add(r);
            });
        });
        parentRot.print("  depth-5 states, h(start) rotation", L + static_cast<int>(extra));
        parentMix.print("  depth-5 states, h(start) colour mix", L + static_cast<int>(extra));
        rot7.print("  depth-" + std::to_string(tableDepth) + " states, h(start) rotation", L);
        mix7.print("  depth-" + std::to_string(tableDepth) + " states, h(start) colour mix", L);
        const double keep = static_cast<double>(rot7.total() - rot7.above(L)) / std::max<double>(1.0, static_cast<double>(rot7.total()));
        tcout << "  -> a goal side filtered by rotation h(start) <= " << L << " keeps " << 100.0 * keep
              << "% of depth-" << tableDepth << " states; generating only from depth-5 states with h <= "
              << L + static_cast<int>(extra) << " streams " << 100.0 * static_cast<double>(rot7Kept.total())
              << "/" << rot7.total() << " (" << 100.0 * static_cast<double>(rot7Kept.total()) / std::max<double>(1.0, static_cast<double>(rot7.total()))
              << "%) of them (" << timer.getSeconds() << "s)\n";
        return 0;
    }
}

int main(const int argc, char** argv) {
    tcout << std::unitbuf;
    if (argc >= 3 && std::string(argv[1]) == "--goal-side") {
        u32 tableDepth = 7, leftDepth = 9, samples = 20000, threads = 4;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
            if (a == "--table-depth") tableDepth = v;
            else if (a == "--left-depth") leftDepth = v;
            else if (a == "--samples") samples = v;
            else if (a == "--threads") threads = v;
        }
        return runGoalSide(argv[2], tableDepth, leftDepth, samples, threads);
    }
    if (argc >= 2 && std::string(argv[1]) == "--check") {
        return runCheck();
    }
    if (argc >= 3 && std::string(argv[1]) == "--deep") {
        u32 seedDepth = 5, tableDepth = 6, streamDepth = 4, samples = 200, threads = 4;
        bool reverse = false;
        lane_bound::Kind kind = lane_bound::Kind::Mix;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--kind") {
                kind = std::string(argv[i + 1]) == "rotation" ? lane_bound::Kind::Rotation : lane_bound::Kind::Mix;
                continue;
            }
            const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
            if (a == "--seed-depth") seedDepth = v;
            else if (a == "--table-depth") tableDepth = v;
            else if (a == "--stream") streamDepth = v;
            else if (a == "--samples") samples = v;
            else if (a == "--threads") threads = v;
            else if (a == "--reverse") reverse = v != 0;
        }
        return runDeep(argv[2], seedDepth, tableDepth, streamDepth, samples, threads, kind, reverse);
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
