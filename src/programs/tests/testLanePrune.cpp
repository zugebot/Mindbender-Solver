// src/programs/tests/testLanePrune.cpp
//
// bounds::LanePrune (the cheap per-lane prune: the moved lane's half shared by all
// five siblings, the other half read lane by lane with an early exit) against
// bounds::LanePruneRef (bound() of every node):
//   1. identical masks for every lane's five shifts of random boards, every limit
//      0-16 and BELOW 0-4, rotation and colour-mix tables, on several normal puzzles;
//   2. identical pruned 4- and 5-move streams from sampled seeds (same leaves in the
//      same order: an order-sensitive checksum of their hashes), and LanePruneLeaves
//      emitting exactly the reference stream's leaves with h <= limit;
//   3. --bench: interleaved timing of 5-move pruned streams with each hook, each into a
//      sink that does the "both" probe mode's per-leaf work (mixKey, then the bound).
//
// Usage: testLanePrune                       (tests 1 and 2, a few seconds, 1 thread)
//        testLanePrune --bench <puzzle> [--seeds N] [--threads N] [--limit L] [--rounds R]

#include "src/chuzzles/include.hpp"
#include "src/solver/bounds/parent_prune.hpp"
#include "src/utils/parallel_for.hpp"
#include "src/utils/timer.hpp"

#include <atomic>
#include <random>
#include <string>
#include <vector>

namespace {

    B1B2 randomBoard(const Board& start, std::mt19937_64& rng, const int moves) {
        Board b = start;
        for (int i = 0; i < moves; ++i) {
            // rows are actions 0-29; columns follow the gap (as in frontier_builder.hpp)
            const u64 r = rng() % (NORMAL_ROW_MOVE_COUNT + NORMAL_COL_MOVE_COUNT);
            const u64 act = r < NORMAL_ROW_MOVE_COUNT ? r
                                                      : NORMAL_MOVE_GAP_BEGIN + NORMAL_MOVE_GAP_COUNT + (r - NORMAL_ROW_MOVE_COUNT);
            allActStructList[act].action(b);
        }
        return b.asB1B2();
    }

    template<int BELOW>
    u64 compareMasks(const B1B2 (&nodes)[5], const bool colLane, u64& checked) {
        u64 bad = 0;
        for (int limit = 0; limit <= 16; ++limit) {
            bounds::g_laneLimit = limit;
            const u32 fast = bounds::LanePrune::skipMask<BELOW>(nodes, colLane);
            const u32 ref = bounds::LanePruneRef::skipMask<BELOW>(nodes, colLane);
            bad += fast != ref;
            ++checked;
        }
        return bad;
    }

    struct ChecksumSink {
        u64 sum = 0, count = 0;
        void operator()(JVec<B1B2>&, JVec<u64>& hashes, const u32 n) {
            for (u32 i = 0; i < n; ++i) {
                sum = (sum ^ hashes[i]) * 0x100000001B3ULL;
                ++count;
            }
        }
    };

    // Like ProbeSinkT<true> per leaf: mix the hash into a key, drop leaves with h > limit.
    struct LeafBoundSink {
        u64 sum = 0, count = 0, seen = 0;
        void operator()(JVec<B1B2>& states, JVec<u64>& hashes, const u32 n) {
            seen += n;
            for (u32 i = 0; i < n; ++i) {
                const u64 key = hashes[i] * 0x9E3779B97F4A7C15ULL;
                if (lane_bound::bound(states[i], *bounds::g_laneTables) > bounds::g_laneLimit) {
                    continue;
                }
                sum = (sum ^ hashes[i] ^ (key >> 63)) * 0x100000001B3ULL;
                ++count;
            }
        }
    };

    template<int DEPTH, typename PRUNE, typename SINK = ChecksumSink>
    SINK streamFrom(const B1B2& seed) {
        SINK sink;
        PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, DEPTH, PRUNE>(makeBoardFromState(seed), sink, 4096);
        return sink;
    }

    int runTests() {
        const char* puzzles[] = {"15-1", "18-3", "16-2", "20-5", "9-3", "3-3", "12-1"};
        std::mt19937_64 rng(12345);
        u64 checked = 0, bad = 0, streams = 0, streamBad = 0, leaves = 0;
        for (const char* name : puzzles) {
            const BoardPair* pair = BoardLookup::getBoardPair(name);
            if (pair == nullptr || pair->getStartState().getFatBool()) {
                continue;
            }
            const Board start = pair->getStartState();
            for (const lane_bound::Kind kind : {lane_bound::Kind::Rotation, lane_bound::Kind::Mix}) {
                lane_bound::LaneTables tabs;
                lane_bound::buildLaneTables(pair->getEndState().asB1B2(), tabs, kind);
                bounds::g_laneTables = &tabs;
                // 1. masks
                for (int b = 0; b < 400; ++b) {
                    const B1B2 board = randomBoard(start, rng, static_cast<int>(rng() % 14));
                    for (u32 lane = 0; lane < 12; ++lane) {
                        B1B2 shifted[5];
                        const bool col = lane >= 6;
                        if (col) {
                            lane_rotations::colShifts(board, lane - 6, shifted);
                        } else {
                            lane_rotations::rowShifts(board, lane, shifted);
                        }
                        bad += compareMasks<0>(shifted, col, checked);
                        bad += compareMasks<1>(shifted, col, checked);
                        bad += compareMasks<2>(shifted, col, checked);
                        bad += compareMasks<3>(shifted, col, checked);
                        bad += compareMasks<4>(shifted, col, checked);
                    }
                }
                // 2. whole pruned streams (limit = a typical table depth)
                for (const int limit : {5, 6, 7}) {
                    bounds::g_laneLimit = limit;
                    for (int s = 0; s < 6; ++s) {
                        const B1B2 seed = randomBoard(start, rng, 5);
                        const ChecksumSink a4 = streamFrom<4, bounds::LanePrune>(seed);
                        const ChecksumSink b4 = streamFrom<4, bounds::LanePruneRef>(seed);
                        const ChecksumSink a5 = streamFrom<5, bounds::LanePrune>(seed);
                        const ChecksumSink b5 = streamFrom<5, bounds::LanePruneRef>(seed);
                        // leaf pruning in the stream: the same leaves the reference keeps after a leaf check
                        const auto l4 = streamFrom<4, bounds::LanePruneLeaves, LeafBoundSink>(seed);
                        const auto r4 = streamFrom<4, bounds::LanePruneRef, LeafBoundSink>(seed);
                        const auto l5 = streamFrom<5, bounds::LanePruneLeaves, LeafBoundSink>(seed);
                        const auto r5 = streamFrom<5, bounds::LanePruneRef, LeafBoundSink>(seed);
                        streams += 4;
                        streamBad += (a4.sum != b4.sum || a4.count != b4.count) + (a5.sum != b5.sum || a5.count != b5.count);
                        streamBad += (l4.sum != r4.sum || l4.count != r4.count || l4.seen != l4.count)
                                     + (l5.sum != r5.sum || l5.count != r5.count || l5.seen != l5.count);
                        leaves += a4.count + a5.count;
                    }
                }
            }
            tcout << name << ": masks so far " << checked << " checked, " << bad << " different; streams " << streams
                  << ", " << streamBad << " different\n";
        }
        const bool ok = bad == 0 && streamBad == 0 && checked > 0 && streams > 0;
        tcout << (ok ? "PASS" : "FAIL") << ": " << checked << " masks (" << bad << " different), " << streams
              << " pruned streams with " << leaves << " leaves (" << streamBad << " different)\n";
        return ok ? 0 : 1;
    }

    int runBench(const std::string& puzzle, const u32 seedCount, const u32 threads, const int limit, const u32 rounds) {
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr || pair->getStartState().getFatBool()) {
            tcout << "need a normal-board puzzle\n";
            return 1;
        }
        lane_bound::LaneTables tabs;
        lane_bound::buildLaneTables(pair->getEndState().asB1B2(), tabs, lane_bound::Kind::Rotation);
        bounds::g_laneTables = &tabs;
        bounds::g_laneLimit = limit;
        std::mt19937_64 rng(777);
        std::vector<B1B2> seeds(seedCount);
        for (auto& s : seeds) {
            s = randomBoard(pair->getStartState(), rng, 5);
        }
        tcout << puzzle << ": " << seedCount << " 5-move random-walk seeds, 5-move pruned streams, limit " << limit
              << ", " << threads << " threads, " << rounds << " rounds (alternating order)\n";
        // 0: LanePruneRef (the old "both"), 1: LanePrune, 2: LanePruneLeaves (the new "both")
        const char* names[3] = {"LanePruneRef   ", "LanePrune      ", "LanePruneLeaves"};
        double secs[3] = {0, 0, 0};
        u64 sums[3] = {0, 0, 0};
        for (u32 r = 0; r < rounds; ++r) {
            for (int j = 0; j < 3; ++j) {
                const int which = static_cast<int>((j + r) % 3);   // rotating order
                std::atomic<u64> sum{0};
                Timer timer;
                parallelFor(threads, seeds.size(), [&](u32, const std::size_t i) {
                    const LeafBoundSink sink = which == 0 ? streamFrom<5, bounds::LanePruneRef, LeafBoundSink>(seeds[i])
                                             : which == 1 ? streamFrom<5, bounds::LanePrune, LeafBoundSink>(seeds[i])
                                                          : streamFrom<5, bounds::LanePruneLeaves, LeafBoundSink>(seeds[i]);
                    sum.fetch_add(sink.sum, std::memory_order_relaxed);
                });
                const double sec = timer.getSeconds();
                secs[which] += sec;
                sums[which] = sum.load();
                tcout << "  round " << r + 1 << " " << names[which] << ": " << sec << " s\n";
            }
        }
        const bool same = sums[0] == sums[1] && sums[1] == sums[2];
        tcout << "vs LanePruneRef: LanePrune " << secs[0] / secs[1] << "x, LanePruneLeaves " << secs[0] / secs[2]
              << "x (" << secs[0] << " / " << secs[1] << " / " << secs[2] << " s); kept leaves "
              << (same ? "identical" : "DIFFERENT") << "\n";
        return same ? 0 : 1;
    }

} // namespace

int main(const int argc, char** argv) {
    tcout << std::unitbuf;
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        u32 seeds = 400, threads = 2, rounds = 4;
        int limit = 6;
        for (int i = 3; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
            if (a == "--seeds") seeds = v;
            else if (a == "--threads") threads = v;
            else if (a == "--limit") limit = static_cast<int>(v);
            else if (a == "--rounds") rounds = v;
        }
        return runBench(argv[2], seeds, threads, limit, rounds);
    }
    return runTests();
}
