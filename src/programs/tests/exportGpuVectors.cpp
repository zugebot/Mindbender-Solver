// src/programs/tests/exportGpuVectors.cpp
//
// Reference data for the GPU left stream (src/gpu): for a few puzzles, a strided
// sample of left seeds (with each seed's last move, as the solver streams them,
// or with none), streamed on the CPU with the solver's own PermStream code,
// unpruned and with the rotation lane bound at every level (LanePruneLeaves,
// the "both" mode). Per seed it records the leaf count and the wrapping sum and
// xor of the leaf keys (mixKey(StateHash(leaf))). test_gpu_stream.exe runs the
// same seeds on the GPU and compares seed by seed.
//
// Usage: exportGpuVectors <out.bin> [--scale S]   (S multiplies the seed counts; default 1)
//        exportGpuVectors <out.bin> --bench <puzzle> <seedDepth> <depth> <limit>
//   --bench writes every seed of one split with its last move and no CPU stream
//   (checksums 0, CPU seconds -1): test_gpu_stream --bench times the GPU on it.
// Single-threaded and small on purpose: it must not load the CPU.
//
// File: "MBGV0001", u32 cases, then per case: char name[16], u64 goalB1, goalB2,
// i32 limit (-1 = unpruned), i32 depth, i32 seedDepth, i32 lastMoves, u64 n, then
// n x u64 b1, n x u64 b2, n x u8 prev, n x u64 count, n x u64 sum, n x u64 xor,
// and f64 CPU seconds. Then (not with --bench) the goal-side child check: "CHLD", u64 n,
// n x u64 b1, n x u64 b2, n x u8 prev (10-3's goal frontier at depth 4 with last moves),
// 256 x u64 raw child counts per partition, u32 p0, u32 p1, u64 m, m x u64 sorted unique
// child keys in partitions [p0, p1) (perm_stream_detail::forEachFirstMove, as no_table_d6).

#include "src/solver/ssd/common.hpp"
#include "src/solver/bounds/parent_prune.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

    struct ChecksumSink {
        u64 count = 0, sum = 0, xr = 0;
        void operator()(JVec<B1B2>&, JVec<u64>& hashes, const u32 n) {
            for (u32 i = 0; i < n; ++i) {
                const u64 key = ssd::mixKey(hashes[i]);
                ++count;
                sum += key;
                xr ^= key;
            }
        }
    };

    template<i32 D>
    void streamOne(const Board& b, const u8 prev, const bool pruned, ChecksumSink& sink) {
        if (pruned) {
            PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, D, bounds::LanePruneLeaves>(b, sink, 4096, prev);
        } else {
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, D>(b, sink, 4096, prev);
        }
    }

    void streamSeed(const int depth, const Board& b, const u8 prev, const bool pruned, ChecksumSink& sink) {
        switch (depth) {
            case 1: streamOne<1>(b, prev, pruned, sink); break;
            case 2: streamOne<2>(b, prev, pruned, sink); break;
            case 3: streamOne<3>(b, prev, pruned, sink); break;
            case 4: streamOne<4>(b, prev, pruned, sink); break;
            case 5: streamOne<5>(b, prev, pruned, sink); break;
            default: throw std::runtime_error("depth 1..5");
        }
    }

    struct Case {
        const char* puzzle;
        int seedDepth;
        bool lastMoves;
        int depth;
        int limit;      // -1: unpruned
        u64 seeds;      // sampled (strided over the layer); 0 = all
    };

    template<typename T>
    void put(std::FILE* f, const T& v) {
        std::fwrite(&v, sizeof(T), 1, f);
    }

}  // namespace

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: exportGpuVectors <out.bin> [--scale S]\n");
        return 2;
    }
    double scale = 1.0;
    bool bench = false;
    Case benchCase{};
    std::string benchPuzzle;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--scale" && i + 1 < argc) {
            scale = std::stod(argv[i + 1]);
        }
        if (std::string(argv[i]) == "--bench" && i + 4 < argc) {
            bench = true;
            benchPuzzle = argv[i + 1];
            benchCase = {nullptr, std::stoi(argv[i + 2]), true, std::stoi(argv[i + 3]), std::stoi(argv[i + 4]), 0};
        }
    }
    // 1-1: 2 colours, many repeating lanes; 9-3 / 10-3: 3 colours; 18-3: 8 colours;
    // 16-2 as at its 15-move split <4,5,6>. Limits make seed depth + stream + limit = the solved length.
    std::vector<Case> cases = {
            {"1-1", 2, true, 3, -1, 0},
            {"10-3", 3, true, 3, -1, 300},
            {"10-3", 3, false, 2, -1, 300},
            {"9-3", 3, true, 4, 6, 2000},
            {"10-3", 3, true, 4, 6, 2000},
            {"10-3", 3, true, 5, 5, 500},
            {"18-3", 4, true, 4, 6, 500},
            {"18-3", 4, true, 5, 6, 200},
            {"16-2", 4, true, 5, 6, 200},
    };

    if (bench) {
        benchCase.puzzle = benchPuzzle.c_str();
        cases = {benchCase};
    }
    std::FILE* f = std::fopen(argv[1], "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }
    std::fwrite("MBGV0001", 1, 8, f);
    put(f, static_cast<u32>(cases.size()));

    static lane_bound::LaneTables tables;
    for (const Case& c : cases) {
        const BoardPair* pair = BoardLookup::getBoardPair(c.puzzle);
        if (pair == nullptr) {
            std::fprintf(stderr, "no puzzle %s\n", c.puzzle);
            return 1;
        }
        const Board root = pair->getStartState();
        const B1B2 goal = pair->getEndState().asB1B2();
        JVec<B1B2> layer;
        ssd::buildSeeds(root, static_cast<u32>(c.seedDepth), layer);
        std::vector<u8> moves(layer.size(), perm_stream_detail::NO_PREV_MOVE);
        if (c.lastMoves) {
            ssd::buildLastMoves(root, static_cast<u32>(c.seedDepth), layer, moves, bench ? 4 : 1);
            ssd::dropUncoded(layer, moves);
        }
        const u64 want = c.seeds == 0 ? layer.size() : std::min<u64>(layer.size(), static_cast<u64>(c.seeds * scale));
        const u64 stride = std::max<u64>(1, layer.size() / std::max<u64>(1, want));
        std::vector<u64> b1, b2, cnt, sum, xr;
        std::vector<u8> prev;
        for (u64 i = 0; i < layer.size() && b1.size() < want; i += stride) {
            b1.push_back(layer[i].b1);
            b2.push_back(layer[i].b2);
            prev.push_back(moves[i]);
        }
        const bool pruned = c.limit >= 0;
        if (pruned) {
            lane_bound::buildLaneTables(goal, tables, lane_bound::Kind::Rotation);
            bounds::g_laneTables = &tables;
            bounds::g_laneLimit = c.limit;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (u64 s = 0; s < b1.size() && !bench; ++s) {
            ChecksumSink sink;
            streamSeed(c.depth, makeBoardFromState(B1B2(b1[s], b2[s])), prev[s], pruned, sink);
            cnt.push_back(sink.count);
            sum.push_back(sink.sum);
            xr.push_back(sink.xr);
        }
        const double secs = bench ? -1.0 : std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cnt.resize(b1.size(), 0);
        sum.resize(b1.size(), 0);
        xr.resize(b1.size(), 0);
        u64 leaves = 0;
        for (const u64 v : cnt) {
            leaves += v;
        }
        std::printf("%-5s seeds depth %d (%s), stream %d, limit %2d: %6llu seeds, %12llu leaves, CPU %.2f s\n", c.puzzle,
                    c.seedDepth, c.lastMoves ? "last moves" : "no prev", c.depth, c.limit,
                    static_cast<unsigned long long>(b1.size()), static_cast<unsigned long long>(leaves), secs);
        char name[16] = {};
        std::snprintf(name, sizeof(name), "%s", c.puzzle);
        std::fwrite(name, 1, sizeof(name), f);
        put(f, goal.b1);
        put(f, goal.b2);
        put(f, static_cast<i32>(c.limit));
        put(f, static_cast<i32>(c.depth));
        put(f, static_cast<i32>(c.seedDepth));
        put(f, static_cast<i32>(c.lastMoves ? 1 : 0));
        put(f, static_cast<u64>(b1.size()));
        std::fwrite(b1.data(), 8, b1.size(), f);
        std::fwrite(b2.data(), 8, b2.size(), f);
        std::fwrite(prev.data(), 1, prev.size(), f);
        std::fwrite(cnt.data(), 8, cnt.size(), f);
        std::fwrite(sum.data(), 8, sum.size(), f);
        std::fwrite(xr.data(), 8, xr.size(), f);
        put(f, secs);
    }
    if (!bench) {
        const BoardPair* pair = BoardLookup::getBoardPair("10-3");
        const Board goal = pair->getEndState();
        JVec<B1B2> layer;
        ssd::buildSeeds(goal, 4, layer);
        std::vector<u8> moves;
        ssd::buildLastMoves(goal, 4, layer, moves, 1);
        ssd::dropUncoded(layer, moves);
        std::vector<u64> counts(256, 0), keys;
        constexpr u32 P0 = 40, P1 = 72;
        for (std::size_t i = 0; i < layer.size(); ++i) {
            perm_stream_detail::forEachFirstMove(layer[i], moves[i], [&](u8, const B1B2& child) {
                const u64 key = ssd::mixKey(StateHash::computeHash(child));
                ++counts[key >> 56];
                if ((key >> 56) >= P0 && (key >> 56) < P1) {
                    keys.push_back(key);
                }
            });
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::fwrite("CHLD", 1, 4, f);
        put(f, static_cast<u64>(layer.size()));
        for (std::size_t i = 0; i < layer.size(); ++i) put(f, layer[i].b1);
        for (std::size_t i = 0; i < layer.size(); ++i) put(f, layer[i].b2);
        std::fwrite(moves.data(), 1, moves.size(), f);
        std::fwrite(counts.data(), 8, 256, f);
        put(f, P0);
        put(f, P1);
        put(f, static_cast<u64>(keys.size()));
        std::fwrite(keys.data(), 8, keys.size(), f);
        u64 raw = 0;
        for (const u64 c : counts) raw += c;
        std::printf("goal side: 10-3 depth-4 frontier %llu states, %llu children, %llu unique keys in [%u,%u)\n",
                    static_cast<unsigned long long>(layer.size()), static_cast<unsigned long long>(raw),
                    static_cast<unsigned long long>(keys.size()), P0, P1);
    }
    std::fclose(f);
    return 0;
}
