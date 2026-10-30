#pragma once
// src/solver/ssd/common.hpp
//
// Shared pieces of the SSD meet-in-the-middle solver (experimentDiskDepth6):
// partitioning keys, the run's Config, keyed states, partition file helpers,
// left seeds and the run-wide globals. See src/solver/ssd/README.md.

#include "src/chuzzles/include.hpp"
#include "src/chuzzles/lane_rotations.hpp"
#include "src/chuzzles/perm_stream.hpp"
#include "src/utils/progress_reporter.hpp"
#include "src/utils/parallel_for.hpp"
#include "src/solver/bounds/lane_bound.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cpuid.h>
#include <cstring>
#include <sstream>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ssd {

    constexpr u32 PARTITION_BITS = 8;
    constexpr u32 PARTITION_COUNT = 1U << PARTITION_BITS;
    constexpr u32 PARTITION_SHIFT = 64U - PARTITION_BITS;
    constexpr std::size_t WRITE_BUFFER_STATES = 8192;

    // StateHash::computeHash keeps the board metadata (color count, fat
    // position) in its top bits, so partition on a fully mixed key instead.
    FORCEINLINE u64 mixKey(u64 x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    FORCEINLINE u64 keyOf(const B1B2& state) {
        return mixKey(StateHash::computeHash(state));
    }

    struct Config {
        std::string puzzle = "6-2";
        fs::path outRoot = "X:/mindbender_d6";
        u64 batchStates = 320'000'000ULL;
        u32 threads = 0;
        bool reverse = false;
        bool rebuild = false;
        std::string solutionsFile;
        u32 seedDepth = 3;
        bool recover = true;
        bool batched = false;
        u64 ramBytes = 0;   // 0: ~60% of free RAM
        bool largePages = true;
        u32 d7Sample = 0;
        u64 d7PassBudget = 375'000'000ULL;
        u32 tableDepth = 6;
        u32 d7Partitions = 256;   // depth 7: build/probe only key partitions
        u32 d7FirstPartition = 0; //   [first, first + count)
        std::string progressFile;
        double progressInterval = 5.0;
        bool ramBuild = false;        // --ram-build: in-RAM depth-6 build (see header)
        bool diskTable = false;       // --disk-table: build/use the on-disk depth-6 table
        bool calibrate = true;
        double calibrateSeconds = 20.0;
        int forcedGroup = 0;          // --probe-variant G:P
        int forcedPipeline = 0;
        int boundMode = -1;           // --bound: -1 auto, 0 off, 1 leaf, 2 both, 3 pdb, 4 pdbleaf,
                                      //          5 parent, 6 pdbparent (see header)
        int pdbMode = -1;             // --pdb: -1 auto, 0 off, 1 on
        u32 streamDepth = 3;          // --stream-depth 3|4: left stream moves after the seeds
        u32 shardIndex = 0, shardCount = 1;     // --shard k/N
        std::string middlesOut;                 // --middles-out FILE
        std::vector<std::string> recoverFrom;   // --recover-from FILE (repeatable)
        u64 pdbMaxEntries = 300'000'000ULL;
        double boundCalibrateSeconds = 300.0;
        bool goalFilter = true;       // --no-goal-filter: keep every depth-7 goal-side state
        u32 recoverStreamDepth = 5;   // --recover-stream-depth: per-middle recovery halves this deep are streamed
        bool d7CountsInRam = false;   // --d7-counts-ram: keep the depth-7 bucket counts in RAM
        bool seedLastMove = true;     // first streamed move follows the canonical-order rules (--no-seed-last-move: off)
        lane_bound::Kind boundKind = lane_bound::Kind::Rotation;   // --bound-kind mix|rotation   // max; the run uses 3% of its search, >= 20 s

        MUND u32 d7End() const { return std::min<u32>(PARTITION_COUNT, d7FirstPartition + d7Partitions); }
        MUND bool d7Subset() const { return tableDepth == 7 && (d7FirstPartition != 0 || d7End() != PARTITION_COUNT); }
    };

    struct KeyedState {
        u64 key;
        B1B2 state;

        bool operator<(const KeyedState& other) const {
            if (key != other.key) {
                return key < other.key;
            }
            return state < other.state;
        }
    };

    inline fs::path rawPath(const fs::path& dir, const u32 p) {
        char name[32];
        std::snprintf(name, sizeof(name), "raw_%03x.bin", p);
        return dir / name;
    }

    inline fs::path sortedPath(const fs::path& dir, const u32 p) {
        char name[32];
        std::snprintf(name, sizeof(name), "sorted_%03x.bin", p);
        return dir / name;
    }

    inline double gbOf(const u64 states) {
        return static_cast<double>(states) * sizeof(B1B2) / 1e9;
    }

    template<typename T>
    std::vector<T> readAll(const fs::path& path) {
        std::vector<T> out(fs::file_size(path) / sizeof(T));
        std::FILE* f = std::fopen(path.string().c_str(), "rb");
        if (f == nullptr || std::fread(out.data(), sizeof(T), out.size(), f) != out.size()) {
            throw std::runtime_error("failed to read " + path.string());
        }
        std::fclose(f);
        return out;
    }

    template<typename T>
    void writeAll(const fs::path& path, const T* data, const std::size_t count) {
        std::FILE* f = std::fopen(path.string().c_str(), "wb");
        if (f == nullptr || std::fwrite(data, sizeof(T), count, f) != count) {
            throw std::runtime_error("failed to write " + path.string());
        }
        std::fclose(f);
    }

    constexpr u32 LEFT_STREAM_DEPTH = 3;
    constexpr u32 PROBE_GROUP = 32;

    inline void buildSeeds(const Board& root, const u32 depth, JVec<B1B2>& seeds) {
        JVec<u64> hashes;
        switch (depth) {
            case 1: buildUniqueNoneDepthFrontierB1B2<1>(root, seeds, hashes, false); return;
            case 2: buildUniqueNoneDepthFrontierB1B2<2>(root, seeds, hashes, false); return;
            case 3: buildUniqueNoneDepthFrontierB1B2<3>(root, seeds, hashes, false); return;
            case 4: buildUniqueNoneDepthFrontierB1B2<4>(root, seeds, hashes, false); return;
            case 5: buildUniqueNoneDepthFrontierB1B2<5>(root, seeds, hashes, false); return;
            default: throw std::runtime_error("seed depth must be 1..5");
        }
    }

    // For every seed (normal boards), the last move of its lexicographically
    // smallest path from the root, moves ordered as stream_lanes_dfs visits them
    // (encoding: perm_stream_detail::decodePrevMove). Streaming a seed with that
    // move as the previous one (ASCENDING) applies the canonical-order rules to
    // its first streamed move too, and still reaches every board at depth
    // S + T2: the smallest path to such a board starts with the smallest path
    // to its seed, and none of its adjacent pairs is one the rules skip (each
    // rule skips a pair whose swap, or merge, is a smaller path to the same
    // board). Suggested by algorithm0r (GitHub issue 1): ~1.47x fewer left
    // states, since the first move otherwise has all 60 choices instead of the
    // ~41 the rules leave.
    // Layers 1..S-1 are rebuilt and coded: a state's code is the smallest
    // (predecessor's code, move) over its predecessors in the layer above, found
    // by undoing each move and binary searching that layer. The seeds (up to
    // 160 M at depth 5) are coded the other way round: every child of the coded
    // layer S-1 is looked up in a hash table of the seeds and lowers that seed's
    // code, so the work is 60 lookups per layer-(S-1) state, not per seed
    // (18-3 at depth 5: 29 s instead of 209 s of binary searches).
    // Returns how many seeds got no code (they keep NO_PREV_MOVE: every first
    // move allowed, which is always safe).
    inline u64 buildSeedLastMoves(const Board& root, const u32 depth, const JVec<B1B2>& seeds,
                                  std::vector<u8>& out, const u32 threads) {
        struct Coded {
            B1B2 state;
            u32 code;
        };
        constexpr u32 NO_CODE = 0xFFFFFFFFu;
        auto lookup = [](const std::vector<Coded>& layer, const B1B2& s) -> u32 {
            const auto it = std::lower_bound(layer.begin(), layer.end(), s,
                                             [](const Coded& c, const B1B2& v) { return c.state < v; });
            return it != layer.end() && it->state == s ? it->code : NO_CODE;
        };
        // smallest (code << 6 | move) over x's predecessors in `above`
        auto smallest = [&](const std::vector<Coded>& above, const B1B2& x) -> u32 {
            u32 best = NO_CODE;
            B1B2 shifted[5];
            for (u32 dir = 0; dir < 2; ++dir) {
                for (u32 lane = 0; lane < 6; ++lane) {
                    if (dir == 0) {
                        lane_rotations::rowShifts(x, lane, shifted);
                    } else {
                        lane_rotations::colShifts(x, lane, shifted);
                    }
                    for (u32 k = 1; k <= 5; ++k) {
                        const u32 c = lookup(above, shifted[5 - k]);   // x shifted by 6 - k: where amount k came from
                        if (c != NO_CODE) {
                            best = std::min(best, (c << 6) | (dir * 30 + lane * 5 + (k - 1)));
                        }
                    }
                }
            }
            return best;
        };

        std::vector<Coded> above{{root.asB1B2(), 0}};
        for (u32 d = 1; d < depth; ++d) {
            JVec<B1B2> states;
            buildSeeds(root, d, states);
            std::vector<Coded> layer(states.size());
            parallelFor(threads, states.size(), [&](u32, const std::size_t i) {
                layer[i] = {states[i], smallest(above, states[i])};
            });
            std::sort(layer.begin(), layer.end(), [](const Coded& a, const Coded& b) { return a.state < b.state; });
            above.swap(layer);
        }

        // Hash table of the seeds: slot = seed index + 1 (0 empty), linear probing, load <= 1/2.
        const std::size_t n = seeds.size();
        if (n + 1 > 0xFFFFFFFFull) {
            throw std::runtime_error("buildSeedLastMoves: too many seeds");
        }
        std::size_t slots = 1;
        while (slots < 2 * n) {
            slots <<= 1;
        }
        const std::size_t slotMask = slots - 1;
        auto slotOf = [&](const B1B2& s) -> std::size_t {
            return mixKey(s.b1 ^ mixKey(s.b2 ^ 0x9E3779B97F4A7C15ULL)) & slotMask;
        };
        std::vector<u32> table(slots, 0);
        parallelFor(threads, n, [&](u32, const std::size_t i) {
            const u32 want = static_cast<u32>(i) + 1;
            std::size_t h = slotOf(seeds[i]);
            for (;;) {
                u32 expected = 0;
                if (std::atomic_ref<u32>(table[h]).compare_exchange_strong(expected, want, std::memory_order_relaxed)) {
                    return;
                }
                h = (h + 1) & slotMask;
            }
        });
        auto find = [&](const B1B2& x) -> u32 {   // seed index, or NO_CODE
            std::size_t h = slotOf(x);
            for (;;) {
                const u32 v = table[h];
                if (v == 0) {
                    return NO_CODE;
                }
                if (seeds[v - 1] == x) {
                    return v - 1;
                }
                h = (h + 1) & slotMask;
            }
        };

        // every child of the coded layer S-1 lowers its seed's code
        std::vector<u32> codes(n, NO_CODE);
        parallelFor(threads, above.size(), [&](u32, const std::size_t i) {
            const B1B2& parent = above[i].state;
            const u32 base = above[i].code << 6;
            B1B2 shifted[5];
            for (u32 dir = 0; dir < 2; ++dir) {
                for (u32 lane = 0; lane < 6; ++lane) {
                    if (dir == 0) {
                        lane_rotations::rowShifts(parent, lane, shifted);
                    } else {
                        lane_rotations::colShifts(parent, lane, shifted);
                    }
                    for (u32 k = 1; k <= 5; ++k) {
                        const u32 idx = find(shifted[k - 1]);
                        if (idx == NO_CODE) {
                            continue;   // a state above the seed layer
                        }
                        const u32 c = base | (dir * 30 + lane * 5 + (k - 1));
                        std::atomic_ref<u32> ref(codes[idx]);
                        u32 cur = ref.load(std::memory_order_relaxed);
                        while (c < cur && !ref.compare_exchange_weak(cur, c, std::memory_order_relaxed)) {
                        }
                    }
                }
            }
        });

        out.assign(n, perm_stream_detail::NO_PREV_MOVE);
        u64 missing = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (codes[i] == NO_CODE) {
                ++missing;
            } else {
                out[i] = static_cast<u8>(codes[i] & 63);
            }
        }
        return missing;
    }

    // Large (2 MB) pages for the probe index when the process may lock memory
    // ("Lock pages in memory" user right), otherwise normal pages. Random
    // probes into a multi-GB array then mostly avoid TLB misses / page walks.
    inline bool g_largePagesWanted = true;
    inline B1B2 g_rightRoot;   // the table side's root: the colour-mix bound's target

} // namespace ssd
