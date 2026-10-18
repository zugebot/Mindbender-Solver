#pragma once
// src/solver/ssd/common.hpp
//
// Shared pieces of the SSD meet-in-the-middle solver (experimentDiskDepth6):
// partitioning keys, the run's Config, keyed states, partition file helpers,
// left seeds and the run-wide globals. See src/solver/ssd/README.md.

#include "src/chuzzles/include.hpp"
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

    // Large (2 MB) pages for the probe index when the process may lock memory
    // ("Lock pages in memory" user right), otherwise normal pages. Random
    // probes into a multi-GB array then mostly avoid TLB misses / page walks.
    inline bool g_largePagesWanted = true;
    inline B1B2 g_rightRoot;   // the table side's root: the colour-mix bound's target

} // namespace ssd
