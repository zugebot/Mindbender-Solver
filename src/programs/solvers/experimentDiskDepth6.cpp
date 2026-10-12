// programs/experimentDiskDepth6.cpp
//
// Experiment: <S, 3, T> meet-in-the-middle with the right frontier on disk
// (S = --seed-depth, default 3; T = --table-depth 6 (default) or 7).
// <3,3,6> = 12 moves, <4,3,6> = 13, <5,3,6> = 14, <5,3,7> = 15.
//
//   Build:  exact depth-5 frontier in RAM -> expand every state by one move ->
//           route each child to one of 256 partition files by the top 8 bits of
//           a mixed state key -> sort + de-dupe each partition independently.
//           --ram-build: regenerate the children per RAM-sized group of
//           partitions, sort them in memory and write the table + index in one
//           go (no raw spill, no re-read). Correct, but on a drive whose
//           sustained write speed has collapsed (X: measured 32 MB/s) it was
//           slower (6-2: 475 s vs ~200 s), because the disk build's writes are
//           absorbed by the OS file cache.
//   Probe (default): load a compact per-partition index (Bloom filter + u16
//           fingerprints + bucket offsets, ~4.25 B/state) for as many partitions
//           as fit --ram-gb, stream the left side once per such range, and
//           confirm fingerprint hits against the full states on disk.
//   Probe (--batched): load a contiguous range of sorted partitions that fits the RAM
//           budget, index it with a direct-address bucket table, then regenerate
//           the whole left side (exact depth-S seeds, canonical depth-3 stream)
//           and probe only states whose key falls in that range. Repeat per range.
//   Depth 7 (--table-depth 7): the depth-7 states (~46 G for a fat goal,
//           ~750 GB as states) are never stored. Every child of the depth-6
//           table is kept only as a 64-bit key, a few partitions per pass over
//           the depth-6 table, then sorted, de-duped and turned into the same
//           compact index (Bloom + u16 fingerprints + bucket offsets, ~3.4
//           B/state). A fingerprint hit L is confirmed exactly by checking that
//           one of L's one-move neighbours is in the exact depth-6 table.
//           --d7-partitions N [--d7-first-partition F] builds and probes only
//           key partitions [F, F+N), a cheap way to check results against known
//           solutions: the middles and solutions whose middle falls in those
//           partitions must match exactly.
//   No table (default for depth 6 when no table is on disk yet): nothing is
//           written. The depth-5 frontier stays in RAM; each probe pass builds
//           its slice of the fingerprint index in RAM straight from the keys of
//           the regenerated depth-6 children, and the fingerprint hits are
//           confirmed by regenerating depth 6 once more. The drive's (slow)
//           write speed never matters. --disk-table forces the on-disk table.
//           With --table-depth 7 the same holds one level deeper: each pass
//           fills its index slice straight from the regenerated depth-7
//           descendants of the in-RAM depth-5 frontier (raw fingerprints in
//           counted buckets, de-duplicated per bucket afterwards).
//   Recover: full solutions through every middle via BoardSolverFrontier.
//
// All disk I/O is sequential. With --solutions, both the middle states and the
// recovered solution set are checked against a complete known solution file.
//
// Usage:
//   experimentDiskDepth6 <puzzle> [--out X:/mindbender_d6] [--batch 320000000]
//                        [--threads N] [--seed-depth S] [--reverse] [--rebuild]
//                        [--no-recover] [--batched] [--ram-gb GB] [--no-large-pages]
//                        [--solutions file]
//                        [--table-depth 6|7 [--d7-partitions N [--d7-first-partition F]]]
//                        [--progress-file path.json [--progress-interval SECONDS]]
//                        [--probe-variant G:P] [--no-calibrate] [--calibrate-seconds S]
//
// Probe self-calibration: the first ~20 s of the first probe pass run the real
// seeds through every probe variant (batch size G = 32/64/128 lookups, P =
// pipelined: the next batch's memory fetches are issued before the current
// batch is tested), interleaved seed by seed so they share the same memory and
// CPU conditions. The fastest variant then runs the rest of the search. That
// work is part of the search (nothing is repeated). Results are cached per
// machine in %LOCALAPPDATA%/MindbenderSolver/probe_profile.txt; --no-calibrate
// uses the cached best, --probe-variant forces one.
//
// --bound auto|off|leaf|both: an admissible lane lower bound
// (--bound-kind rotation, the default: per row / column, the fewest cells wrong
// over its 6 rotations; or mix: colour multisets only)
// (src/solver/colour_mix_bound.hpp) skips left states that cannot be within
// the table depth of the right root ("leaf": before their lookup) and, on
// normal boards, whole last stream levels under nodes that cannot have such a
// child ("both": inside PermStream). auto audits the bound on this run's own
// seeds (fast tables vs the reference definition, |dh| <= 1 on real edges,
// and pruned vs full streams keeping identical wanted leaves), then times
// every eligible mode on real seeds and keeps the fastest: blocks of seeds,
// one mode on all threads per block, rotating; clearly slower modes are
// dropped and timing goes on until the rest are told apart, up to 3% of the
// search (20 s minimum, --bound-calibrate-seconds maximum, default 300).
// Any audit failure turns it off. 2x2 boards: off (the halved bound prunes
// nothing).
// --pdb auto|off|on adds grouped pattern databases (src/solver/group_pdb.hpp):
// up to 3 merged-colour classes solved exactly at startup (2 bits per entry),
// asked at the same parent nodes ("pdb") and optionally at leaves
// ("pdbleaf"); audited the same way and timed with the other modes. auto:
// normal boards, 14+ moves (seed depth >= 5), when colour-mix leaves room.
// --stream-depth 4 (default 3): a 4-move left stream from the seeds, so
// <5,4,6> = 15 moves; the parent prune then runs at every level above the
// leaves (a node with r moves below is dropped when h > depth + r).
// "parent" / "pdbparent" prune only at parents, with no per-leaf check: on
// puzzles where the leaf check skips little (16-4: 39%) it costs more than
// the lookups it saves, while a parent check covers ~40 leaves.
//
// Work units (searches split across runs or machines):
//   --shard k/N         search only the left seeds with index = k (mod N); the
//                       N shards together are exactly the full search
//   --middles-out FILE  write the confirmed middles (text, with the puzzle,
//                       split and shard in the header and an "end" line)
//   --recover-from FILE (repeatable) skip the search: merge the middles of
//                       shards 0..N-1 of one search (each exactly once, same
//                       puzzle and split), then recover and validate as usual
//
// --progress-file: a background thread rewrites a small JSON status (phase,
// done/total units, elapsed) every few seconds for python/scheduler; workers
// only add to an atomic counter once per seed / partition. Log lines are
// flushed as they are written, so a redirected log can be followed live.
//                        [--d7-sample N [--d7-pass-budget STATES]]

#include "src/chuzzles/include.hpp"
#include "src/utils/progress_reporter.hpp"
#include "src/solver/colour_mix_bound.hpp"
#include "src/solver/group_pdb.hpp"

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

namespace {

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
        colour_mix::Kind boundKind = colour_mix::Kind::Rotation;   // --bound-kind mix|rotation   // max; the run uses 3% of its search, >= 20 s

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

    fs::path rawPath(const fs::path& dir, const u32 p) {
        char name[32];
        std::snprintf(name, sizeof(name), "raw_%03x.bin", p);
        return dir / name;
    }

    fs::path sortedPath(const fs::path& dir, const u32 p) {
        char name[32];
        std::snprintf(name, sizeof(name), "sorted_%03x.bin", p);
        return dir / name;
    }

    double gbOf(const u64 states) {
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

    // ------------------------------------------------------------------
    // Build: depth-6 exact frontier -> sorted unique partition files
    // ------------------------------------------------------------------
    std::vector<u64> buildDepth6OnDisk(const Board& root, const fs::path& dir, const u32 threads) {
        fs::create_directories(dir);

        Timer d5Timer;
        g_progress.phase("d6-build", 0, "depth-5 frontier");
        JVec<B1B2> depth5;
        JVec<u64> depth5Hashes;
        buildUniqueNoneDepthFrontierB1B2<5>(root, depth5, depth5Hashes, false);
        JVec<u64>{}.swap(depth5Hashes);
        tcout << "depth-5 exact unique: " << depth5.size() << " states in "
              << d5Timer.getSeconds() << "s\n";

        // Expand by one move, routing children to partition files.
        Timer expandTimer;
        std::vector<std::FILE*> files(PARTITION_COUNT);
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            files[p] = std::fopen(rawPath(dir, p).string().c_str(), "wb");
            if (files[p] == nullptr) {
                throw std::runtime_error("failed to open partition file");
            }
        }
        std::vector<std::mutex> fileLocks(PARTITION_COUNT);
        std::vector<std::vector<u64>> rawCounts(threads, std::vector<u64>(PARTITION_COUNT, 0));
        std::vector<std::vector<u64>> plainHashSkew(threads, std::vector<u64>(PARTITION_COUNT, 0));

        const bool isFat = root.getFatBool();
        const std::size_t chunk = 1U << 16;
        const std::size_t chunkCount = (depth5.size() + chunk - 1) / chunk;

        std::vector<std::vector<std::vector<B1B2>>> buffers(threads);
        for (auto& perThread : buffers) {
            perThread.resize(PARTITION_COUNT);
            for (auto& buf : perThread) {
                buf.reserve(WRITE_BUFFER_STATES);
            }
        }

        auto flush = [&](const u32 p, std::vector<B1B2>& buf) {
            std::lock_guard<std::mutex> lock(fileLocks[p]);
            if (std::fwrite(buf.data(), sizeof(B1B2), buf.size(), files[p]) != buf.size()) {
                throw std::runtime_error("partition write failed");
            }
            buf.clear();
        };

        g_progress.phase("d6-build", chunkCount, "expand depth 5 -> 6");
        parallelFor(threads, chunkCount, [&](const u32 t, const std::size_t c) {
            B1B2 children[60];
            u64 childHashes[60];
            auto& bufs = buffers[t];
            const std::size_t begin = c * chunk;
            const std::size_t end = std::min(depth5.size(), begin + chunk);
            for (std::size_t i = begin; i < end; ++i) {
                const std::size_t n = isFat
                        ? frontier_recovery_detail::emitNoneChildren<true>(depth5[i], children, childHashes)
                        : frontier_recovery_detail::emitNoneChildren<false>(depth5[i], children, childHashes);
                for (std::size_t k = 0; k < n; ++k) {
                    const u32 p = static_cast<u32>(mixKey(childHashes[k]) >> PARTITION_SHIFT);
                    ++rawCounts[t][p];
                    ++plainHashSkew[t][childHashes[k] >> PARTITION_SHIFT];
                    bufs[p].push_back(children[k]);
                    if (bufs[p].size() == WRITE_BUFFER_STATES) {
                        flush(p, bufs[p]);
                    }
                }
            }
            g_progress.add();
        });
        for (u32 t = 0; t < threads; ++t) {
            for (u32 p = 0; p < PARTITION_COUNT; ++p) {
                if (!buffers[t][p].empty()) {
                    flush(p, buffers[t][p]);
                }
            }
        }
        for (std::FILE* f : files) {
            std::fclose(f);
        }
        JVec<B1B2>{}.swap(depth5);
        buffers.clear();

        u64 rawTotal = 0, rawMax = 0, rawMin = ~0ULL, plainMax = 0, plainNonEmpty = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            u64 raw = 0, plain = 0;
            for (u32 t = 0; t < threads; ++t) {
                raw += rawCounts[t][p];
                plain += plainHashSkew[t][p];
            }
            rawTotal += raw;
            rawMax = std::max(rawMax, raw);
            rawMin = std::min(rawMin, raw);
            plainMax = std::max(plainMax, plain);
            plainNonEmpty += plain != 0;
        }
        const double expandSeconds = expandTimer.getSeconds();
        tcout << "depth-6 raw children: " << rawTotal << " (" << gbOf(rawTotal) << " GB) written in "
              << expandSeconds << "s = " << gbOf(rawTotal) / expandSeconds * 1000.0 << " MB/s\n";
        tcout << "partition sizes (mixed key): min " << rawMin << ", max " << rawMax
              << ", avg " << rawTotal / PARTITION_COUNT << '\n';
        tcout << "same split on StateHash top 8 bits would be: " << plainNonEmpty << "/"
              << PARTITION_COUNT << " partitions used, max " << plainMax << '\n';

        // Sort + de-dupe each partition.
        Timer sortTimer;
        std::vector<u64> uniqueCounts(PARTITION_COUNT, 0);
        const u32 sortThreads = std::max(1U, threads / 2);
        g_progress.phase("d6-build", PARTITION_COUNT, "sort + de-dupe partitions");
        parallelFor(sortThreads, PARTITION_COUNT, [&](u32, const std::size_t p) {
            const fs::path raw = rawPath(dir, static_cast<u32>(p));
            std::vector<B1B2> states = readAll<B1B2>(raw);
            std::vector<KeyedState> keyed(states.size());
            for (std::size_t i = 0; i < states.size(); ++i) {
                keyed[i] = {keyOf(states[i]), states[i]};
            }
            std::sort(keyed.begin(), keyed.end());
            std::size_t w = 0;
            for (std::size_t i = 0; i < keyed.size(); ++i) {
                if (w == 0 || keyed[i].key != keyed[w - 1].key || !(keyed[i].state == keyed[w - 1].state)) {
                    keyed[w++] = keyed[i];
                }
            }
            states.resize(w);
            for (std::size_t i = 0; i < w; ++i) {
                states[i] = keyed[i].state;
            }
            writeAll(sortedPath(dir, static_cast<u32>(p)), states.data(), states.size());
            fs::remove(raw);
            uniqueCounts[p] = w;
            g_progress.add();
        });
        u64 uniqueTotal = 0;
        for (const u64 c : uniqueCounts) {
            uniqueTotal += c;
        }
        tcout << "depth-6 exact unique: " << uniqueTotal << " (" << gbOf(uniqueTotal) << " GB on disk), "
              << "sort+dedupe " << sortTimer.getSeconds() << "s\n";
        return uniqueCounts;
    }

    std::vector<u64> loadPartitionCounts(const fs::path& dir) {
        std::vector<u64> counts(PARTITION_COUNT, 0);
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            const fs::path path = sortedPath(dir, p);
            if (!fs::exists(path)) {
                return {};
            }
            counts[p] = fs::file_size(path) / sizeof(B1B2);
        }
        return counts;
    }

    // ------------------------------------------------------------------
    // Probe: <S, 3> left side against batches of partitions
    // ------------------------------------------------------------------

    constexpr u32 LEFT_STREAM_DEPTH = 3;
    constexpr u32 PROBE_GROUP = 32;

    // Direct-address index over one batch of key-sorted partitions. Keys are
    // uniform (mixKey), so a bucket is just the top bits of the key, sized for
    // ~2 states per bucket, and the bucket starts are built in one linear pass.
    struct BatchIndex {
        JVec<B1B2> states;
        std::vector<u32> starts;   // bucketCount + 1 offsets into states
        u32 shift = 0;             // key >> shift = global bucket number
        u64 firstBucket = 0;

        void build(const u32 threads, const u32 pBegin, const u32 pEnd) {
            const u64 n = states.size();
            if (n >= (1ULL << 32)) {
                throw std::runtime_error("batch too large for 32-bit bucket offsets");
            }
            const u64 partitions = pEnd - pBegin;
            u32 extraBits = 0;
            while (extraBits < PARTITION_SHIFT && (partitions << (extraBits + 1)) <= std::max<u64>(1, n / 2)) {
                ++extraBits;
            }
            shift = PARTITION_SHIFT - extraBits;
            firstBucket = static_cast<u64>(pBegin) << extraBits;
            const u64 bucketCount = partitions << extraBits;

            std::vector<u32> bucketOf(n);
            parallelFor(threads, (n + 65535) / 65536, [&](u32, const std::size_t c) {
                const u64 end = std::min<u64>(n, (c + 1) * 65536);
                for (u64 i = c * 65536; i < end; ++i) {
                    bucketOf[i] = static_cast<u32>((keyOf(states[i]) >> shift) - firstBucket);
                }
            });
            starts.assign(bucketCount + 1, 0);
            u64 next = 0;
            for (u64 i = 0; i < n; ++i) {
                while (next <= bucketOf[i]) {
                    starts[next++] = static_cast<u32>(i);
                }
            }
            while (next <= bucketCount) {
                starts[next++] = static_cast<u32>(n);
            }
        }

        // Probes a group of in-range states, prefetching the bucket offsets and
        // then the bucket contents so the DRAM misses overlap.
        void probeGroup(const B1B2* probes, const u64* keys, const u32 count, std::vector<B1B2>& found) const {
            u64 bucket[PROBE_GROUP];
            u32 lo[PROBE_GROUP], hi[PROBE_GROUP];
            for (u32 i = 0; i < count; ++i) {
                bucket[i] = (keys[i] >> shift) - firstBucket;
                __builtin_prefetch(&starts[bucket[i]]);
            }
            for (u32 i = 0; i < count; ++i) {
                lo[i] = starts[bucket[i]];
                hi[i] = starts[bucket[i] + 1];
                if (lo[i] < hi[i]) {
                    __builtin_prefetch(&states[lo[i]]);
                }
            }
            for (u32 i = 0; i < count; ++i) {
                for (u32 j = lo[i]; j < hi[i]; ++j) {
                    if (states[j] == probes[i]) {
                        found.push_back(probes[i]);
                        break;
                    }
                }
            }
        }
    };

    void buildSeeds(const Board& root, const u32 depth, JVec<B1B2>& seeds) {
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

    std::vector<B1B2> probeAllBatches(const Board& leftRoot,
                                      const fs::path& dir,
                                      const std::vector<u64>& counts,
                                      const Config& cfg,
                                      const u32 threads) {
        Timer seedTimer;
        JVec<B1B2> seeds;
        buildSeeds(leftRoot, cfg.seedDepth, seeds);
        tcout << "left seeds (exact depth " << cfg.seedDepth << "): " << seeds.size()
              << " in " << seedTimer.getSeconds() << "s\n";

        std::vector<std::pair<u32, u32>> batches;
        for (u32 p = 0; p < PARTITION_COUNT;) {
            u32 q = p;
            u64 size = 0;
            while (q < PARTITION_COUNT && (q == p || size + counts[q] <= cfg.batchStates)) {
                size += counts[q++];
            }
            batches.emplace_back(p, q);
            p = q;
        }
        tcout << "probe batches: " << batches.size() << " (budget " << cfg.batchStates << " states each)\n";

        std::vector<B1B2> allMiddles;
        double loadSeconds = 0.0, indexSeconds = 0.0, probeSeconds = 0.0;
        u64 streamedTotal = 0, inRangeTotal = 0;

        for (std::size_t b = 0; b < batches.size(); ++b) {
            const auto [pBegin, pEnd] = batches[b];

            Timer loadTimer;
            u64 batchSize = 0;
            for (u32 p = pBegin; p < pEnd; ++p) {
                batchSize += counts[p];
            }
            BatchIndex index;
            index.states.resize(batchSize);
            u64 offset = 0;
            for (u32 p = pBegin; p < pEnd; ++p) {
                std::FILE* f = std::fopen(sortedPath(dir, p).string().c_str(), "rb");
                if (f == nullptr || std::fread(index.states.data() + offset, sizeof(B1B2), counts[p], f) != counts[p]) {
                    throw std::runtime_error("failed to read sorted partition");
                }
                std::fclose(f);
                offset += counts[p];
            }
            const double readSeconds = loadTimer.getSeconds();
            loadSeconds += readSeconds;

            Timer indexTimer;
            index.build(threads, pBegin, pEnd);
            const double batchIndexSeconds = indexTimer.getSeconds();
            indexSeconds += batchIndexSeconds;

            Timer probeTimer;
            std::mutex middleLock;
            std::vector<u64> streamed(threads, 0), inRange(threads, 0);
            parallelFor(threads, seeds.size(), [&](const u32 t, const std::size_t s) {
                struct Sink {
                    const BatchIndex& index;
                    const u32 pBegin, pEnd;
                    B1B2 groupStates[PROBE_GROUP];
                    u64 groupKeys[PROBE_GROUP];
                    u32 groupCount = 0;
                    std::vector<B1B2> found;
                    u64 streamed = 0, inRange = 0;

                    void flush() {
                        index.probeGroup(groupStates, groupKeys, groupCount, found);
                        groupCount = 0;
                    }

                    void operator()(JVec<B1B2>& chunkStates, JVec<u64>& chunkHashes, const u32 count) {
                        streamed += count;
                        for (u32 i = 0; i < count; ++i) {
                            const u64 key = mixKey(chunkHashes[i]);
                            const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                            if (p < pBegin || p >= pEnd) {
                                continue;
                            }
                            groupStates[groupCount] = chunkStates[i];
                            groupKeys[groupCount] = key;
                            ++inRange;
                            if (++groupCount == PROBE_GROUP) {
                                flush();
                            }
                        }
                    }
                } sink{index, pBegin, pEnd, {}, {}, 0, {}, 0, 0};

                const Board seedBoard = makeBoardFromState(seeds[s]);
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_STREAM_DEPTH>(seedBoard, sink, 4096);
                sink.flush();

                streamed[t] += sink.streamed;
                inRange[t] += sink.inRange;
                if (!sink.found.empty()) {
                    std::lock_guard<std::mutex> lock(middleLock);
                    allMiddles.insert(allMiddles.end(), sink.found.begin(), sink.found.end());
                }
            });
            const double batchProbeSeconds = probeTimer.getSeconds();
            probeSeconds += batchProbeSeconds;

            u64 batchStreamed = 0, batchInRange = 0;
            for (u32 t = 0; t < threads; ++t) {
                batchStreamed += streamed[t];
                batchInRange += inRange[t];
            }
            streamedTotal += batchStreamed;
            inRangeTotal += batchInRange;
            tcout << "  batch " << (b + 1) << "/" << batches.size()
                  << " partitions [" << pBegin << "," << pEnd << ") " << batchSize << " states"
                  << ": read " << readSeconds << "s (" << gbOf(batchSize) / readSeconds * 1000.0 << " MB/s)"
                  << ", index " << batchIndexSeconds << "s"
                  << ", stream+probe " << batchProbeSeconds << "s ("
                  << static_cast<double>(batchStreamed) / batchProbeSeconds / 1e6 << " M left states/s, "
                  << static_cast<double>(batchInRange) / batchProbeSeconds / 1e6 << " M probes/s)\n";
        }

        std::sort(allMiddles.begin(), allMiddles.end());
        allMiddles.erase(std::unique(allMiddles.begin(), allMiddles.end()), allMiddles.end());

        tcout << "probe totals: load " << loadSeconds << "s, index " << indexSeconds << "s, stream+probe "
              << probeSeconds << "s; left states streamed " << streamedTotal << " (all batches), probes "
              << inRangeTotal << '\n';
        return allMiddles;
    }

    // ------------------------------------------------------------------
    // Fingerprint mode: compact in-RAM index, one left pass per key range
    // ------------------------------------------------------------------
    //
    // Everything is laid out per partition, so any contiguous partition range
    // can be loaded on its own:
    //   fp16.bin    : u16 low key bits of every table state, in key order
    //   fpdir32.bin : per partition, (bucketsPerPartition + 1) u32 offsets
    //                 relative to the partition start (~4 states per bucket)
    //   bloomp.bin  : per partition, a split-block Bloom filter (10 bits/state)
    //   fp16meta.txt: "<bucketBitsPerPartition> <bloomBlocksPerPartition> <states>"
    //
    // A probe passes its partition range check, then the Bloom block (one
    // cache line, ~1.3% pass), then the bucket's fingerprints. Survivors are
    // confirmed against the full states on disk.

    constexpr u32 BLOOM_BITS_PER_STATE = 10;
    constexpr u32 TARGET_STATES_PER_BUCKET = 4;
    constexpr u32 BLOOM_SALT[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                   0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};

    // Independent of the partition/bucket (top) and fingerprint (low) key bits.
    FORCEINLINE u64 bloomHash(const u64 key) {
        return mixKey(key ^ 0x5851f42d4c957f2dULL);
    }

    // Split-block Bloom blocks: 8 x u32 (32 bytes, never straddling a cache
    // line); a key sets one bit in each word of its block.
    FORCEINLINE void bloomInsertAtomic(u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        for (u32 i = 0; i < 8; ++i) {
            __atomic_fetch_or(&block[i], 1U << ((low * BLOOM_SALT[i]) >> 27), __ATOMIC_RELAXED);
        }
    }

    // For a block only one thread writes.
    FORCEINLINE void bloomInsert(u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        for (u32 i = 0; i < 8; ++i) {
            block[i] |= 1U << ((low * BLOOM_SALT[i]) >> 27);
        }
    }

    FORCEINLINE bool bloomTest(const u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        u32 missing = 0;
        for (u32 i = 0; i < 8; ++i) {
            missing |= ~block[i] & (1U << ((low * BLOOM_SALT[i]) >> 27));
        }
        return missing == 0;
    }

    FORCEINLINE u64 bloomBlockInPartition(const u64 h, const u64 blocksPerPartition) {
        return ((h >> 32) * blocksPerPartition) >> 32;
    }

    // u32 storage over-allocated so `aligned` can start on a 64-byte boundary.
    struct AlignedWords {
        std::vector<u32> storage;
        u32* aligned = nullptr;

        void allocate(const u64 words) {
            storage.assign(words + 16, 0);
            const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
            aligned = reinterpret_cast<u32*>((address + 63) & ~static_cast<std::uintptr_t>(63));
        }
    };

    // Large (2 MB) pages for the probe index when the process may lock memory
    // ("Lock pages in memory" user right), otherwise normal pages. Random
    // probes into a multi-GB array then mostly avoid TLB misses / page walks.
    bool g_largePagesWanted = true;
    B1B2 g_rightRoot;   // the table side's root: the colour-mix bound's target

    // Parent-level prune: colour-mix, then the grouped pattern databases.
    // Per lane (5 nodes): every table entry still needed is located and
    // prefetched first, then tested, so the cache misses overlap instead of
    // waiting one after another.
    struct ParentPrunePdb {
        static constexpr bool ENABLED = true;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5]) {
            u32 m = colour_mix::ParentPrune::skipMask<BELOW>(nodes);
            const u32 nt = group_pdb::g_tableCount;
            if (m == 31U || nt == 0 || BELOW > 2) {   // the tables encode up to limit + 2
                return m;
            }
            u64 idx[4][5];
            for (u32 i = 0; i < nt; ++i) {
                for (u32 k = 0; k < 5; ++k) {
                    if ((m & (1U << k)) == 0) {
                        idx[i][k] = group_pdb::g_tables[i]->indexOf(nodes[k]);
                        group_pdb::g_tables[i]->prefetchAt(idx[i][k]);
                    }
                }
            }
            for (u32 i = 0; i < nt; ++i) {
                for (u32 k = 0; k < 5; ++k) {
                    if ((m & (1U << k)) == 0 && group_pdb::g_tables[i]->codeAt(idx[i][k]) > static_cast<u32>(BELOW)) {
                        m |= 1U << k;
                    }
                }
            }
            return m;
        }
    };
    bool g_indexBuiltWithTable = false;   // set by the in-RAM depth-6 build

    bool enableLockMemoryPrivilege() {
        static const bool enabled = [] {
            HANDLE token = nullptr;
            if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
                return false;
            }
            TOKEN_PRIVILEGES privileges{};
            privileges.PrivilegeCount = 1;
            privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &privileges.Privileges[0].Luid)
                      && AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr)
                      && GetLastError() == ERROR_SUCCESS;
            CloseHandle(token);
            return ok;
        }();
        return enabled;
    }

    template<typename T>
    class BigArray {
        T* data_ = nullptr;
        u64 size_ = 0;
        u64 capacity_ = 0;
        bool large_ = false;

        void release() {
            if (data_ != nullptr) {
                VirtualFree(data_, 0, MEM_RELEASE);
            }
            data_ = nullptr;
            size_ = 0;
            capacity_ = 0;
            large_ = false;
        }

    public:
        BigArray() = default;
        BigArray(const BigArray&) = delete;
        BigArray& operator=(const BigArray&) = delete;
        ~BigArray() { release(); }

        // Storage for `count` elements, reusing the current block when it is
        // big enough (contents are then NOT cleared: callers overwrite them).
        // Allocating once, early, and reusing it across passes avoids
        // re-committing (and re-zeroing) many GB per pass and keeps a large-page
        // block once one was obtained.
        void ensure(const u64 count) {
            if (data_ != nullptr && count <= capacity_) {
                size_ = count;
                return;
            }
            allocate(count);
        }

        // Zero-filled, page-aligned storage for `count` elements.
        void allocate(const u64 count) {
            release();
            size_ = count;
            capacity_ = count;
            const u64 bytes = std::max<u64>(1, count * sizeof(T));
            if (g_largePagesWanted && enableLockMemoryPrivilege()) {
                const u64 page = GetLargePageMinimum();
                if (page != 0) {
                    data_ = static_cast<T*>(VirtualAlloc(nullptr, (bytes + page - 1) / page * page,
                                                         MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                         PAGE_READWRITE));
                    large_ = data_ != nullptr;
                }
            }
            if (data_ == nullptr) {
                data_ = static_cast<T*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            }
            if (data_ == nullptr) {
                throw std::runtime_error("VirtualAlloc failed for probe index");
            }
        }

        MUND T* data() { return data_; }
        MUND const T* data() const { return data_; }
        MUND u64 size() const { return size_; }
        MUND bool largePages() const { return large_; }
        FORCEINLINE T& operator[](const u64 i) { return data_[i]; }
        FORCEINLINE const T& operator[](const u64 i) const { return data_[i]; }
    };

    struct FingerprintMeta {
        u32 bucketBits = 0;           // buckets per partition = 2^bucketBits
        u64 bloomBlocksPerPartition = 0;
        u64 stateCount = 0;
        u32 partitionBegin = 0;                 // index covers partitions [begin, limit);
        u32 partitionLimit = PARTITION_COUNT;   // its files start at partition `begin`

        MUND u64 bucketsPerPartition() const { return 1ULL << bucketBits; }
        MUND u32 bucketShift() const { return PARTITION_SHIFT - bucketBits; }
        MUND u64 indexBytes(const u64 states, const u32 partitions) const {
            return states * 2 + partitions * ((bucketsPerPartition() + 1) * 4 + bloomBlocksPerPartition * 32);
        }
    };

    fs::path fp16Path(const fs::path& dir) { return dir / "fp16.bin"; }
    fs::path fpDir32Path(const fs::path& dir) { return dir / "fpdir32.bin"; }
    fs::path bloomPartPath(const fs::path& dir) { return dir / "bloomp.bin"; }
    fs::path fp16MetaPath(const fs::path& dir) { return dir / "fp16meta.txt"; }

    // The files of one compact index: the depth-6 table's (next to its states)
    // or the index-only depth-7 table's (no states; per-partition counts kept).
    struct IndexFiles {
        fs::path fp16, dir32, bloom, meta, counts;
    };

    IndexFiles d6Files(const fs::path& dir) {
        return {fp16Path(dir), fpDir32Path(dir), bloomPartPath(dir), fp16MetaPath(dir), {}};
    }

    IndexFiles d7Files(const fs::path& dir) {
        return {dir / "d7_fp16.bin", dir / "d7_fpdir32.bin", dir / "d7_bloomp.bin", dir / "d7_meta.txt",
                dir / "d7_counts.bin"};
    }

    FingerprintMeta readFingerprintMeta(const fs::path& metaPath) {
        FingerprintMeta meta;
        std::ifstream in(metaPath);
        in >> meta.bucketBits >> meta.bloomBlocksPerPartition >> meta.stateCount;
        if (!(in >> meta.partitionLimit)) {
            meta.partitionLimit = PARTITION_COUNT;
        }
        if (!(in >> meta.partitionBegin)) {
            meta.partitionBegin = 0;
        }
        return meta;
    }

    void buildFingerprintIndex(const fs::path& dir, const std::vector<u64>& counts, const u32 threads) {
        Timer timer;
        u64 n = 0, maxCount = 0;
        for (const u64 c : counts) {
            n += c;
            maxCount = std::max(maxCount, c);
        }
        FingerprintMeta meta;
        meta.stateCount = n;
        while (meta.bucketBits < PARTITION_SHIFT
               && (maxCount >> meta.bucketBits) > TARGET_STATES_PER_BUCKET) {
            ++meta.bucketBits;
        }
        meta.bloomBlocksPerPartition = std::max<u64>(1, (maxCount * BLOOM_BITS_PER_STATE + 255) / 256);
        const u64 bucketsPerPartition = meta.bucketsPerPartition();
        const u32 shift = meta.bucketShift();

        std::FILE* fpOut = std::fopen(fp16Path(dir).string().c_str(), "wb");
        std::FILE* dirOut = std::fopen(fpDir32Path(dir).string().c_str(), "wb");
        std::FILE* bloomOut = std::fopen(bloomPartPath(dir).string().c_str(), "wb");
        if (fpOut == nullptr || dirOut == nullptr || bloomOut == nullptr) {
            throw std::runtime_error("failed to open fingerprint index files");
        }

        AlignedWords bloom;
        std::vector<u32> starts(bucketsPerPartition + 1);
        g_progress.phase("d6-index", PARTITION_COUNT, "fingerprint index of the depth-6 table");
        for (u32 p = 0; p < PARTITION_COUNT; ++p, g_progress.add()) {
            const std::vector<B1B2> states = readAll<B1B2>(sortedPath(dir, p));
            std::vector<u64> keys(states.size());
            bloom.allocate(meta.bloomBlocksPerPartition * 8);
            parallelFor(threads, (states.size() + 65535) / 65536, [&](u32, const std::size_t c) {
                const std::size_t end = std::min(states.size(), (c + 1) * 65536);
                for (std::size_t i = c * 65536; i < end; ++i) {
                    keys[i] = keyOf(states[i]);
                    const u64 h = bloomHash(keys[i]);
                    bloomInsertAtomic(bloom.aligned + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
                }
            });

            std::vector<u16> fps(states.size());
            u64 next = 0;
            for (std::size_t i = 0; i < states.size(); ++i) {
                const u64 bucket = (keys[i] >> shift) & (bucketsPerPartition - 1);
                while (next <= bucket) {
                    starts[next++] = static_cast<u32>(i);
                }
                fps[i] = static_cast<u16>(keys[i]);
            }
            while (next <= bucketsPerPartition) {
                starts[next++] = static_cast<u32>(states.size());
            }

            if (std::fwrite(fps.data(), sizeof(u16), fps.size(), fpOut) != fps.size()
                || std::fwrite(starts.data(), sizeof(u32), starts.size(), dirOut) != starts.size()
                || std::fwrite(bloom.aligned, sizeof(u32), meta.bloomBlocksPerPartition * 8, bloomOut)
                   != meta.bloomBlocksPerPartition * 8) {
                throw std::runtime_error("fingerprint index write failed");
            }
        }
        std::fclose(fpOut);
        std::fclose(dirOut);
        std::fclose(bloomOut);
        std::ofstream(fp16MetaPath(dir)) << meta.bucketBits << ' ' << meta.bloomBlocksPerPartition
                                         << ' ' << n << '\n';

        tcout << "BUILD fingerprint index: " << n << " states, 2^" << meta.bucketBits
              << " buckets/partition, " << meta.indexBytes(n, PARTITION_COUNT) / 1e9 << " GB ("
              << static_cast<double>(meta.indexBytes(n, PARTITION_COUNT)) / static_cast<double>(n)
              << " B/state) in " << timer.getSeconds() << "s\n";
    }

    constexpr u32 MAX_PROBE_GROUP = 128;

    // One batch of in-range probes and its stage-1 results.
    struct ProbeBatch {
        B1B2 states[MAX_PROBE_GROUP];
        u64 keys[MAX_PROBE_GROUP];
        u64 h[MAX_PROBE_GROUP];
        const u32* blocks[MAX_PROBE_GROUP];
        u32 count = 0;
    };

    // The index for partitions [pBegin, pEnd), loaded with three contiguous reads.
    struct FingerprintRange {
        FingerprintMeta meta;
        u32 pBegin = 0, pEnd = 0;
        std::vector<u64> partitionFirst;   // first fp index of each partition in range, + end
        BigArray<u16> fingerprints;
        BigArray<u32> starts;              // (pEnd - pBegin) * (bucketsPerPartition + 1)
        BigArray<u32> bloom;               // page-aligned, so blocks never straddle lines

        void load(const IndexFiles& files, const FingerprintMeta& m, const std::vector<u64>& counts,
                  const u32 begin, const u32 end) {
            meta = m;
            pBegin = begin;
            pEnd = end;
            u64 before = 0;
            for (u32 p = 0; p < pBegin; ++p) {
                before += counts[p];
            }
            partitionFirst.assign(1, 0);
            for (u32 p = pBegin; p < pEnd; ++p) {
                partitionFirst.push_back(partitionFirst.back() + counts[p]);
            }
            const u64 parts = pEnd - pBegin;
            const u64 dirPer = meta.bucketsPerPartition() + 1;
            const u64 bloomPer = meta.bloomBlocksPerPartition * 8;

            fingerprints.ensure(partitionFirst.back());
            starts.ensure(parts * dirPer);
            bloom.ensure(parts * bloomPer);
            readRange(files.fp16, before * sizeof(u16), fingerprints.data(), fingerprints.size() * sizeof(u16));
            const u64 filePart = pBegin - meta.partitionBegin;   // files start at partitionBegin
            readRange(files.dir32, filePart * dirPer * sizeof(u32), starts.data(), starts.size() * sizeof(u32));
            readRange(files.bloom, filePart * bloomPer * sizeof(u32), bloom.data(), parts * bloomPer * sizeof(u32));
        }

        // Allocates the buffers for the largest pass up front, before the left
        // seeds and per-pass work fragment memory.
        void reserve(const FingerprintMeta& m, const u64 maxStates, const u64 maxPartitions) {
            fingerprints.ensure(maxStates);
            starts.ensure(maxPartitions * (m.bucketsPerPartition() + 1));
            bloom.ensure(maxPartitions * m.bloomBlocksPerPartition * 8);
        }

        MUND bool largePages() const {
            return fingerprints.largePages() && starts.largePages() && bloom.largePages();
        }

        // Chunked fread with a 64-bit seek: a single ifstream::read of more
        // than 2 GB fails on MinGW/Windows.
        static void readRange(const fs::path& path, const u64 offset, void* dst, const u64 bytes) {
            std::FILE* f = std::fopen(path.string().c_str(), "rb");
            if (f == nullptr || _fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) {
                throw std::runtime_error("failed to open/seek " + path.string());
            }
            constexpr u64 CHUNK = 1ULL << 30;
            char* out = static_cast<char*>(dst);
            for (u64 done = 0; done < bytes;) {
                const u64 n = std::min(CHUNK, bytes - done);
                if (std::fread(out + done, 1, n, f) != n) {
                    std::fclose(f);
                    throw std::runtime_error("failed to read " + path.string());
                }
                done += n;
            }
            std::fclose(f);
        }

        MUND u64 bytes() const {
            return fingerprints.size() * 2 + starts.size() * 4
                   + (pEnd - pBegin) * meta.bloomBlocksPerPartition * 32;
        }

        // Stage 1 of a probe batch: Bloom hashes and block addresses, with the
        // block lines prefetched. Splitting it from finish() lets a caller
        // issue the next batch's fetches before testing the current batch.
        void issue(ProbeBatch& b) const {
            for (u32 i = 0; i < b.count; ++i) {
                const u64 part = (b.keys[i] >> PARTITION_SHIFT) - pBegin;
                b.h[i] = bloomHash(b.keys[i]);
                b.blocks[i] = bloom.data()
                              + (part * meta.bloomBlocksPerPartition
                                 + bloomBlockInPartition(b.h[i], meta.bloomBlocksPerPartition)) * 8;
                __builtin_prefetch(b.blocks[i]);
            }
        }

        // Stages 2-4: Bloom test, then bucket offsets, then fingerprints; each
        // stage prefetches the next stage's lines for the whole batch. Returns
        // the number of probes that passed the Bloom filter.
        u32 finish(const ProbeBatch& b, std::vector<KeyedState>& candidates) const {
            const u64 bucketsPer = meta.bucketsPerPartition();
            const u64 dirPer = bucketsPer + 1;
            const u32 shift = meta.bucketShift();
            const u64* keys = b.keys;
            const B1B2* probes = b.states;
            const u32 count = b.count;

            u32 pass[MAX_PROBE_GROUP];
            u32 passCount = 0;
            for (u32 i = 0; i < count; ++i) {
                if (bloomTest(b.blocks[i], b.h[i])) {
                    pass[passCount++] = i;
                }
            }

            const u32* dirSlot[MAX_PROBE_GROUP];
            u64 base[MAX_PROBE_GROUP], lo[MAX_PROBE_GROUP], hi[MAX_PROBE_GROUP];
            for (u32 j = 0; j < passCount; ++j) {
                const u64 key = keys[pass[j]];
                const u64 part = (key >> PARTITION_SHIFT) - pBegin;
                base[j] = partitionFirst[part];
                dirSlot[j] = &starts[part * dirPer + ((key >> shift) & (bucketsPer - 1))];
                __builtin_prefetch(dirSlot[j]);
            }
            for (u32 j = 0; j < passCount; ++j) {
                lo[j] = base[j] + dirSlot[j][0];
                hi[j] = base[j] + dirSlot[j][1];
                if (lo[j] < hi[j]) {
                    __builtin_prefetch(&fingerprints[lo[j]]);
                }
            }
            for (u32 j = 0; j < passCount; ++j) {
                const u32 i = pass[j];
                const u16 fp = static_cast<u16>(keys[i]);
                for (u64 k = lo[j]; k < hi[j]; ++k) {
                    if (fingerprints[k] == fp) {
                        candidates.push_back({keys[i], probes[i]});
                        break;
                    }
                }
            }
            return passCount;
        }

        // Unpipelined probe of up to MAX_PROBE_GROUP states (issue + finish).
        u32 probeGroup(const B1B2* probes, const u64* keys, const u32 count,
                       std::vector<KeyedState>& candidates) const {
            ProbeBatch b;
            b.count = count;
            for (u32 i = 0; i < count; ++i) {
                b.states[i] = probes[i];
                b.keys[i] = keys[i];
            }
            issue(b);
            return finish(b, candidates);
        }
    };

    // ------------------------------------------------------------------
    // Probe variants and per-machine calibration
    // ------------------------------------------------------------------
    struct ProbeVariant {
        u32 group = PROBE_GROUP;
        bool pipeline = false;

        MUND std::string name() const { return std::to_string(group) + (pipeline ? "/pipelined" : "/plain"); }
    };

    const std::vector<ProbeVariant>& probeVariants() {
        static const std::vector<ProbeVariant> v = {{32, false}, {32, true}, {64, false}, {64, true},
                                                    {128, false}, {128, true}};
        return v;
    }

    // Collects in-range left states into batches and probes them with the
    // chosen variant; pipelined batches overlap their fetches with the
    // previous batch's tests.
    // LEAF_BOUND: in-range states whose colour-mix bound exceeds `limit`
    // cannot be within the table depth of the right root and are not probed.
    template<bool LEAF_BOUND, bool LEAF_PDB = false>
    struct ProbeSinkT {
        const FingerprintRange& index;
        ProbeVariant variant;
        ProbeBatch batches[2];
        u32 cur = 0;
        bool prevPending = false;
        std::vector<KeyedState> found;
        u64 streamed = 0, inRange = 0, bloomPasses = 0, boundSkipped = 0;
        const colour_mix::LaneTables* tables = nullptr;
        int limit = 0;

        ProbeSinkT(const FingerprintRange& idx, const ProbeVariant v) : index(idx), variant(v) {}
        ProbeSinkT(const FingerprintRange& idx, const ProbeVariant v, const colour_mix::LaneTables* tabs, const int lim)
            : index(idx), variant(v), tables(tabs), limit(lim) {}

        void submit() {
            ProbeBatch& c = batches[cur];
            index.issue(c);
            if (!variant.pipeline) {
                bloomPasses += index.finish(c, found);
                c.count = 0;
                return;
            }
            if (prevPending) {
                ProbeBatch& prev = batches[cur ^ 1];
                bloomPasses += index.finish(prev, found);
                prev.count = 0;
            }
            prevPending = true;
            cur ^= 1;
        }

        void drain() {
            if (batches[cur].count != 0) {
                submit();
            }
            if (prevPending) {
                ProbeBatch& prev = batches[cur ^ 1];
                bloomPasses += index.finish(prev, found);
                prev.count = 0;
                prevPending = false;
            }
        }

        void operator()(JVec<B1B2>& chunkStates, JVec<u64>& chunkHashes, const u32 count) {
            streamed += count;
            for (u32 i = 0; i < count; ++i) {
                const u64 key = mixKey(chunkHashes[i]);
                const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                if (p < index.pBegin || p >= index.pEnd) {
                    continue;
                }
                ++inRange;
                if constexpr (LEAF_BOUND) {
                    if (colour_mix::bound(chunkStates[i], *tables) > limit) {
                        ++boundSkipped;
                        continue;
                    }
                }
                if constexpr (LEAF_PDB) {
                    bool isFar = false;
                    for (u32 k = 0; k < group_pdb::g_tableCount && !isFar; ++k) {
                        isFar = group_pdb::g_tables[k]->farLeaf(chunkStates[i]);
                    }
                    if (isFar) {
                        ++boundSkipped;
                        continue;
                    }
                }
                ProbeBatch& b = batches[cur];
                b.states[b.count] = chunkStates[i];
                b.keys[b.count] = key;
                if (++b.count == variant.group) {
                    submit();
                }
            }
        }
    };

    // "<cpu brand>|<RAM GB>|<threads>": a calibration result belongs to one machine setup.
    std::string machineKey(const u32 threads) {
        char brand[49] = {};
        unsigned int regs[4];
        for (unsigned int leaf = 0; leaf < 3; ++leaf) {
            if (__get_cpuid(0x80000002 + leaf, &regs[0], &regs[1], &regs[2], &regs[3])) {
                std::memcpy(brand + leaf * 16, regs, 16);
            }
        }
        std::string cpu = brand;
        while (!cpu.empty() && cpu.front() == ' ') cpu.erase(cpu.begin());
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        const u64 ramGb = (status.ullTotalPhys + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return cpu + "|" + std::to_string(ramGb) + "GB|" + std::to_string(threads) + "t";
    }

    fs::path probeProfilePath() {
        const char* local = std::getenv("LOCALAPPDATA");
        return fs::path(local != nullptr ? local : ".") / "MindbenderSolver" / "probe_profile.txt";
    }

    // Line format: "<machine key>\t<group>\t<pipeline>\t<ns/state best>\t<ns/state baseline>"
    bool readCachedVariant(const std::string& key, ProbeVariant& out) {
        std::ifstream in(probeProfilePath());
        std::string line;
        while (std::getline(in, line)) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos || line.substr(0, tab) != key) {
                continue;
            }
            std::istringstream rest(line.substr(tab + 1));
            int g = 0, pl = 0;
            if (rest >> g >> pl && g > 0 && g <= static_cast<int>(MAX_PROBE_GROUP)) {
                out = {static_cast<u32>(g), pl != 0};
                return true;
            }
        }
        return false;
    }

    void writeCachedVariant(const std::string& key, const ProbeVariant& v, const double bestNs, const double baseNs) {
        const fs::path path = probeProfilePath();
        std::vector<std::string> lines;
        {
            std::ifstream in(path);
            std::string line;
            while (std::getline(in, line)) {
                if (line.rfind(key + "\t", 0) != 0 && !line.empty()) {
                    lines.push_back(line);
                }
            }
        }
        std::ostringstream entry;
        entry << key << '\t' << v.group << '\t' << (v.pipeline ? 1 : 0) << '\t' << bestNs << '\t' << baseNs;
        lines.push_back(entry.str());
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        std::ofstream out(path, std::ios::trunc);
        for (const std::string& l : lines) {
            out << l << '\n';
        }
    }

    // RAM for the index (or the depth-7 build's key buffers), measured on this
    // machine when the run starts: free physical memory minus a reserve for
    // the OS and other programs (max(3 GB, 10% of RAM)) and minus `otherBytes`
    // this run will also allocate (seeds, candidate buffers). At least 2 GB.
    // (Previously a flat 60% of free memory, which left ~8 GB idle and cost
    // whole extra passes, e.g. 12-1's 13.1 GB index ran in 2 passes.)
    // --ram-gb overrides it.
    u64 defaultRamBudgetBytes(const u64 otherBytes = 0) {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        const u64 reserve = std::max<u64>(3'000'000'000ULL, status.ullTotalPhys / 10);
        const u64 used = reserve + otherBytes;
        const u64 avail = status.ullAvailPhys > used ? status.ullAvailPhys - used : 0;
        return std::max<u64>(2'000'000'000ULL, avail);
    }

    // ------------------------------------------------------------------
    // Colour-mix bound: audit on this run's own seeds
    // ------------------------------------------------------------------
    struct BoundAudit {
        bool ok = true;
        std::string why;
        u64 nodes = 0, leaves = 0, edges = 0;
        double leafSkip = 0.0, parentSkip = 0.0;   // sampled shares
        u64 streamChecks = 0;
    };

    template<typename Fn>
    struct StateCallbackSink {
        Fn& fn;
        void operator()(JVec<B1B2>& states, JVec<u64>&, const u32 count) {
            for (u32 i = 0; i < count; ++i) {
                fn(states[i]);
            }
        }
    };

    template<i32 DEPTH, typename PRUNE, typename Fn>
    void streamStates(const B1B2& root, Fn&& fn) {
        StateCallbackSink<Fn> sink{fn};
        if constexpr (std::is_same_v<PRUNE, perm_stream_detail::NoParentPrune>) {
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, DEPTH>(makeBoardFromState(root), sink, 4096);
        } else {
            PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, DEPTH, PRUNE>(makeBoardFromState(root), sink, 4096);
        }
    }

    // On ~256 seeds spread over the seed list: (1) the fast lane tables give
    // the reference bound on every node and leaf; (2) on every sampled edge
    // (node -> child) h changes by at most 1, which the parent prune relies
    // on; (3) on 32 seeds the pruned stream keeps exactly the leaves with
    // h <= limit that the full stream has. Also measures the skip shares.
    template<i32 STREAM_DEPTH>
    BoundAudit auditColourMixBound(const JVec<B1B2>& seeds, const colour_mix::LaneTables& tabs, const int limit,
                                   const u32 threads) {
        BoundAudit a;
        const std::size_t samples = std::min<std::size_t>(256, seeds.size());
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, samples));
        std::atomic<u64> nodes{0}, leaves{0}, edges{0}, nodesSkip{0}, leavesSkip{0}, leavesUnderSkip{0}, bad{0};
        std::mutex whyLock;
        std::string firstWhy;
        auto fail = [&](const std::string& why) {
            bad.fetch_add(1);
            std::lock_guard<std::mutex> lock(whyLock);
            if (firstWhy.empty()) {
                firstWhy = why;
            }
        };
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            const B1B2& seed = seeds[k * step];
            // nodes one move above the leaves of the STREAM_DEPTH stream
            streamStates<STREAM_DEPTH - 1, perm_stream_detail::NoParentPrune>(seed, [&](const B1B2& node) {
                const int hn = colour_mix::bound(node, tabs);
                if (hn != colour_mix::referenceBound(node, g_rightRoot, tabs.kind)) {
                    fail("fast bound != reference on a node");
                }
                nodes.fetch_add(1, std::memory_order_relaxed);
                const bool skipNode = hn > limit + 1;
                nodesSkip.fetch_add(skipNode, std::memory_order_relaxed);
                streamStates<1, perm_stream_detail::NoParentPrune>(node, [&](const B1B2& leaf) {
                    const int hl = colour_mix::bound(leaf, tabs);
                    if (hl != colour_mix::referenceBound(leaf, g_rightRoot, tabs.kind)) {
                        fail("fast bound != reference on a leaf");
                    }
                    if (hl > hn + 1 || hl < hn - 1) {
                        fail("bound changed by more than 1 in one move");
                    }
                    leaves.fetch_add(1, std::memory_order_relaxed);
                    edges.fetch_add(1, std::memory_order_relaxed);
                    leavesSkip.fetch_add(hl > limit, std::memory_order_relaxed);
                    leavesUnderSkip.fetch_add(skipNode, std::memory_order_relaxed);
                });
            });
        });
        // (3) pruned vs full stream, wanted leaves identical
        const std::size_t streamSeeds = std::min<std::size_t>(32, seeds.size());
        const std::size_t sstep = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, streamSeeds));
        colour_mix::g_parentTables = &tabs;
        colour_mix::g_parentLimit = limit;
        std::atomic<u64> checks{0};
        parallelFor(threads, (seeds.size() + sstep - 1) / sstep, [&](u32, const std::size_t k) {
            std::vector<B1B2> full, pruned;
            streamStates<STREAM_DEPTH, perm_stream_detail::NoParentPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (colour_mix::bound(s, tabs) <= limit) {
                    full.push_back(s);
                }
            });
            streamStates<STREAM_DEPTH, colour_mix::ParentPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (colour_mix::bound(s, tabs) <= limit) {
                    pruned.push_back(s);
                }
            });
            std::sort(full.begin(), full.end());
            std::sort(pruned.begin(), pruned.end());
            if (full != pruned) {
                fail("pruned stream lost or added a wanted leaf");
            }
            checks.fetch_add(full.size(), std::memory_order_relaxed);
        });
        a.nodes = nodes.load();
        a.leaves = leaves.load();
        a.edges = edges.load();
        a.streamChecks = checks.load();
        a.leafSkip = static_cast<double>(leavesSkip.load()) / static_cast<double>(std::max<u64>(1, a.leaves));
        a.parentSkip = static_cast<double>(leavesUnderSkip.load()) / static_cast<double>(std::max<u64>(1, a.leaves));
        a.ok = bad.load() == 0;
        a.why = firstWhy;
        return a;
    }

    // Grouped pattern databases on this run's seeds: zero at the target; on
    // sampled real edges, every child of a "> limit + 1" node is "> limit";
    // the pruned stream (colour-mix + tables at parents) keeps exactly the
    // leaves the full stream has that neither bound rules out. Also measures
    // the extra share of whole last levels the tables skip.
    template<i32 STREAM_DEPTH>
    BoundAudit auditGroupPdb(const JVec<B1B2>& seeds, const colour_mix::LaneTables& tabs, const int limit,
                             const u32 threads) {
        BoundAudit a;
        std::atomic<u64> nodes{0}, edges{0}, nodeSkipMix{0}, nodeSkipAny{0}, bad{0}, checks{0};
        std::mutex whyLock;
        std::string firstWhy;
        auto fail = [&](const std::string& why) {
            bad.fetch_add(1);
            std::lock_guard<std::mutex> lock(whyLock);
            if (firstWhy.empty()) {
                firstWhy = why;
            }
        };
        for (u32 i = 0; i < group_pdb::g_tableCount; ++i) {
            if (group_pdb::g_tables[i]->code(g_rightRoot) != 0) {
                fail("table " + group_pdb::g_tables[i]->name() + " is not 0 at the target");
            }
        }
        auto anyFar = [](const B1B2& s, const bool node) {
            for (u32 i = 0; i < group_pdb::g_tableCount; ++i) {
                if (node ? group_pdb::g_tables[i]->farNode(s) : group_pdb::g_tables[i]->farLeaf(s)) {
                    return true;
                }
            }
            return false;
        };
        const std::size_t samples = std::min<std::size_t>(256, seeds.size());
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, samples));
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            streamStates<STREAM_DEPTH - 1, perm_stream_detail::NoParentPrune>(seeds[k * step], [&](const B1B2& node) {
                nodes.fetch_add(1, std::memory_order_relaxed);
                const bool mixFar = colour_mix::bound(node, tabs) > limit + 1;
                const bool pdbFar = anyFar(node, true);
                nodeSkipMix.fetch_add(mixFar, std::memory_order_relaxed);
                nodeSkipAny.fetch_add(mixFar || pdbFar, std::memory_order_relaxed);
                if (pdbFar) {
                    streamStates<1, perm_stream_detail::NoParentPrune>(node, [&](const B1B2& leaf) {
                        edges.fetch_add(1, std::memory_order_relaxed);
                        if (!anyFar(leaf, false)) {
                            fail("a far node has a child within the limit");
                        }
                    });
                }
            });
        });
        const std::size_t streamSeeds = std::min<std::size_t>(32, seeds.size());
        const std::size_t sstep = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, streamSeeds));
        parallelFor(threads, (seeds.size() + sstep - 1) / sstep, [&](u32, const std::size_t k) {
            std::vector<B1B2> full, pruned;
            auto wanted = [&](const B1B2& s) { return colour_mix::bound(s, tabs) <= limit && !anyFar(s, false); };
            streamStates<STREAM_DEPTH, perm_stream_detail::NoParentPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (wanted(s)) {
                    full.push_back(s);
                }
            });
            streamStates<STREAM_DEPTH, ParentPrunePdb>(seeds[k * sstep], [&](const B1B2& s) {
                if (wanted(s)) {
                    pruned.push_back(s);
                }
            });
            std::sort(full.begin(), full.end());
            std::sort(pruned.begin(), pruned.end());
            if (full != pruned) {
                fail("pruned stream lost or added a wanted leaf");
            }
            checks.fetch_add(full.size(), std::memory_order_relaxed);
        });
        a.nodes = nodes.load();
        a.edges = edges.load();
        a.streamChecks = checks.load();
        a.parentSkip = static_cast<double>(nodeSkipAny.load()) / static_cast<double>(std::max<u64>(1, a.nodes));
        a.leafSkip = static_cast<double>(nodeSkipMix.load()) / static_cast<double>(std::max<u64>(1, a.nodes));
        a.ok = bad.load() == 0;
        a.why = firstWhy;
        return a;
    }

    // Streams the whole left side once per RAM-sized range of the index's
    // partitions and returns the unique fingerprint hits (key, state), sorted.
    // `builder`, if given, fills the index for a pass in RAM instead of
    // loading it from `files` (no-table mode); `counts` are then estimates.
    using IndexBuilder = std::function<void(FingerprintRange&, u32, u32)>;

    template<i32 STREAM_DEPTH>
    std::vector<KeyedState> streamAndProbeT(const Board& leftRoot,
                                            const IndexFiles& files,
                                            const FingerprintMeta& meta,
                                            const std::vector<u64>& counts,
                                            const Config& cfg,
                                            const u32 threads,
                                            const IndexBuilder* builder,
                                            const u64 builderReserveBytes) {
        // Partition ranges whose index fits the RAM budget.
        const u32 pFirst = meta.partitionBegin, pLimit = meta.partitionLimit;
        // Seeds: exact depth-S frontier, ~ (25.4^S) states of 16 B (+ hashes while built).
        const u64 seedBytes = static_cast<u64>(std::pow(25.4, cfg.seedDepth) * 2 * sizeof(B1B2));
        const u64 fullBudget = cfg.ramBytes != 0 ? cfg.ramBytes : defaultRamBudgetBytes(seedBytes);
        const u64 budget = fullBudget > builderReserveBytes + 2'000'000'000ULL ? fullBudget - builderReserveBytes
                                                                                : 2'000'000'000ULL;
        std::vector<std::pair<u32, u32>> passes;
        u64 coveredStates = 0;
        for (u32 p = pFirst; p < pLimit;) {
            u32 q = p;
            u64 states = 0;
            while (q < pLimit
                   && (q == p || meta.indexBytes(states + counts[q], q + 1 - p) <= budget)) {
                states += counts[q++];
            }
            coveredStates += states;
            passes.emplace_back(p, q);
            p = q;
        }
        tcout << "fingerprint index: " << meta.indexBytes(coveredStates, pLimit - pFirst) / 1e9
              << " GB over partitions [" << pFirst << "," << pLimit << "), RAM budget " << budget / 1e9 << " GB -> "
              << passes.size() << " pass(es)\n";

        // One index buffer for every pass, sized for the largest pass and
        // allocated before anything else.
        FingerprintRange index;
        {
            u64 maxStates = 0, maxParts = 0;
            for (const auto& [p0, p1] : passes) {
                u64 states = 0;
                for (u32 p = p0; p < p1; ++p) {
                    states += counts[p];
                }
                maxStates = std::max(maxStates, states);
                maxParts = std::max<u64>(maxParts, p1 - p0);
            }
            if (builder == nullptr) {
                index.reserve(meta, maxStates, maxParts);
            }
        }

        Timer seedTimer;
        JVec<B1B2> seeds;
        buildSeeds(leftRoot, cfg.seedDepth, seeds);
        tcout << "left seeds (exact depth " << cfg.seedDepth << "): " << seeds.size()
              << " in " << seedTimer.getSeconds() << "s\n";
        if (cfg.shardCount > 1) {
            // Interleaved, so every shard gets a similar mix of seeds.
            const std::size_t all = seeds.size();
            std::size_t w = 0;
            for (std::size_t s = cfg.shardIndex; s < all; s += cfg.shardCount) {
                seeds[w++] = seeds[s];
            }
            seeds.resize(w);
            tcout << "shard " << cfg.shardIndex << "/" << cfg.shardCount << ": " << seeds.size() << " of " << all
                  << " seeds (index = " << cfg.shardIndex << " mod " << cfg.shardCount << ")\n";
        }

        // Colour-mix bound: eligible modes after the audit (see header).
        // boundMode: 0 off, 1 leaf, 2 leaf + parent; -1 = decide by timing.
        static colour_mix::LaneTables boundTables;   // read by colour_mix::ParentPrune
        const int boundLimit = static_cast<int>(cfg.tableDepth);
        int boundMode = 0;
        std::vector<int> boundModes;   // modes to time (auto)
        static const char* BOUND_NAMES[7] = {"off", "leaf", "both", "pdb", "pdbleaf", "parent", "pdbparent"};
        if (cfg.boundMode != 0 && !cfg.reverse) {
            if (g_rightRoot.getFatBool()) {
                tcout << "bound: off (2x2 board; the halved bound prunes nothing)\n";
            } else {
                Timer auditTimer;
                colour_mix::buildLaneTables(g_rightRoot, boundTables, cfg.boundKind);
                const BoundAudit audit = auditColourMixBound<STREAM_DEPTH>(seeds, boundTables, boundLimit, threads);
                tcout << "bound (" << colour_mix::kindName(cfg.boundKind) << "): audit " << (audit.ok ? "ok" : "FAILED (" + audit.why + ")") << ": " << audit.nodes
                      << " nodes, " << audit.leaves << " leaves, " << audit.edges << " edges (|dh| <= 1), "
                      << audit.streamChecks << " wanted leaves identical pruned vs full; sampled skip: leaf "
                      << 100.0 * audit.leafSkip << "%, whole last level " << 100.0 * audit.parentSkip << "%; tables "
                      << boundTables.bytes << " B, " << auditTimer.getSeconds() << "s\n";
                if (audit.ok) {
                    colour_mix::g_parentTables = &boundTables;
                    colour_mix::g_parentLimit = boundLimit;
                    bool pdbReady = false;
                    const bool wantPdb = cfg.pdbMode == 1
                                         || (cfg.pdbMode == -1 && cfg.seedDepth >= 5 && audit.parentSkip < 0.9);
                    if (wantPdb) {
                        static std::vector<group_pdb::Table> pdbTables;
                        pdbTables.clear();
                        Timer pdbTimer;
                        for (const u8 mask : group_pdb::chooseClasses(g_rightRoot, cfg.pdbMaxEntries, 3)) {
                            pdbTables.push_back(group_pdb::build(g_rightRoot, mask, boundLimit, threads));
                            const group_pdb::Table& tb = pdbTables.back();
                            tcout << "pdb: table " << tb.name() << ": " << tb.cells << " cells, " << tb.entries / 1e6
                                  << " M entries (" << tb.bits.size() * 8 / 1e6 << " MB), max distance " << tb.maxDist
                                  << ", built in " << tb.buildSeconds << "s\n";
                        }
                        group_pdb::g_tableCount = static_cast<u32>(pdbTables.size());
                        for (u32 i = 0; i < group_pdb::g_tableCount; ++i) {
                            group_pdb::g_tables[i] = &pdbTables[i];
                        }
                        if (group_pdb::g_tableCount != 0) {
                            const BoundAudit pa = auditGroupPdb<STREAM_DEPTH>(seeds, boundTables, boundLimit, threads);
                            tcout << "pdb: audit " << (pa.ok ? "ok" : "FAILED (" + pa.why + ")") << ": " << pa.nodes
                                  << " nodes, " << pa.edges << " edges below far nodes, " << pa.streamChecks
                                  << " wanted leaves identical pruned vs full; whole last level skipped "
                                  << 100.0 * pa.leafSkip << "% -> " << 100.0 * pa.parentSkip << "% with the tables ("
                                  << pdbTimer.getSeconds() << "s total)\n";
                            pdbReady = pa.ok;
                            if (!pa.ok) {
                                group_pdb::g_tableCount = 0;
                            }
                        } else {
                            tcout << "pdb: no colour class fits " << cfg.pdbMaxEntries / 1000000 << " M entries\n";
                        }
                    }
                    if (cfg.boundMode > 0) {
                        const bool needsPdb = cfg.boundMode == 3 || cfg.boundMode == 4 || cfg.boundMode == 6;
                        boundMode = needsPdb && !pdbReady ? 2 : cfg.boundMode;
                        tcout << "bound: " << BOUND_NAMES[boundMode] << " (forced)\n";
                    } else if (audit.leafSkip >= 0.05 || pdbReady) {
                        boundMode = -1;
                        boundModes = {0, 1};
                        if (audit.parentSkip >= 0.01) {
                            boundModes.push_back(2);
                            boundModes.push_back(5);
                        }
                        if (pdbReady) {
                            boundModes.push_back(3);
                            boundModes.push_back(4);
                            boundModes.push_back(6);
                        }
                    } else {
                        tcout << "bound: off (skips under 5% here)\n";
                    }
                }
            }
        }

        std::vector<KeyedState> candidates;
        double loadSeconds = 0.0, probeSeconds = 0.0;
        u64 streamedTotal = 0, inRangeTotal = 0, bloomPassTotal = 0, boundSkippedTotal = 0;
        g_progress.phase("probe", static_cast<u64>(passes.size()) * seeds.size(),
                         "pass 1/" + std::to_string(passes.size()) + " loading index");
        ProbeVariant chosenVariant{0, false};   // group 0: not chosen yet
        if (cfg.forcedGroup > 0) {
            chosenVariant = {static_cast<u32>(std::min<int>(cfg.forcedGroup, MAX_PROBE_GROUP)), cfg.forcedPipeline != 0};
            tcout << "probe variant: " << chosenVariant.name() << " (forced)\n";
        }

        // Visiting order: chunks of SEED_CHUNK consecutive seeds, taken with a
        // large stride (a bijection on the chunks), so any stretch of the order,
        // in particular the first one that the calibrations time, is spread over
        // the whole seed list. The list is sorted, similar boards side by side,
        // and how much the bounds prune differs a lot between its regions: on
        // 16-4 the first 1% of seeds timed pdbparent at 1.11x vs off, full runs
        // 1.45-1.5x. Every seed is still visited exactly once; the order does not
        // affect the results (candidates are sorted afterwards).
        constexpr std::size_t SEED_CHUNK = 4096;
        const std::size_t seedChunks = std::max<std::size_t>(1, (seeds.size() + SEED_CHUNK - 1) / SEED_CHUNK);
        std::size_t chunkStride = static_cast<std::size_t>(0.6180339887 * static_cast<double>(seedChunks)) | 1;
        while (std::gcd(chunkStride, seedChunks) != 1) {
            ++chunkStride;
        }
        const std::size_t seedPositions = seedChunks * SEED_CHUNK;   // positions >= seeds.size() are skipped
        auto seedAt = [&](const std::size_t pos) {
            return (pos / SEED_CHUNK) * chunkStride % seedChunks * SEED_CHUNK + pos % SEED_CHUNK;
        };

        for (std::size_t pass = 0; pass < passes.size(); ++pass) {
            const auto [pBegin, pEnd] = passes[pass];
            const std::string passLabel = "pass " + std::to_string(pass + 1) + "/" + std::to_string(passes.size());
            g_progress.detail(passLabel + " loading index");

            Timer loadTimer;
            if (builder != nullptr) {
                g_progress.detail(passLabel + " building index in RAM");
                (*builder)(index, pBegin, pEnd);
            } else {
                index.load(files, meta, counts, pBegin, pEnd);
            }
            const double passLoad = loadTimer.getSeconds();
            loadSeconds += passLoad;
            g_progress.detail(passLabel + " streaming");

            Timer probeTimer;
            std::mutex candidateLock;
            std::vector<u64> streamed(threads, 0), inRange(threads, 0), bloomPasses(threads, 0), boundSkipped(threads, 0);
            std::atomic<u64> visited = 0;   // seeds streamed this pass: must be every seed, once
            auto finishSeed = [&](const u32 t, auto& sink) -> u64 {
                sink.drain();
                visited.fetch_add(1, std::memory_order_relaxed);
                streamed[t] += sink.streamed;
                inRange[t] += sink.inRange;
                bloomPasses[t] += sink.bloomPasses;
                boundSkipped[t] += sink.boundSkipped;
                if (!sink.found.empty()) {
                    std::lock_guard<std::mutex> lock(candidateLock);
                    candidates.insert(candidates.end(), sink.found.begin(), sink.found.end());
                }
                g_progress.add();   // one relaxed atomic add per seed (~ms of work)
                return sink.streamed;
            };
            auto processSeed = [&](const u32 t, const std::size_t pos, const ProbeVariant variant,
                                   const int mode = 0) -> u64 {
                const std::size_t s = seedAt(pos);   // pos: place in the visiting order
                if (s >= seeds.size()) {
                    return 0;
                }
                const Board seedBoard = makeBoardFromState(seeds[s]);
                if (mode == 0 || mode == 5 || mode == 6) {
                    ProbeSinkT<false> sink(index, variant);
                    if (mode == 5) {
                        PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, colour_mix::ParentPrune>(
                                seedBoard, sink, 4096);
                    } else if (mode == 6) {
                        PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, ParentPrunePdb>(
                                seedBoard, sink, 4096);
                    } else {
                        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, STREAM_DEPTH>(seedBoard, sink, 4096);
                    }
                    return finishSeed(t, sink);
                }
                if (mode == 4) {
                    ProbeSinkT<true, true> sink(index, variant, &boundTables, boundLimit);
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, ParentPrunePdb>(
                            seedBoard, sink, 4096);
                    return finishSeed(t, sink);
                }
                ProbeSinkT<true> sink(index, variant, &boundTables, boundLimit);
                if (mode == 3) {
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, ParentPrunePdb>(
                            seedBoard, sink, 4096);
                } else if (mode == 2) {
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, colour_mix::ParentPrune>(
                            seedBoard, sink, 4096);
                } else {
                    PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, STREAM_DEPTH>(seedBoard, sink, 4096);
                }
                return finishSeed(t, sink);
            };

            std::size_t firstSeed = 0;   // position in the visiting order (seedAt)
            if (pass == 0 && chosenVariant.group == 0) {
                const auto& variants = probeVariants();
                const std::size_t nV = variants.size();
                const std::size_t round = static_cast<std::size_t>(threads) * nV * 4;
                if (cfg.calibrate && seedPositions >= round * 40) {
                    // Real seeds, interleaved over all variants until the time is up.
                    g_progress.detail(passLabel + " calibrating probe variants");
                    std::vector<std::atomic<u64>> vStates(nV), vNanos(nV);
                    for (std::size_t v = 0; v < nV; ++v) {
                        vStates[v].store(0);
                        vNanos[v].store(0);
                    }
                    const Timer calibTimer;
                    while (firstSeed + round <= seedPositions / 4 && calibTimer.getSeconds() < cfg.calibrateSeconds) {
                        const std::size_t base = firstSeed;
                        parallelFor(threads, round, [&](const u32 t, const std::size_t i) {
                            const std::size_t v = i % nV;
                            const auto t0 = std::chrono::steady_clock::now();
                            const u64 n = processSeed(t, base + i, variants[v]);
                            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - t0).count();
                            vStates[v].fetch_add(n, std::memory_order_relaxed);
                            vNanos[v].fetch_add(static_cast<u64>(ns), std::memory_order_relaxed);
                        });
                        firstSeed += round;
                    }
                    std::size_t best = 0;
                    std::vector<double> nsPer(nV);
                    for (std::size_t v = 0; v < nV; ++v) {
                        nsPer[v] = static_cast<double>(vNanos[v].load()) / static_cast<double>(std::max<u64>(1, vStates[v].load()));
                        if (nsPer[v] < nsPer[best]) {
                            best = v;
                        }
                    }
                    chosenVariant = variants[best];
                    tcout << "calibrate: " << firstSeed << " seeds in " << calibTimer.getSeconds() << "s, ns/state per thread:";
                    for (std::size_t v = 0; v < nV; ++v) {
                        tcout << ' ' << variants[v].name() << '=' << nsPer[v];
                    }
                    tcout << " -> " << chosenVariant.name() << " (" << nsPer[0] / nsPer[best] << "x vs 32/plain)\n";
                    writeCachedVariant(machineKey(threads), chosenVariant, nsPer[best], nsPer[0]);
                } else {
                    ProbeVariant cached;
                    chosenVariant = readCachedVariant(machineKey(threads), cached) ? cached : ProbeVariant{};
                    tcout << "probe variant: " << chosenVariant.name()
                          << (cfg.calibrate ? " (cached; too few seeds to calibrate)" : " (cached, --no-calibrate)") << '\n';
                }
                g_progress.detail(passLabel + " streaming");
            }
            if (boundMode == -1) {
                // Time the eligible modes on real seeds (their work counts;
                // nothing is redone), the way the run will use them: each block
                // runs ONE mode on every thread. Timing the modes seed by seed side
                // by side let their cache and memory traffic disturb each other
                // (16-4: pdbparent timed at 1.0-1.1x vs 1.5x in full runs).
                // A round takes the next chunk of seeds and gives each mode every
                // k-th of them (k = modes still timed), one mode after another in a
                // rotating order. Neighbouring seeds cost about the same, so the
                // modes are compared round by round (log of the time ratio), which
                // cancels both seed-to-seed cost and changes in machine load. After
                // MIN_ROUNDS a mode clearly slower than the best is dropped; timing
                // goes on until the rest are told apart (2 standard errors) or known
                // to ~1%, at least MIN_SECONDS, within a budget of 3% of this pass's
                // estimated search time (20 s to --bound-calibrate-seconds).
                const std::size_t nM = boundModes.size();
                const std::size_t block = static_cast<std::size_t>(threads) * 32;
                constexpr int MIN_ROUNDS = 4;
                constexpr double MIN_SECONDS = 5.0;
                if (cfg.calibrate && seedPositions - firstSeed >= block * nM * MIN_ROUNDS * 4) {
                    g_progress.detail(passLabel + " timing the lower-bound modes");
                    // perRound[m][r]: mean ns per seed of mode m in round r (< 0: not timed)
                    std::vector<std::vector<double>> perRound(nM);
                    std::vector<int> droppedAt(nM, 0);
                    auto alive = [&](const std::size_t m) { return droppedAt[m] == 0; };
                    // mean and standard error of log(time m / time b) over rounds timing both
                    auto logRatio = [&](const std::size_t m, const std::size_t b) {
                        std::vector<double> d;
                        for (std::size_t r = 0; r < perRound[m].size() && r < perRound[b].size(); ++r) {
                            if (perRound[m][r] > 0.0 && perRound[b][r] > 0.0) {
                                d.push_back(std::log(perRound[m][r] / perRound[b][r]));
                            }
                        }
                        double mu = 0.0, ss = 0.0;
                        for (const double v : d) {
                            mu += v;
                        }
                        mu /= static_cast<double>(std::max<std::size_t>(1, d.size()));
                        for (const double v : d) {
                            ss += (v - mu) * (v - mu);
                        }
                        const double n = static_cast<double>(d.size());
                        return std::pair<double, double>{mu, d.size() < 2 ? 1e300 : std::sqrt(ss / (n - 1.0) / n)};
                    };
                    auto meanOf = [&](const std::size_t m) {
                        double s = 0.0;
                        std::size_t n = 0;
                        for (const double v : perRound[m]) {
                            if (v > 0.0) {
                                s += v;
                                ++n;
                            }
                        }
                        return s / static_cast<double>(std::max<std::size_t>(1, n));
                    };
                    const Timer boundTimer;
                    const std::size_t limitSeed = firstSeed + (seedPositions - firstSeed) / 4;
                    double budget = cfg.boundCalibrateSeconds;
                    int rounds = 0;
                    std::size_t timedSeeds = 0;
                    const char* why = "budget";
                    std::size_t best = 0;
                    while (true) {
                        std::vector<std::size_t> live;
                        for (std::size_t m = 0; m < nM; ++m) {
                            if (alive(m)) {
                                live.push_back(m);
                            }
                        }
                        const std::size_t k = live.size();
                        if (firstSeed + block * k > limitSeed) {
                            why = "seed limit";
                            break;
                        }
                        const std::size_t base = firstSeed;
                        for (std::size_t m = 0; m < nM; ++m) {
                            perRound[m].push_back(-1.0);
                        }
                        for (std::size_t j = 0; j < k; ++j) {
                            const std::size_t slot = (j + static_cast<std::size_t>(rounds)) % k;
                            const std::size_t m = live[slot];
                            std::atomic<u64> nanos = 0;
                            parallelFor(threads, block, [&](const u32 t, const std::size_t i) {
                                const auto t0 = std::chrono::steady_clock::now();
                                processSeed(t, base + i * k + slot, chosenVariant, boundModes[m]);
                                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - t0).count();
                                nanos.fetch_add(static_cast<u64>(ns), std::memory_order_relaxed);
                            });
                            perRound[m].back() = static_cast<double>(nanos.load()) / static_cast<double>(block);
                        }
                        firstSeed += block * k;
                        timedSeeds += block * k;
                        ++rounds;

                        // best = the lowest geometric-mean time among the modes still timed
                        best = live[0];
                        for (const std::size_t m : live) {
                            if (m != best && logRatio(m, best).first < 0.0) {
                                best = m;
                            }
                        }
                        if (rounds < MIN_ROUNDS) {
                            continue;
                        }
                        bool settled = true;
                        std::size_t others = 0;
                        for (const std::size_t m : live) {
                            if (m == best) {
                                continue;
                            }
                            const auto [mu, se] = logRatio(m, best);
                            if (mu > std::log(1.05) && mu > 3.0 * se) {
                                droppedAt[m] = rounds;
                                continue;
                            }
                            ++others;
                            if (mu <= 2.0 * se && se > 0.01) {
                                settled = false;   // not told apart yet, and not known to ~1%
                            }
                        }
                        const double passEstimate = meanOf(best) * static_cast<double>(seeds.size()) / threads / 1e9;
                        budget = std::min(cfg.boundCalibrateSeconds, std::max(20.0, 0.03 * passEstimate));
                        if (boundTimer.getSeconds() < MIN_SECONDS) {
                            continue;
                        }
                        if (others == 0) {
                            why = "one mode left";
                            break;
                        }
                        if (settled) {
                            why = "settled";
                            break;
                        }
                        if (boundTimer.getSeconds() >= budget) {
                            break;
                        }
                    }
                    // off's time relative to the best, from the rounds that timed both
                    const double offOverBest = best == 0 ? 1.0 : std::exp(logRatio(0, best).first);
                    if (best != 0 && offOverBest < 1.03) {
                        best = 0;   // under 3%: not worth it
                    }
                    boundMode = boundModes[best];
                    tcout << "bound calibrate: " << timedSeeds << " seeds, " << rounds << " rounds of " << block
                          << "-seed blocks per mode in " << boundTimer.getSeconds() << "s (" << why << ", budget "
                          << budget << "s); time vs " << BOUND_NAMES[boundModes[best]] << ":";
                    for (std::size_t m = 0; m < nM; ++m) {
                        const auto [mu, se] = logRatio(m, best);
                        tcout << ' ' << BOUND_NAMES[boundModes[m]] << '=';
                        if (m == best) {
                            tcout << "1 (" << meanOf(m) / 1000.0 << " us/seed)";
                        } else {
                            tcout << std::exp(mu) << " +-" << 100.0 * se << '%';
                        }
                        if (!alive(m)) {
                            tcout << " (dropped after " << droppedAt[m] << ")";
                        }
                    }
                    tcout << " -> " << BOUND_NAMES[boundMode] << " (" << offOverBest << "x vs off)\n";
                } else {
                    // too few seeds to time: parents only (never slower than off by much)
                    boundMode = std::find(boundModes.begin(), boundModes.end(), 6) != boundModes.end() ? 6
                              : std::find(boundModes.begin(), boundModes.end(), 5) != boundModes.end() ? 5 : 1;
                    tcout << "bound: " << BOUND_NAMES[boundMode] << " (too few seeds to time)\n";
                }
                g_progress.detail(passLabel + " streaming");
            }
            const ProbeVariant variant = chosenVariant;
            const int mode = std::max(0, boundMode);
            const std::size_t startSeed = firstSeed;
            parallelFor(threads, seedPositions - startSeed, [&](const u32 t, const std::size_t i) {
                processSeed(t, startSeed + i, variant, mode);
            });
            if (visited.load() != seeds.size()) {
                throw std::runtime_error(passLabel + ": streamed " + std::to_string(visited.load()) + " seeds of "
                                         + std::to_string(seeds.size()));
            }
            const double passProbe = probeTimer.getSeconds();
            probeSeconds += passProbe;

            u64 passStreamed = 0, passInRange = 0, passBloom = 0, passBound = 0;
            for (u32 t = 0; t < threads; ++t) {
                passStreamed += streamed[t];
                passInRange += inRange[t];
                passBloom += bloomPasses[t];
                passBound += boundSkipped[t];
            }
            streamedTotal += passStreamed;
            inRangeTotal += passInRange;
            bloomPassTotal += passBloom;
            boundSkippedTotal += passBound;
            tcout << "  pass " << (pass + 1) << "/" << passes.size() << " partitions [" << pBegin << ","
                  << pEnd << "): load " << index.bytes() / 1e9 << " GB ("
                  << (index.largePages() ? "large pages" : "normal pages") << ") in " << passLoad << "s, stream+probe "
                  << passStreamed << " left states in " << passProbe << "s ("
                  << static_cast<double>(passStreamed) / passProbe / 1e6 << " M/s), Bloom pass "
                  << 100.0 * static_cast<double>(passBloom) / static_cast<double>(std::max<u64>(1, passInRange))
                  << "%" << (passBound != 0 ? ", bound skipped " + std::to_string(passBound) + " lookups" : "") << "\n";
        }

        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](const KeyedState& a, const KeyedState& b) {
                                         return a.key == b.key && a.state == b.state;
                                     }),
                         candidates.end());
        tcout << "probe totals: load " << loadSeconds << "s, stream+probe " << probeSeconds << "s, "
              << streamedTotal << " left states streamed, " << candidates.size() << " unique candidates\n";
        return candidates;
    }

    std::vector<KeyedState> streamAndProbe(const Board& leftRoot,
                                           const IndexFiles& files,
                                           const FingerprintMeta& meta,
                                           const std::vector<u64>& counts,
                                           const Config& cfg,
                                           const u32 threads,
                                           const IndexBuilder* builder = nullptr,
                                           const u64 builderReserveBytes = 0) {
        switch (cfg.streamDepth) {
            case 3: return streamAndProbeT<3>(leftRoot, files, meta, counts, cfg, threads, builder, builderReserveBytes);
            case 4: return streamAndProbeT<4>(leftRoot, files, meta, counts, cfg, threads, builder, builderReserveBytes);
            default: throw std::runtime_error("--stream-depth must be 3 or 4");
        }
    }

    // Confirms key-sorted candidates against the full depth-6 states on disk.
    // A partition with many candidates is read once sequentially; otherwise
    // each candidate reads just its bucket. Returns the confirmed states, sorted.
    std::vector<B1B2> verifyAgainstTable(const fs::path& dir, const FingerprintMeta& meta,
                                         const std::vector<KeyedState>& candidates) {
        Timer verifyTimer;
        g_progress.phase("verify", candidates.size(), "candidates against the depth-6 states on disk");
        constexpr std::size_t SEQUENTIAL_READ_THRESHOLD = 4096;
        const u64 bucketsPer = meta.bucketsPerPartition();
        const u32 shift = meta.bucketShift();
        std::vector<B1B2> middles;
        u64 falseMatches = 0;
        std::vector<u32> partitionStarts(bucketsPer + 1);
        std::size_t i = 0;
        while (i < candidates.size()) {
            const u32 p = static_cast<u32>(candidates[i].key >> PARTITION_SHIFT);
            std::size_t end = i;
            while (end < candidates.size() && (candidates[end].key >> PARTITION_SHIFT) == p) {
                ++end;
            }
            FingerprintRange::readRange(fpDir32Path(dir), p * (bucketsPer + 1) * sizeof(u32),
                                        partitionStarts.data(), partitionStarts.size() * sizeof(u32));

            std::vector<B1B2> whole;
            std::ifstream in;
            if (end - i >= SEQUENTIAL_READ_THRESHOLD) {
                whole = readAll<B1B2>(sortedPath(dir, p));
            } else {
                in.open(sortedPath(dir, p), std::ios::binary);
            }
            for (; i < end; ++i) {
                const u64 bucket = (candidates[i].key >> shift) & (bucketsPer - 1);
                const u32 lo = partitionStarts[bucket];
                const u32 hi = partitionStarts[bucket + 1];
                bool found = false;
                if (!whole.empty()) {
                    found = std::find(whole.begin() + lo, whole.begin() + hi, candidates[i].state)
                            != whole.begin() + hi;
                } else {
                    std::vector<B1B2> run(hi - lo);
                    in.seekg(static_cast<std::streamoff>(static_cast<u64>(lo) * sizeof(B1B2)));
                    in.read(reinterpret_cast<char*>(run.data()),
                            static_cast<std::streamsize>(run.size() * sizeof(B1B2)));
                    if (!in) {
                        throw std::runtime_error("failed to read candidate bucket");
                    }
                    found = std::find(run.begin(), run.end(), candidates[i].state) != run.end();
                }
                if (found) {
                    middles.push_back(candidates[i].state);
                } else {
                    ++falseMatches;
                }
                g_progress.add();
            }
        }
        std::sort(middles.begin(), middles.end());
        middles.erase(std::unique(middles.begin(), middles.end()), middles.end());
        tcout << "verify: " << middles.size() << " confirmed, " << falseMatches
              << " fingerprint false matches, " << verifyTimer.getSeconds() << "s\n";
        return middles;
    }

    std::vector<B1B2> probeFingerprint(const Board& leftRoot,
                                       const fs::path& dir,
                                       const std::vector<u64>& counts,
                                       const Config& cfg,
                                       const u32 threads) {
        if (!fs::exists(fp16MetaPath(dir)) || (cfg.rebuild && !g_indexBuiltWithTable)) {
            buildFingerprintIndex(dir, counts, threads);
        }
        const FingerprintMeta meta = readFingerprintMeta(fp16MetaPath(dir));
        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, d6Files(dir), meta, counts, cfg, threads);
        return verifyAgainstTable(dir, meta, candidates);
    }

    // ------------------------------------------------------------------
    // No-table depth-6 probe (default)
    // ------------------------------------------------------------------
    // fn(thread, child, key) for every one-move child of `frontier`.
    template<typename Fn>
    void expandFrontier(const JVec<B1B2>& frontier, const bool isFat, const u32 threads, Fn&& fn) {
        const std::size_t n = frontier.size();
        constexpr std::size_t CHUNK = 1U << 14;
        parallelFor(threads, (n + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
            B1B2 children[60];
            u64 childHashes[60];
            const std::size_t end = std::min(n, (c + 1) * CHUNK);
            for (std::size_t i = c * CHUNK; i < end; ++i) {
                const std::size_t m = isFat
                        ? frontier_recovery_detail::emitNoneChildren<true>(frontier[i], children, childHashes)
                        : frontier_recovery_detail::emitNoneChildren<false>(frontier[i], children, childHashes);
                for (std::size_t k = 0; k < m; ++k) {
                    fn(t, children[k], mixKey(childHashes[k]));
                }
            }
        });
    }

    std::vector<B1B2> probeNoTable(const Board& leftRoot, const Board& rightRoot, const Config& cfg,
                                   const u32 threads) {
        const bool isFat = rightRoot.getFatBool();
        Timer buildTimer;
        g_progress.phase("d6-build", 0, "depth-5 frontier (no table)");
        JVec<B1B2> depth5;
        {
            JVec<u64> hashes;
            buildUniqueNoneDepthFrontierB1B2<5>(rightRoot, depth5, hashes, false);
        }
        tcout << "depth-5 exact unique: " << depth5.size() << " states in " << buildTimer.getSeconds()
              << "s (no table: depth 6 is regenerated in RAM, nothing is written)\n";

        // Exact raw child count per partition.
        std::vector<std::vector<u64>> rawPer(threads, std::vector<u64>(PARTITION_COUNT, 0));
        expandFrontier(depth5, isFat, threads, [&](const u32 t, const B1B2&, const u64 key) {
            ++rawPer[t][key >> PARTITION_SHIFT];
        });
        std::vector<u64> raw(PARTITION_COUNT, 0);
        u64 rawTotal = 0, rawMax = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            for (u32 th = 0; th < threads; ++th) {
                raw[p] += rawPer[th][p];
            }
            rawTotal += raw[p];
            rawMax = std::max(rawMax, raw[p]);
        }

        // De-dupe ratio from partition 0 (partitions are near-uniform).
        double ratio = 1.0;
        {
            std::vector<u64> keys0;
            keys0.reserve(raw[0]);
            std::mutex m0;
            std::vector<std::vector<u64>> local(threads);
            expandFrontier(depth5, isFat, threads, [&](const u32 t, const B1B2&, const u64 key) {
                if ((key >> PARTITION_SHIFT) == 0) {
                    local[t].push_back(key);
                }
            });
            for (auto& v : local) {
                keys0.insert(keys0.end(), v.begin(), v.end());
            }
            std::sort(keys0.begin(), keys0.end());
            const u64 u0 = static_cast<u64>(std::unique(keys0.begin(), keys0.end()) - keys0.begin());
            ratio = static_cast<double>(u0) / static_cast<double>(std::max<u64>(1, keys0.size()));
        }
        std::vector<u64> est(PARTITION_COUNT, 0);
        u64 estTotal = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            est[p] = static_cast<u64>(static_cast<double>(raw[p]) * ratio * 1.02) + 1024;
            estTotal += est[p];
        }
        FingerprintMeta meta;
        const u64 estMax = static_cast<u64>(static_cast<double>(rawMax) * ratio * 1.02);
        while (meta.bucketBits < PARTITION_SHIFT && (estMax >> meta.bucketBits) > TARGET_STATES_PER_BUCKET) {
            ++meta.bucketBits;
        }
        meta.bloomBlocksPerPartition = std::max<u64>(1, (estMax * BLOOM_BITS_PER_STATE + 255) / 256);
        meta.stateCount = estTotal;
        tcout << "depth-6 raw children: " << rawTotal << ", de-dupe ratio " << ratio << " -> ~"
              << static_cast<u64>(static_cast<double>(rawTotal) * ratio) << " unique (counted in "
              << buildTimer.getSeconds() << "s)\n";
        const double setupSeconds = buildTimer.getSeconds();

        // Building a pass's index needs room for raw keys of a few partitions at a time.
        const u64 keyReserve = std::max<u64>(2'000'000'000ULL, rawMax * sizeof(u64) * 8);
        double indexBuildSeconds = 0.0;
        u64 uniqueTotal = 0;

        const IndexBuilder builder = [&](FingerprintRange& idx, const u32 p0, const u32 p1) {
            Timer timer;
            const u64 bucketsPer = meta.bucketsPerPartition();
            const u64 dirPer = bucketsPer + 1;
            const u64 bloomPer = meta.bloomBlocksPerPartition * 8;
            const u32 shift = meta.bucketShift();
            const u64 parts = p1 - p0;
            u64 cap = 0;
            for (u32 p = p0; p < p1; ++p) {
                cap += std::min(raw[p], est[p]);
            }
            idx.meta = meta;
            idx.pBegin = p0;
            idx.pEnd = p1;
            idx.fingerprints.ensure(cap);
            idx.starts.ensure(parts * dirPer);
            idx.bloom.ensure(parts * bloomPer);
            std::memset(idx.bloom.data(), 0, parts * bloomPer * sizeof(u32));
            idx.partitionFirst.assign(1, 0);

            // Sub-groups of partitions whose raw keys fit the leftover RAM,
            // measured now. Large-page buffers are already out of free RAM;
            // untouched normal pages are not yet, so count those as used.
            const u64 indexBytes = (idx.fingerprints.largePages() ? 0 : cap * sizeof(u16))
                                   + (idx.starts.largePages() ? 0 : parts * dirPer * sizeof(u32));
            const u64 avail = cfg.ramBytes != 0 ? keyReserve
                                                : std::max(keyReserve, defaultRamBudgetBytes(indexBytes));
            u64 fill = 0;   // fingerprints written so far in this range
            for (u32 s0 = p0; s0 < p1;) {
                u32 s1 = s0;
                u64 bytes = 0;
                while (s1 < p1 && (s1 == s0 || bytes + raw[s1] * sizeof(u64) <= avail)) {
                    bytes += raw[s1++] * sizeof(u64);
                }
                const u32 gN = s1 - s0;
                if (s0 == p0) {
                    tcout << "  index build: " << avail / 1e9 << " GB for keys -> about "
                          << (parts + gN - 1) / gN << " regeneration(s) of depth 6\n";
                }
                std::vector<std::unique_ptr<u64[]>> keys(gN);
                std::vector<std::atomic<u64>> got(gN);
                for (u32 i = 0; i < gN; ++i) {
                    keys[i].reset(new u64[std::max<u64>(1, raw[s0 + i])]);
                    got[i].store(0);
                }
                constexpr std::size_t STAGE = 1024;
                std::vector<std::vector<std::vector<u64>>> stage(threads, std::vector<std::vector<u64>>(gN));
                auto spill = [&](const u32 i, std::vector<u64>& buf) {
                    const u64 at = got[i].fetch_add(buf.size());
                    if (at + buf.size() > raw[s0 + i]) {
                        throw std::runtime_error("no-table partition overflow");
                    }
                    std::memcpy(keys[i].get() + at, buf.data(), buf.size() * sizeof(u64));
                    buf.clear();
                };
                expandFrontier(depth5, isFat, threads, [&](const u32 t, const B1B2&, const u64 key) {
                    const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                    if (p < s0 || p >= s1) {
                        return;
                    }
                    std::vector<u64>& buf = stage[t][p - s0];
                    if (buf.capacity() == 0) {
                        buf.reserve(STAGE);
                    }
                    buf.push_back(key);
                    if (buf.size() == STAGE) {
                        spill(p - s0, buf);
                    }
                });
                for (u32 th = 0; th < threads; ++th) {
                    for (u32 i = 0; i < gN; ++i) {
                        if (!stage[th][i].empty()) {
                            spill(i, stage[th][i]);
                        }
                    }
                }
                stage.clear();
                std::vector<u64> uniq(gN, 0);
                parallelFor(threads, gN, [&](u32, const std::size_t i) {
                    u64* a = keys[i].get();
                    const u64 n = got[i].load();
                    std::sort(a, a + n);
                    uniq[i] = static_cast<u64>(std::unique(a, a + n) - a);
                });
                std::vector<u64> offset(gN, 0);
                for (u32 i = 0; i < gN; ++i) {
                    offset[i] = fill;
                    fill += uniq[i];
                    if (fill > cap) {
                        throw std::runtime_error("no-table index estimate too small");
                    }
                    idx.partitionFirst.push_back(fill);
                }
                parallelFor(threads, gN, [&](u32, const std::size_t i) {
                    const u64* a = keys[i].get();
                    const u64 n = uniq[i];
                    const u64 part = (s0 + i) - p0;
                    u16* fps = idx.fingerprints.data() + offset[i];
                    u32* starts = idx.starts.data() + part * dirPer;
                    u32* bloom = idx.bloom.data() + part * bloomPer;
                    u64 next = 0;
                    for (u64 k = 0; k < n; ++k) {
                        const u64 bucket = (a[k] >> shift) & (bucketsPer - 1);
                        while (next <= bucket) {
                            starts[next++] = static_cast<u32>(k);
                        }
                        fps[k] = static_cast<u16>(a[k]);
                        const u64 h = bloomHash(a[k]);
                        bloomInsertAtomic(bloom + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
                    }
                    while (next <= bucketsPer) {
                        starts[next++] = static_cast<u32>(n);
                    }
                });
                s0 = s1;
            }
            uniqueTotal += fill;
            indexBuildSeconds += timer.getSeconds();
            tcout << "  index for partitions [" << p0 << "," << p1 << ") built in RAM: " << fill
                  << " unique states in " << timer.getSeconds() << "s\n";
        };

        FingerprintMeta probeMeta = meta;
        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, IndexFiles{}, probeMeta, est, cfg,
                                                                  threads, &builder, keyReserve);
        tcout << "depth-6 exact unique: " << uniqueTotal << " (index-only, in RAM; no table written)\n";
        tcout << "BUILD fingerprint index: " << uniqueTotal << " states, 2^" << meta.bucketBits
              << " buckets/partition, built in RAM in " << indexBuildSeconds << "s\n";
        tcout << "BUILD depth-6 table: " << setupSeconds << "s (no table; depth-5 frontier + counts)\n";

        // Confirm hits: a candidate is real iff it is one of the regenerated
        // depth-6 children. A bitmap on key bits skips almost every child.
        Timer verifyTimer;
        g_progress.phase("verify", 1, "regenerating depth 6 to confirm candidates");
        std::vector<KeyedState> cand(candidates.begin(), candidates.end());   // sorted by key, unique
        constexpr u32 BITMAP_BITS = 1U << 27;
        std::vector<u64> bitmap(BITMAP_BITS / 64, 0);
        auto bitOf = [](const u64 key) { return static_cast<u32>((key >> 20) & (BITMAP_BITS - 1)); };
        for (const KeyedState& c : cand) {
            bitmap[bitOf(c.key) >> 6] |= 1ULL << (bitOf(c.key) & 63);
        }
        std::vector<std::atomic<u8>> confirmed(cand.size());
        for (auto& c : confirmed) {
            c.store(0);
        }
        if (!cand.empty()) {
            expandFrontier(depth5, isFat, threads, [&](u32, const B1B2& child, const u64 key) {
                const u32 b = bitOf(key);
                if ((bitmap[b >> 6] >> (b & 63) & 1) == 0) {
                    return;
                }
                auto it = std::lower_bound(cand.begin(), cand.end(), key,
                                           [](const KeyedState& c, const u64 k) { return c.key < k; });
                for (; it != cand.end() && it->key == key; ++it) {
                    if (it->state == child) {
                        confirmed[static_cast<std::size_t>(it - cand.begin())].store(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        std::vector<B1B2> middles;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            if (confirmed[i].load() != 0) {
                middles.push_back(cand[i].state);
            }
        }
        std::sort(middles.begin(), middles.end());
        middles.erase(std::unique(middles.begin(), middles.end()), middles.end());
        g_progress.add();
        tcout << "verify: " << middles.size() << " confirmed, " << (cand.size() - middles.size())
              << " fingerprint false matches, " << verifyTimer.getSeconds() << "s (by regeneration)\n";
        return middles;
    }

    // ------------------------------------------------------------------
    // No-table depth-7 probe (default for --table-depth 7 without an index on disk)
    // ------------------------------------------------------------------
    // The canonical 2-move stream from each frontier state (the left side's
    // generator: no same-lane pairs, one order per commuting pair). Every
    // state at exact depth 7 is a depth-5 state plus such a pair.
    constexpr u32 NT7_STREAM_DEPTH = 2;

    template<typename Fn>
    struct StreamCallbackSink {
        Fn& fn;
        void operator()(JVec<B1B2>& states, JVec<u64>& hashes, const u32 count) {
            for (u32 i = 0; i < count; ++i) {
                fn(states[i], mixKey(hashes[i]));
            }
        }
    };

    template<typename Fn>
    FORCEINLINE void streamTwo(const B1B2& parent, Fn&& fn) {
        StreamCallbackSink<Fn> sink{fn};
        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, NT7_STREAM_DEPTH>(makeBoardFromState(parent), sink, 2048);
    }

    // fn(thread, descendant, key) for every streamed descendant of `frontier`.
    template<typename Fn>
    void expandFrontierTwice(const JVec<B1B2>& frontier, const bool, const u32 threads, Fn&& fn) {
        const std::size_t n = frontier.size();
        constexpr std::size_t CHUNK = 1U << 10;
        parallelFor(threads, (n + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
            const std::size_t end = std::min(n, (c + 1) * CHUNK);
            for (std::size_t i = c * CHUNK; i < end; ++i) {
                streamTwo(frontier[i], [&](const B1B2& s, const u64 key) { fn(t, s, key); });
            }
        });
    }

    constexpr u32 NT7_RAW_PER_BUCKET = 64;   // raw fingerprints per bucket before de-dupe (~25 after)
    constexpr double NT7_COUNT_BLOCK_KEYS = 400e6;   // 6.4 GB in flight
    constexpr double NT7_FILL_BLOCK_KEYS = 250e6;    // 4 GB in flight, kept back from the index budget

    // Like expandFrontierTwice, but each key in partitions [p0, p1) goes to
    // the one thread that owns its partition, so consumers need no atomics
    // (random locked RMWs cannot overlap misses). Works in blocks of
    // `blockKeys` routed keys: generate into per-owner buffers, then each
    // owner groups its keys by (partition, top `subBits` of the bucket) for
    // byBucket(owner, key) and, if given, by (partition, top bits of the
    // Bloom hash) for byBloom(owner, key), so every write lands in a small,
    // reused window (cache- and TLB-friendly even on normal pages).
    struct NoConsumer {
        void operator()(u32, u64) const {}
    };

    template<typename FnA, typename FnB = NoConsumer>
    void expandTwiceRouted(const JVec<B1B2>& frontier, const u32 threads, const u32 p0, const u32 p1,
                           const double perParent, const double blockKeys, const u32 subBits, FnA&& byBucket,
                           FnB&& byBloom = NoConsumer{}) {
        constexpr bool BLOOM_PASS = !std::is_same_v<std::decay_t<FnB>, NoConsumer>;
        const std::size_t n = frontier.size();
        const u32 span = p1 - p0;
        const double inRange = static_cast<double>(span) / PARTITION_COUNT;
        const auto block = static_cast<std::size_t>(std::clamp(
                blockKeys / std::max(1.0, perParent * inRange), 1024.0, 1e8));
        const u32 producers = threads * 4;
        std::vector<std::vector<std::vector<u64>>> out(producers, std::vector<std::vector<u64>>(threads));
        std::vector<BigArray<u64>> scratch(threads);
        std::vector<std::vector<std::size_t>> subStart(threads);
        double genSeconds = 0.0, drainSeconds = 0.0;
        u64 routed = 0;
        for (std::size_t start = 0; start < n; start += block) {
            const std::size_t bEnd = std::min(n, start + block);
            Timer genTimer;
            parallelFor(threads, producers, [&](u32, const std::size_t task) {
                std::vector<std::vector<u64>>& mine = out[task];
                for (auto& v : mine) {
                    v.clear();
                }
                const std::size_t lo = start + (bEnd - start) * task / producers;
                const std::size_t hi = start + (bEnd - start) * (task + 1) / producers;
                for (std::size_t i = lo; i < hi; ++i) {
                    streamTwo(frontier[i], [&](const B1B2&, const u64 key) {
                        const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                        if (p >= p0 && p < p1) {
                            mine[static_cast<u64>(p - p0) * threads / span].push_back(key);
                        }
                    });
                }
            });
            genSeconds += genTimer.getSeconds();
            Timer drainTimer;
            parallelFor(threads, threads, [&](u32, const std::size_t owner) {
                const u32 oLo = p0 + static_cast<u32>((owner * span + threads - 1) / threads);
                const u32 oHi = p0 + static_cast<u32>(((owner + 1) * span + threads - 1) / threads);
                if (oLo >= oHi) {
                    return;
                }
                const std::size_t nSub = static_cast<std::size_t>(oHi - oLo) << subBits;
                const u64 subMask = (1ULL << subBits) - 1;
                std::vector<std::size_t>& at = subStart[owner];
                BigArray<u64>& sorted = scratch[owner];
                // Counting sort of the owner's keys by subOf, then fn on each in order.
                auto groupAndDrain = [&](auto&& subOf, auto&& fn) {
                    at.assign(nSub + 1, 0);
                    std::size_t total = 0;
                    for (u32 prod = 0; prod < producers; ++prod) {
                        for (const u64 key : out[prod][owner]) {
                            ++at[subOf(key) + 1];
                        }
                        total += out[prod][owner].size();
                    }
                    for (std::size_t s = 0; s < nSub; ++s) {
                        at[s + 1] += at[s];
                    }
                    sorted.ensure(total + total / 4 + 1024);   // headroom: no realloc per block
                    for (u32 prod = 0; prod < producers; ++prod) {
                        for (const u64 key : out[prod][owner]) {
                            sorted[at[subOf(key)]++] = key;
                        }
                    }
                    for (std::size_t k = 0; k < total; ++k) {
                        fn(static_cast<u32>(owner), sorted[k]);
                    }
                };
                groupAndDrain([&](const u64 key) {
                    return (static_cast<std::size_t>((key >> PARTITION_SHIFT) - oLo) << subBits)
                           | ((key >> (PARTITION_SHIFT - subBits)) & subMask);
                }, byBucket);
                if constexpr (BLOOM_PASS) {
                    // bloomBlockInPartition is monotonic in the top bits of bloomHash.
                    groupAndDrain([&](const u64 key) {
                        return (static_cast<std::size_t>((key >> PARTITION_SHIFT) - oLo) << subBits)
                               | (bloomHash(key) >> (64 - subBits));
                    }, byBloom);
                }
            });
            drainSeconds += drainTimer.getSeconds();
            for (const auto& row : out) {
                for (const auto& v : row) {
                    routed += v.size();
                }
            }
        }
        tcout << "    regenerated depth 7 for partitions [" << p0 << "," << p1 << "): " << routed << " keys in blocks of ~"
              << static_cast<u64>(blockKeys) << ", generate " << genSeconds << "s, group+write " << drainSeconds << "s\n";
    }

    std::vector<B1B2> probeNoTableDepth7(const Board& leftRoot, const Board& rightRoot, const Config& cfg,
                                         const u32 threads) {
        const bool isFat = rightRoot.getFatBool();
        Timer buildTimer;
        g_progress.phase("d7-build", 0, "depth-5 frontier (no table)");
        JVec<B1B2> depth5;
        {
            JVec<u64> hashes;
            buildUniqueNoneDepthFrontierB1B2<5>(rightRoot, depth5, hashes, false);
        }
        tcout << "depth-5 exact unique: " << depth5.size() << " states in " << buildTimer.getSeconds()
              << "s (no table: depth 7 is regenerated in RAM, nothing is written)\n";

        // Size the buckets from a sample of the stream's fan-out.
        double fanOut = 0.0;
        {
            const std::size_t step = std::max<std::size_t>(1, depth5.size() / 4096);
            u64 got = 0, parents = 0;
            for (std::size_t i = 0; i < depth5.size(); i += step, ++parents) {
                streamTwo(depth5[i], [&](const B1B2&, u64) { ++got; });
            }
            fanOut = static_cast<double>(got) / static_cast<double>(std::max<u64>(1, parents));
        }
        const double rawEstimate = static_cast<double>(depth5.size()) * fanOut;
        FingerprintMeta meta;
        const auto rawMaxEstimate = static_cast<u64>(rawEstimate / PARTITION_COUNT * 1.05);
        while (meta.bucketBits < PARTITION_SHIFT && (rawMaxEstimate >> meta.bucketBits) > NT7_RAW_PER_BUCKET) {
            ++meta.bucketBits;
        }
        const u64 bucketsPer = meta.bucketsPerPartition();
        const u32 shift = meta.bucketShift();

        // Count pass: exact raw fingerprints per bucket (u16: ~32-64 each), plus
        // a 1/4096 key sample whose de-dupe ratio sizes the Bloom filters.
        g_progress.detail("counting depth-7 descendants per bucket");
        Timer countTimer;
        BigArray<u16> bucketCounts;
        bucketCounts.allocate(static_cast<u64>(PARTITION_COUNT) * bucketsPer);   // zero-filled
        tcout << "bucket counts: " << bucketCounts.size() * 2 / 1e9 << " GB, large pages "
              << (bucketCounts.largePages() ? "yes" : "no") << "\n";
        std::vector<std::vector<u64>> sample(threads);
        expandTwiceRouted(depth5, threads, 0, PARTITION_COUNT, fanOut, NT7_COUNT_BLOCK_KEYS, 6, [&](const u32 owner, const u64 key) {
            ++bucketCounts[(key >> PARTITION_SHIFT) * bucketsPer + ((key >> shift) & (bucketsPer - 1))];
            if ((key >> (64 - 12)) == 0) {
                sample[owner].push_back(key);
            }
        });
        std::vector<u64> raw(PARTITION_COUNT, 0);
        u64 maxBucket = 0;
        parallelFor(threads, PARTITION_COUNT, [&](u32, const std::size_t p) {
            u64 sum = 0;
            for (u64 b = 0; b < bucketsPer; ++b) {
                sum += bucketCounts[p * bucketsPer + b];
            }
            raw[p] = sum;
        });
        u64 rawTotal = 0, rawMax = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            rawTotal += raw[p];
            rawMax = std::max(rawMax, raw[p]);
        }
        for (u64 i = 0; i < bucketCounts.size(); ++i) {
            maxBucket = std::max<u64>(maxBucket, bucketCounts[i]);
        }
        if (maxBucket >= 65000) {
            throw std::runtime_error("depth-7 bucket count overflow");
        }
        double ratio = 0.5;
        {
            std::vector<u64> keys;
            for (auto& v : sample) {
                keys.insert(keys.end(), v.begin(), v.end());
                std::vector<u64>{}.swap(v);
            }
            std::sort(keys.begin(), keys.end());
            const u64 u = static_cast<u64>(std::unique(keys.begin(), keys.end()) - keys.begin());
            if (!keys.empty()) {
                ratio = static_cast<double>(u) / static_cast<double>(keys.size());
            }
        }
        meta.bloomBlocksPerPartition = std::max<u64>(
                1, (static_cast<u64>(static_cast<double>(rawMax) * ratio * 1.02) * BLOOM_BITS_PER_STATE + 255) / 256);
        meta.stateCount = rawTotal;
        tcout << "depth-7 raw descendants: " << rawTotal << " (" << fanOut << " per depth-5 state), de-dupe ratio " << ratio
              << " -> ~" << static_cast<u64>(static_cast<double>(rawTotal) * ratio) << " unique; 2^" << meta.bucketBits
              << " buckets/partition, largest bucket " << maxBucket << " (counted in " << countTimer.getSeconds()
              << "s)\n";
        const double setupSeconds = buildTimer.getSeconds();

        double indexBuildSeconds = 0.0;
        u64 uniqueTotal = 0;
        const IndexBuilder builder = [&](FingerprintRange& idx, const u32 p0, const u32 p1) {
            Timer timer;
            const u64 dirPer = bucketsPer + 1;
            const u64 bloomPer = meta.bloomBlocksPerPartition * 8;
            const u64 parts = p1 - p0;
            idx.meta = meta;
            idx.pBegin = p0;
            idx.pEnd = p1;
            idx.partitionFirst.assign(1, 0);
            for (u32 p = p0; p < p1; ++p) {
                idx.partitionFirst.push_back(idx.partitionFirst.back() + raw[p]);
            }
            idx.fingerprints.ensure(idx.partitionFirst.back());
            idx.starts.ensure(parts * dirPer);
            idx.bloom.ensure(parts * bloomPer);
            std::memset(idx.bloom.data(), 0, parts * bloomPer * sizeof(u32));
            u32* starts = idx.starts.data();
            u16* fps = idx.fingerprints.data();
            u32* bloom = idx.bloom.data();
            parallelFor(threads, parts, [&](u32, const std::size_t i) {
                const u16* c = bucketCounts.data() + (p0 + i) * bucketsPer;
                u32* s = starts + i * dirPer;
                u32 run = 0;
                for (u64 b = 0; b < bucketsPer; ++b) {
                    s[b] = run;
                    run += c[b];
                }
                s[bucketsPer] = run;
            });
            // Fill: starts[b] is bucket b's write cursor; afterwards its end.
            expandTwiceRouted(depth5, threads, p0, p1, fanOut, NT7_FILL_BLOCK_KEYS, 8, [&](u32, const u64 key) {
                const u64 part = (key >> PARTITION_SHIFT) - p0;
                const u64 b = (key >> shift) & (bucketsPer - 1);
                const u32 pos = starts[part * dirPer + b]++;
                fps[idx.partitionFirst[part] + pos] = static_cast<u16>(key);
            }, [&](u32, const u64 key) {
                const u64 part = (key >> PARTITION_SHIFT) - p0;
                const u64 h = bloomHash(key);
                bloomInsert(bloom + part * bloomPer + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
            });
            // De-dupe each bucket and compact it to the front of its
            // partition's region (regions keep their raw size).
            std::vector<u64> uniq(parts, 0);
            std::atomic<bool> mismatch{false};
            parallelFor(threads, parts, [&](u32, const std::size_t i) {
                u32* s = starts + i * dirPer;
                u16* f = fps + idx.partitionFirst[i];
                const u16* c = bucketCounts.data() + (p0 + i) * bucketsPer;
                u32 w = 0, begin = 0;
                for (u64 b = 0; b < bucketsPer; ++b) {
                    const u32 end = s[b];
                    if (end - begin != c[b]) {
                        mismatch = true;
                    }
                    std::sort(f + begin, f + end);
                    s[b] = w;
                    for (u32 k = begin; k < end; ++k) {
                        if (k == begin || f[k] != f[k - 1]) {
                            f[w++] = f[k];
                        }
                    }
                    begin = end;
                }
                s[bucketsPer] = w;
                uniq[i] = w;
            });
            if (mismatch) {
                throw std::runtime_error("depth-7 fill does not match the count pass");
            }
            u64 passUnique = 0, passRaw = 0;
            for (u64 i = 0; i < parts; ++i) {
                passUnique += uniq[i];
                passRaw += raw[p0 + i];
            }
            uniqueTotal += passUnique;
            indexBuildSeconds += timer.getSeconds();
            tcout << "  index for partitions [" << p0 << "," << p1 << ") built in RAM: raw " << passRaw
                  << " -> " << passUnique << " fingerprints in " << timer.getSeconds() << "s\n";
        };

        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, IndexFiles{}, meta, raw, cfg, threads,
                                                                  &builder, 5'000'000'000ULL);
        tcout << "BUILD depth-7 index: " << uniqueTotal << " fingerprints (bucket-unique; raw " << rawTotal
              << "), built in RAM in " << indexBuildSeconds << "s\n";
        tcout << "BUILD depth-7 setup: " << setupSeconds << "s (no table; depth-5 frontier + count pass)\n";

        // Confirm hits: a candidate is real iff it is one of the regenerated
        // depth-7 descendants. A bitmap on key bits skips almost every one.
        Timer verifyTimer;
        g_progress.phase("verify", 1, "regenerating depth 7 to confirm candidates");
        constexpr u32 BITMAP_BITS = 1U << 27;
        std::vector<u64> bitmap(BITMAP_BITS / 64, 0);
        auto bitOf = [](const u64 key) { return static_cast<u32>((key >> 20) & (BITMAP_BITS - 1)); };
        for (const KeyedState& c : candidates) {
            bitmap[bitOf(c.key) >> 6] |= 1ULL << (bitOf(c.key) & 63);
        }
        std::vector<std::atomic<u8>> confirmed(candidates.size());
        for (auto& c : confirmed) {
            c.store(0);
        }
        if (!candidates.empty()) {
            expandFrontierTwice(depth5, isFat, threads, [&](u32, const B1B2& state, const u64 key) {
                const u32 b = bitOf(key);
                if ((bitmap[b >> 6] >> (b & 63) & 1) == 0) {
                    return;
                }
                auto it = std::lower_bound(candidates.begin(), candidates.end(), key,
                                           [](const KeyedState& c, const u64 k) { return c.key < k; });
                for (; it != candidates.end() && it->key == key; ++it) {
                    if (it->state == state) {
                        confirmed[static_cast<std::size_t>(it - candidates.begin())].store(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        std::vector<B1B2> middles;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (confirmed[i].load() != 0) {
                middles.push_back(candidates[i].state);
            }
        }
        std::sort(middles.begin(), middles.end());
        middles.erase(std::unique(middles.begin(), middles.end()), middles.end());
        g_progress.add();
        tcout << "verify: " << middles.size() << " confirmed, " << (candidates.size() - middles.size())
              << " fingerprint false matches, " << verifyTimer.getSeconds() << "s (by regeneration)\n";
        tcout << "depth-7 confirmed middles: " << middles.size() << " of " << candidates.size()
              << " fingerprint hits\n";
        return middles;
    }

    // ------------------------------------------------------------------
    // In-RAM depth-6 build (--ram-build)
    // ------------------------------------------------------------------
    // The disk build writes every raw child (~1.5x the table) to 256 files,
    // reads them back to sort, then re-reads the sorted table to index it: on
    // a SATA SSD that I/O is most of the build (16-3: 55 min). Here the
    // depth-5 frontier stays in RAM; one count pass sizes every partition,
    // then each pass regenerates the children of a RAM-sized group of
    // partitions, sorts and de-dupes them in memory, and writes the sorted
    // partition files plus their fingerprint index. The only I/O is those
    // final sequential writes. Output is identical to the disk build (same
    // key order and de-dupe), with the index sized from the first group.
    std::vector<u64> buildDepth6InRam(const Board& root, const fs::path& dir, const Config& cfg, const u32 threads) {
        fs::create_directories(dir);
        const bool isFat = root.getFatBool();

        Timer d5Timer;
        g_progress.phase("d6-build", 0, "depth-5 frontier");
        JVec<B1B2> depth5;
        JVec<u64> depth5Hashes;
        buildUniqueNoneDepthFrontierB1B2<5>(root, depth5, depth5Hashes, false);
        JVec<u64>{}.swap(depth5Hashes);
        const std::size_t d5Count = depth5.size();
        tcout << "depth-5 exact unique: " << d5Count << " states in " << d5Timer.getSeconds() << "s\n";

        // fn(thread, child, key) for every one-move child of the depth-5 frontier.
        auto expand = [&](auto&& fn) {
            constexpr std::size_t CHUNK = 1U << 14;
            parallelFor(threads, (d5Count + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
                B1B2 children[60];
                u64 childHashes[60];
                const std::size_t end = std::min(d5Count, (c + 1) * CHUNK);
                for (std::size_t i = c * CHUNK; i < end; ++i) {
                    const std::size_t n = isFat
                            ? frontier_recovery_detail::emitNoneChildren<true>(depth5[i], children, childHashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(depth5[i], children, childHashes);
                    for (std::size_t k = 0; k < n; ++k) {
                        fn(t, children[k], mixKey(childHashes[k]));
                    }
                }
            });
        };

        // Count pass: exact raw children per partition.
        Timer countTimer;
        std::vector<std::vector<u64>> rawPer(threads, std::vector<u64>(PARTITION_COUNT, 0));
        expand([&](const u32 t, const B1B2&, const u64 key) { ++rawPer[t][key >> PARTITION_SHIFT]; });
        std::vector<u64> raw(PARTITION_COUNT, 0);
        u64 rawTotal = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            for (u32 th = 0; th < threads; ++th) {
                raw[p] += rawPer[th][p];
            }
            rawTotal += raw[p];
        }
        tcout << "depth-6 raw children: " << rawTotal << " (" << gbOf(rawTotal)
              << " GB, kept in RAM) counted in " << countTimer.getSeconds() << "s\n";

        const u64 budget = cfg.ramBytes != 0 ? cfg.ramBytes : defaultRamBudgetBytes(d5Count * sizeof(B1B2));
        auto partitionBytes = [&](const u32 p) { return raw[p] * (sizeof(KeyedState) + sizeof(u16)) + (64ULL << 20); };
        std::vector<std::pair<u32, u32>> groups;
        for (u32 p = 0; p < PARTITION_COUNT;) {
            u32 q = p;
            u64 bytes = 0;
            while (q < PARTITION_COUNT && (q == p || bytes + partitionBytes(q) <= budget)) {
                bytes += partitionBytes(q++);
            }
            groups.emplace_back(p, q);
            p = q;
        }
        tcout << "in-RAM depth-6 build: RAM budget " << budget / 1e9 << " GB -> " << groups.size()
              << " pass(es) over the depth-5 frontier\n";

        std::FILE* fpOut = std::fopen(fp16Path(dir).string().c_str(), "wb");
        std::FILE* dirOut = std::fopen(fpDir32Path(dir).string().c_str(), "wb");
        std::FILE* bloomOut = std::fopen(bloomPartPath(dir).string().c_str(), "wb");
        if (fpOut == nullptr || dirOut == nullptr || bloomOut == nullptr) {
            throw std::runtime_error("failed to open fingerprint index files");
        }
        FingerprintMeta meta;
        bool metaSet = false;
        std::vector<u64> unique(PARTITION_COUNT, 0);
        Timer writeTimerTotal;
        double writeSeconds = 0.0;
        g_progress.phase("d6-build", groups.size(), "in-RAM passes");

        for (std::size_t g = 0; g < groups.size(); ++g) {
            const auto [g0, g1] = groups[g];
            const u32 gN = g1 - g0;
            g_progress.detail("in-RAM pass " + std::to_string(g + 1) + "/" + std::to_string(groups.size()));
            Timer passTimer;
            std::vector<std::unique_ptr<KeyedState[]>> arr(gN);
            std::vector<std::atomic<u64>> fill(gN);
            for (u32 i = 0; i < gN; ++i) {
                arr[i].reset(new KeyedState[std::max<u64>(1, raw[g0 + i])]);
                fill[i].store(0);
            }
            constexpr std::size_t STAGE = 512;
            std::vector<std::vector<std::vector<KeyedState>>> stage(threads, std::vector<std::vector<KeyedState>>(gN));
            for (auto& perThread : stage) {
                for (auto& buf : perThread) {
                    buf.reserve(STAGE);
                }
            }
            auto spill = [&](const u32 i, std::vector<KeyedState>& buf) {
                const u64 at = fill[i].fetch_add(buf.size());
                if (at + buf.size() > raw[g0 + i]) {
                    throw std::runtime_error("depth-6 partition overflow (count pass mismatch)");
                }
                std::memcpy(arr[i].get() + at, buf.data(), buf.size() * sizeof(KeyedState));
                buf.clear();
            };
            expand([&](const u32 t, const B1B2& child, const u64 key) {
                const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                if (p < g0 || p >= g1) {
                    return;
                }
                std::vector<KeyedState>& buf = stage[t][p - g0];
                buf.push_back({key, child});
                if (buf.size() == STAGE) {
                    spill(p - g0, buf);
                }
            });
            for (u32 th = 0; th < threads; ++th) {
                for (u32 i = 0; i < gN; ++i) {
                    if (!stage[th][i].empty()) {
                        spill(i, stage[th][i]);
                    }
                }
            }
            stage.clear();
            const double expandSeconds = passTimer.getSeconds();

            // Sort + de-dupe each partition in place (same order as the disk build).
            Timer sortTimer;
            std::vector<u64> groupUnique(gN, 0);
            parallelFor(threads, gN, [&](u32, const std::size_t i) {
                KeyedState* a = arr[i].get();
                const u64 n = fill[i].load();
                std::sort(a, a + n);
                u64 w = 0;
                for (u64 k = 0; k < n; ++k) {
                    if (w == 0 || a[k].key != a[w - 1].key || !(a[k].state == a[w - 1].state)) {
                        a[w++] = a[k];
                    }
                }
                groupUnique[i] = w;
            });
            const double sortSeconds = sortTimer.getSeconds();

            if (!metaSet) {
                // Size the index from the first group (partitions are near-uniform).
                const u64 maxUnique = *std::max_element(groupUnique.begin(), groupUnique.end());
                const u64 sized = maxUnique + maxUnique / 50;
                while (meta.bucketBits < PARTITION_SHIFT && (sized >> meta.bucketBits) > TARGET_STATES_PER_BUCKET) {
                    ++meta.bucketBits;
                }
                meta.bloomBlocksPerPartition = std::max<u64>(1, (sized * BLOOM_BITS_PER_STATE + 255) / 256);
                metaSet = true;
            }

            // Index pieces per partition, in parallel.
            struct Out {
                std::vector<u16> fps;
                std::vector<u32> starts;
                AlignedWords bloom;
            };
            std::vector<Out> outs(gN);
            const u64 bucketsPer = meta.bucketsPerPartition();
            const u32 shift = meta.bucketShift();
            parallelFor(threads, gN, [&](u32, const std::size_t i) {
                const KeyedState* a = arr[i].get();
                const u64 n = groupUnique[i];
                Out& o = outs[i];
                o.fps.resize(n);
                o.starts.assign(bucketsPer + 1, 0);
                o.bloom.allocate(meta.bloomBlocksPerPartition * 8);
                u64 next = 0;
                for (u64 k = 0; k < n; ++k) {
                    const u64 key = a[k].key;
                    const u64 bucket = (key >> shift) & (bucketsPer - 1);
                    while (next <= bucket) {
                        o.starts[next++] = static_cast<u32>(k);
                    }
                    o.fps[k] = static_cast<u16>(key);
                    const u64 h = bloomHash(key);
                    bloomInsertAtomic(o.bloom.aligned + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
                }
                while (next <= bucketsPer) {
                    o.starts[next++] = static_cast<u32>(n);
                }
            });

            // Write the sorted partitions and append the index, in partition order.
            Timer writeTimer;
            std::vector<B1B2> chunk(1U << 20);
            for (u32 i = 0; i < gN; ++i) {
                const u32 p = g0 + i;
                const KeyedState* a = arr[i].get();
                const u64 n = groupUnique[i];
                std::FILE* f = std::fopen(sortedPath(dir, p).string().c_str(), "wb");
                if (f == nullptr) {
                    throw std::runtime_error("failed to write " + sortedPath(dir, p).string());
                }
                for (u64 k = 0; k < n;) {
                    const u64 m = std::min<u64>(chunk.size(), n - k);
                    for (u64 j = 0; j < m; ++j) {
                        chunk[j] = a[k + j].state;
                    }
                    if (std::fwrite(chunk.data(), sizeof(B1B2), m, f) != m) {
                        throw std::runtime_error("sorted partition write failed");
                    }
                    k += m;
                }
                std::fclose(f);
                arr[i].reset();
                Out& o = outs[i];
                if (std::fwrite(o.fps.data(), sizeof(u16), o.fps.size(), fpOut) != o.fps.size()
                    || std::fwrite(o.starts.data(), sizeof(u32), o.starts.size(), dirOut) != o.starts.size()
                    || std::fwrite(o.bloom.aligned, sizeof(u32), meta.bloomBlocksPerPartition * 8, bloomOut)
                       != meta.bloomBlocksPerPartition * 8) {
                    throw std::runtime_error("fingerprint index write failed");
                }
                unique[p] = n;
                outs[i] = Out{};
            }
            writeSeconds += writeTimer.getSeconds();
            u64 groupRaw = 0, groupUniq = 0;
            for (u32 i = 0; i < gN; ++i) {
                groupRaw += raw[g0 + i];
                groupUniq += groupUnique[i];
            }
            tcout << "  pass " << (g + 1) << "/" << groups.size() << " partitions [" << g0 << "," << g1
                  << "): expand " << expandSeconds << "s, sort " << sortSeconds << "s, write "
                  << writeTimer.getSeconds() << "s, raw " << groupRaw << " -> unique " << groupUniq << '\n';
            g_progress.add();
        }
        std::fclose(fpOut);
        std::fclose(dirOut);
        std::fclose(bloomOut);
        u64 uniqueTotal = 0;
        for (const u64 c : unique) {
            uniqueTotal += c;
        }
        meta.stateCount = uniqueTotal;
        std::ofstream(fp16MetaPath(dir)) << meta.bucketBits << ' ' << meta.bloomBlocksPerPartition
                                         << ' ' << uniqueTotal << '\n';
        g_indexBuiltWithTable = true;
        tcout << "depth-6 exact unique: " << uniqueTotal << " (" << gbOf(uniqueTotal) << " GB on disk), "
              << "in-RAM build, writing " << writeSeconds << "s\n";
        tcout << "BUILD fingerprint index: " << uniqueTotal << " states, 2^" << meta.bucketBits
              << " buckets/partition, " << meta.indexBytes(uniqueTotal, PARTITION_COUNT) / 1e9 << " GB ("
              << static_cast<double>(meta.indexBytes(uniqueTotal, PARTITION_COUNT)) / static_cast<double>(std::max<u64>(1, uniqueTotal))
              << " B/state) in 0s (built with the table)\n";
        return unique;
    }

    // ------------------------------------------------------------------
    // Index-only depth-7 table (see the header comment)
    // ------------------------------------------------------------------
    constexpr u32 D7_TARGET_STATES_PER_BUCKET = 32;   // 0.125 B/state of offsets
    constexpr double D7_UNIQUE_PER_RAW = 0.70;        // measured 0.63 on 6-4 and 0.60 on 6-2

    // One pass over the depth-6 table: fn(thread, key) for every one-move child.
    template<typename Fn>
    void scanDepth6Children(const fs::path& dir, const bool isFat, const u32 threads, Fn&& fn) {
        auto load = [&](const u32 p) { return readAll<B1B2>(sortedPath(dir, p)); };
        std::future<std::vector<B1B2>> pending = std::async(std::launch::async, load, 0U);
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            const std::vector<B1B2> parents = pending.get();
            if (p + 1 < PARTITION_COUNT) {
                pending = std::async(std::launch::async, load, p + 1);
            }
            constexpr std::size_t CHUNK = 1U << 14;
            parallelFor(threads, (parents.size() + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
                B1B2 children[60];
                u64 childHashes[60];
                const std::size_t end = std::min(parents.size(), (c + 1) * CHUNK);
                for (std::size_t i = c * CHUNK; i < end; ++i) {
                    const std::size_t n = isFat
                            ? frontier_recovery_detail::emitNoneChildren<true>(parents[i], children, childHashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(parents[i], children, childHashes);
                    for (std::size_t k = 0; k < n; ++k) {
                        fn(t, mixKey(childHashes[k]));
                    }
                }
            });
            g_progress.add();   // one depth-6 partition scanned
        }
    }

    struct D7PartitionOut {
        std::vector<u16> fps;
        std::vector<u32> starts;
        AlignedWords bloom;
    };

    // Merges `chunks` sorted runs of one partition's raw keys, dropping
    // duplicates, straight into its fingerprint / bucket / Bloom index.
    void mergeD7Partition(const u64* keys, const std::vector<std::pair<u64, u64>>& runs,
                          const FingerprintMeta& meta, const u64 rawCount, D7PartitionOut& out) {
        const u64 bucketsPer = meta.bucketsPerPartition();
        const u32 shift = meta.bucketShift();
        out.fps.clear();
        out.fps.reserve(static_cast<std::size_t>(static_cast<double>(rawCount) * D7_UNIQUE_PER_RAW * 1.05));
        out.starts.assign(bucketsPer + 1, 0);
        out.bloom.allocate(meta.bloomBlocksPerPartition * 8);

        using Head = std::pair<u64, u32>;   // (key, run)
        std::vector<Head> heap;
        std::vector<u64> pos(runs.size());
        for (u32 r = 0; r < runs.size(); ++r) {
            pos[r] = runs[r].first;
            if (pos[r] < runs[r].second) {
                heap.emplace_back(keys[pos[r]], r);
            }
        }
        auto greater = [](const Head& a, const Head& b) { return a.first > b.first; };
        std::make_heap(heap.begin(), heap.end(), greater);
        u64 next = 0;
        bool have = false;
        u64 last = 0;
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), greater);
            auto [key, r] = heap.back();
            if (++pos[r] < runs[r].second) {
                heap.back() = {keys[pos[r]], r};
                std::push_heap(heap.begin(), heap.end(), greater);
            } else {
                heap.pop_back();
            }
            if (have && key == last) {
                continue;
            }
            have = true;
            last = key;
            const u64 bucket = (key >> shift) & (bucketsPer - 1);
            while (next <= bucket) {
                out.starts[next++] = static_cast<u32>(out.fps.size());
            }
            out.fps.push_back(static_cast<u16>(key));
            const u64 h = bloomHash(key);
            bloomInsertAtomic(out.bloom.aligned + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
        }
        while (next <= bucketsPer) {
            out.starts[next++] = static_cast<u32>(out.fps.size());
        }
    }

    void buildDepth7Index(const Board& root, const fs::path& dir, const Config& cfg, const u32 threads) {
        const IndexFiles files = d7Files(dir);
        const u32 pFirst = cfg.d7FirstPartition, pLimit = cfg.d7End();
        const bool isFat = root.getFatBool();
        const Timer total;

        // Pass 0: exact raw child count per partition, to size every buffer.
        // Progress unit: one depth-6 partition scanned (256 per pass).
        Timer countTimer;
        g_progress.phase("d7-index", PARTITION_COUNT, "count pass");
        std::vector<std::vector<u64>> rawPer(threads, std::vector<u64>(PARTITION_COUNT, 0));
        scanDepth6Children(dir, isFat, threads, [&](const u32 t, const u64 key) {
            ++rawPer[t][key >> PARTITION_SHIFT];
        });
        std::vector<u64> raw(PARTITION_COUNT, 0);
        u64 rawTotal = 0, rawMax = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            for (u32 th = 0; th < threads; ++th) {
                raw[p] += rawPer[th][p];
            }
            if (p >= pFirst && p < pLimit) {
                rawTotal += raw[p];
                rawMax = std::max(rawMax, raw[p]);
            }
        }
        tcout << "depth-7 raw children in partitions [" << pFirst << "," << pLimit << "): " << rawTotal
              << " (count pass " << countTimer.getSeconds() << "s)\n";

        FingerprintMeta meta;
        meta.partitionBegin = pFirst;
        meta.partitionLimit = pLimit;
        const auto uniqueEstimate = static_cast<u64>(static_cast<double>(rawMax) * D7_UNIQUE_PER_RAW);
        while (meta.bucketBits < PARTITION_SHIFT
               && (uniqueEstimate >> meta.bucketBits) > D7_TARGET_STATES_PER_BUCKET) {
            ++meta.bucketBits;
        }
        meta.bloomBlocksPerPartition = std::max<u64>(1, (uniqueEstimate * BLOOM_BITS_PER_STATE + 255) / 256);

        // Groups of partitions whose raw keys + index output fit the RAM budget.
        const u64 budget = cfg.ramBytes != 0 ? cfg.ramBytes : defaultRamBudgetBytes();
        auto partitionBytes = [&](const u32 p) {
            return raw[p] * 8 + static_cast<u64>(static_cast<double>(raw[p]) * D7_UNIQUE_PER_RAW * 2.1)
                   + (meta.bucketsPerPartition() + 1) * 4 + meta.bloomBlocksPerPartition * 32;
        };
        std::vector<std::pair<u32, u32>> groups;
        for (u32 p = pFirst; p < pLimit;) {
            u32 q = p;
            u64 bytes = 0;
            while (q < pLimit && (q == p || bytes + partitionBytes(q) <= budget)) {
                bytes += partitionBytes(q++);
            }
            groups.emplace_back(p, q);
            p = q;
        }
        g_progress.total(static_cast<u64>(PARTITION_COUNT) * (1 + groups.size()));
        tcout << "depth-7 index build: 2^" << meta.bucketBits << " buckets/partition, "
              << meta.bloomBlocksPerPartition << " Bloom blocks/partition, RAM budget " << budget / 1e9
              << " GB -> " << groups.size() << " pass(es) over the depth-6 table\n";

        std::FILE* fpOut = std::fopen(files.fp16.string().c_str(), "wb");
        std::FILE* dirOut = std::fopen(files.dir32.string().c_str(), "wb");
        std::FILE* bloomOut = std::fopen(files.bloom.string().c_str(), "wb");
        if (fpOut == nullptr || dirOut == nullptr || bloomOut == nullptr) {
            throw std::runtime_error("failed to open depth-7 index files");
        }
        std::vector<u64> unique(PARTITION_COUNT, 0);

        for (std::size_t g = 0; g < groups.size(); ++g) {
            const auto [g0, g1] = groups[g];
            const u32 gN = g1 - g0;
            g_progress.detail("pass " + std::to_string(g + 1) + "/" + std::to_string(groups.size())
                              + " partitions [" + std::to_string(g0) + "," + std::to_string(g1) + ")");
            Timer scanTimer;
            std::vector<std::unique_ptr<u64[]>> keys(gN);
            std::vector<std::atomic<u64>> fill(gN);
            for (u32 i = 0; i < gN; ++i) {
                keys[i].reset(new u64[std::max<u64>(1, raw[g0 + i])]);
                fill[i].store(0);
            }
            constexpr std::size_t STAGE = 1024;
            std::vector<std::vector<std::vector<u64>>> stage(threads, std::vector<std::vector<u64>>(gN));
            for (auto& perThread : stage) {
                for (auto& buf : perThread) {
                    buf.reserve(STAGE);
                }
            }
            auto spill = [&](const u32 i, std::vector<u64>& buf) {
                const u64 at = fill[i].fetch_add(buf.size());
                if (at + buf.size() > raw[g0 + i]) {
                    throw std::runtime_error("depth-7 partition overflow (count pass mismatch)");
                }
                std::memcpy(keys[i].get() + at, buf.data(), buf.size() * sizeof(u64));
                buf.clear();
            };
            scanDepth6Children(dir, isFat, threads, [&](const u32 t, const u64 key) {
                const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                if (p < g0 || p >= g1) {
                    return;
                }
                std::vector<u64>& buf = stage[t][p - g0];
                buf.push_back(key);
                if (buf.size() == STAGE) {
                    spill(p - g0, buf);
                }
            });
            for (u32 th = 0; th < threads; ++th) {
                for (u32 i = 0; i < gN; ++i) {
                    if (!stage[th][i].empty()) {
                        spill(i, stage[th][i]);
                    }
                }
            }
            for (u32 i = 0; i < gN; ++i) {
                if (fill[i].load() != raw[g0 + i]) {
                    throw std::runtime_error("depth-7 partition fill mismatch");
                }
            }
            const double scanSeconds = scanTimer.getSeconds();

            // Sort `threads` runs per partition in parallel, then merge each
            // partition (one thread per partition) into its index.
            Timer mergeTimer;
            std::vector<std::vector<std::pair<u64, u64>>> runs(gN);
            for (u32 i = 0; i < gN; ++i) {
                const u64 n = raw[g0 + i];
                for (u32 r = 0; r < threads; ++r) {
                    runs[i].emplace_back(n * r / threads, n * (r + 1) / threads);
                }
            }
            parallelFor(threads, static_cast<std::size_t>(gN) * threads, [&](u32, const std::size_t task) {
                const u32 i = static_cast<u32>(task / threads);
                const auto [lo, hi] = runs[i][task % threads];
                std::sort(keys[i].get() + lo, keys[i].get() + hi);
            });
            std::vector<D7PartitionOut> outs(gN);
            parallelFor(std::min(threads, gN), gN, [&](u32, const std::size_t i) {
                mergeD7Partition(keys[i].get(), runs[i], meta, raw[g0 + i], outs[i]);
                keys[i].reset();
            });
            for (u32 i = 0; i < gN; ++i) {
                D7PartitionOut& out = outs[i];
                unique[g0 + i] = out.fps.size();
                if (out.fps.size() >= (1ULL << 32)) {
                    throw std::runtime_error("depth-7 partition too large for 32-bit offsets");
                }
                if (std::fwrite(out.fps.data(), sizeof(u16), out.fps.size(), fpOut) != out.fps.size()
                    || std::fwrite(out.starts.data(), sizeof(u32), out.starts.size(), dirOut) != out.starts.size()
                    || std::fwrite(out.bloom.aligned, sizeof(u32), meta.bloomBlocksPerPartition * 8, bloomOut)
                       != meta.bloomBlocksPerPartition * 8) {
                    throw std::runtime_error("depth-7 index write failed");
                }
            }
            u64 groupUnique = 0, groupRaw = 0;
            for (u32 i = g0; i < g1; ++i) {
                groupUnique += unique[i];
                groupRaw += raw[i];
            }
            tcout << "  pass " << (g + 1) << "/" << groups.size() << " partitions [" << g0 << "," << g1
                  << "): scan " << scanSeconds << "s, sort+merge " << mergeTimer.getSeconds() << "s, raw "
                  << groupRaw << " -> unique " << groupUnique << '\n';
        }
        std::fclose(fpOut);
        std::fclose(dirOut);
        std::fclose(bloomOut);
        writeAll(files.counts, unique.data(), unique.size());
        for (u32 p = pFirst; p < pLimit; ++p) {
            meta.stateCount += unique[p];
        }
        // The meta file is written last: its presence marks a complete index.
        std::ofstream(files.meta) << meta.bucketBits << ' ' << meta.bloomBlocksPerPartition << ' '
                                  << meta.stateCount << ' ' << meta.partitionLimit << ' '
                                  << meta.partitionBegin << '\n';
        const u64 bytes = meta.indexBytes(meta.stateCount, pLimit - pFirst);
        tcout << "BUILD depth-7 index: " << meta.stateCount << " unique states in partitions [" << pFirst
              << "," << pLimit
              << "), " << bytes / 1e9 << " GB (" << static_cast<double>(bytes) / static_cast<double>(std::max<u64>(1, meta.stateCount))
              << " B/state) in " << total.getSeconds() << "s\n";
    }

    // Left states at distance <= 7 from the right root: fingerprint hits of
    // the depth-7 index whose one-move neighbourhood meets the depth-6 table.
    std::vector<B1B2> probeDepth7(const Board& leftRoot, const Board& rightRoot, const fs::path& dir,
                                  const std::vector<u64>& d6Counts, const Config& cfg, const u32 threads) {
        const IndexFiles files = d7Files(dir);
        bool build = cfg.rebuild || !fs::exists(files.meta);
        if (!build) {
            const FingerprintMeta existing = readFingerprintMeta(files.meta);
            build = existing.partitionBegin != cfg.d7FirstPartition || existing.partitionLimit != cfg.d7End();
        }
        if (build) {
            buildDepth7Index(rightRoot, dir, cfg, threads);
        }
        const FingerprintMeta meta = readFingerprintMeta(files.meta);
        const std::vector<u64> d7Counts = readAll<u64>(files.counts);

        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, files, meta, d7Counts, cfg, threads);

        // Exact confirmation through the depth-6 table: L is a child of a
        // depth-6 state iff one of L's children (moves are reversible) is one.
        Timer neighbourTimer;
        if (!fs::exists(fp16MetaPath(dir))) {
            buildFingerprintIndex(dir, d6Counts, threads);
        }
        const FingerprintMeta meta6 = readFingerprintMeta(fp16MetaPath(dir));
        FingerprintRange d6Index;
        d6Index.load(d6Files(dir), meta6, d6Counts, 0, PARTITION_COUNT);
        const bool isFat = rightRoot.getFatBool();

        std::vector<std::pair<KeyedState, u32>> hits;   // (depth-6 fingerprint hit, candidate)
        std::mutex hitLock;
        constexpr std::size_t CHUNK = 4096;
        g_progress.phase("d7-confirm", (candidates.size() + CHUNK - 1) / CHUNK, "neighbours vs depth-6 index");
        parallelFor(threads, (candidates.size() + CHUNK - 1) / CHUNK, [&](u32, const std::size_t c) {
            std::vector<std::pair<KeyedState, u32>> local;
            std::vector<KeyedState> found;
            B1B2 children[60];
            u64 childHashes[60];
            u64 childKeys[60];
            const std::size_t end = std::min(candidates.size(), (c + 1) * CHUNK);
            for (std::size_t i = c * CHUNK; i < end; ++i) {
                const std::size_t n = isFat
                        ? frontier_recovery_detail::emitNoneChildren<true>(candidates[i].state, children, childHashes)
                        : frontier_recovery_detail::emitNoneChildren<false>(candidates[i].state, children, childHashes);
                for (std::size_t k = 0; k < n; ++k) {
                    childKeys[k] = mixKey(childHashes[k]);
                }
                found.clear();
                for (std::size_t k = 0; k < n; k += PROBE_GROUP) {
                    d6Index.probeGroup(children + k, childKeys + k,
                                       static_cast<u32>(std::min<std::size_t>(PROBE_GROUP, n - k)), found);
                }
                for (const KeyedState& f : found) {
                    local.emplace_back(f, static_cast<u32>(i));
                }
            }
            if (!local.empty()) {
                std::lock_guard<std::mutex> lock(hitLock);
                hits.insert(hits.end(), local.begin(), local.end());
            }
            g_progress.add();
        });
        std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<KeyedState> uniqueChildren;
        for (const auto& h : hits) {
            if (uniqueChildren.empty() || uniqueChildren.back().key != h.first.key
                || !(uniqueChildren.back().state == h.first.state)) {
                uniqueChildren.push_back(h.first);
            }
        }
        tcout << "depth-7 confirmation: " << candidates.size() << " fingerprint hits, " << hits.size()
              << " neighbours pass the depth-6 index (" << neighbourTimer.getSeconds() << "s)\n";
        const std::vector<B1B2> confirmedChildren = verifyAgainstTable(dir, meta6, uniqueChildren);

        std::vector<B1B2> middles;
        for (const auto& [child, parent] : hits) {
            if (std::binary_search(confirmedChildren.begin(), confirmedChildren.end(), child.state)) {
                middles.push_back(candidates[parent].state);
            }
        }
        std::sort(middles.begin(), middles.end());
        middles.erase(std::unique(middles.begin(), middles.end()), middles.end());
        tcout << "depth-7 confirmed middles: " << middles.size() << " of " << candidates.size()
              << " fingerprint hits\n";
        return middles;
    }

    // ------------------------------------------------------------------
    // Depth-7 sample: one real pass over the depth-6 table, keeping N of the
    // 1024 depth-7 key slices. A full depth-7 build would repeat this pass for
    // every group of slices that fits in RAM; the raw depth-7 children (~1 TB
    // for a fat goal) cannot be spilled to disk and sorted afterwards.
    // ------------------------------------------------------------------
    void sampleDepth7(const Board& root,
                      const fs::path& d6Dir,
                      const std::vector<u64>& d6Counts,
                      const u32 threads,
                      const u32 sampleSlices,
                      const u64 passBudgetStates) {
        constexpr u32 D7_BITS = 10;
        constexpr u32 D7_SLICES = 1U << D7_BITS;
        constexpr u32 D7_SHIFT = 64U - D7_BITS;
        const fs::path d7Dir = d6Dir.parent_path() / (d6Dir.filename().string() + "_d7_sample");
        fs::create_directories(d7Dir);

        const bool isFat = root.getFatBool();
        std::vector<std::vector<std::vector<B1B2>>> local(threads, std::vector<std::vector<B1B2>>(sampleSlices));

        Timer passTimer;
        double readWaitSeconds = 0.0, expandSeconds = 0.0;
        u64 parents = 0, childrenTotal = 0;

        auto load = [&](const u32 p) { return readAll<B1B2>(sortedPath(d6Dir, p)); };
        std::future<std::vector<B1B2>> pending = std::async(std::launch::async, load, 0U);
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            Timer waitTimer;
            const std::vector<B1B2> parentsChunk = pending.get();
            readWaitSeconds += waitTimer.getSeconds();
            if (p + 1 < PARTITION_COUNT) {
                pending = std::async(std::launch::async, load, p + 1);
            }

            Timer expandTimer;
            std::vector<u64> perThreadChildren(threads, 0);
            const std::size_t chunk = 1U << 14;
            parallelFor(threads, (parentsChunk.size() + chunk - 1) / chunk, [&](const u32 t, const std::size_t c) {
                B1B2 children[60];
                u64 childHashes[60];
                const std::size_t end = std::min(parentsChunk.size(), (c + 1) * chunk);
                for (std::size_t i = c * chunk; i < end; ++i) {
                    const std::size_t n = isFat
                            ? frontier_recovery_detail::emitNoneChildren<true>(parentsChunk[i], children, childHashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(parentsChunk[i], children, childHashes);
                    perThreadChildren[t] += n;
                    for (std::size_t k = 0; k < n; ++k) {
                        const u32 slice = static_cast<u32>(mixKey(childHashes[k]) >> D7_SHIFT);
                        if (slice < sampleSlices) {
                            local[t][slice].push_back(children[k]);
                        }
                    }
                }
            });
            expandSeconds += expandTimer.getSeconds();
            parents += parentsChunk.size();
            for (const u64 n : perThreadChildren) {
                childrenTotal += n;
            }
        }
        const double scanSeconds = passTimer.getSeconds();
        tcout << "depth-7 pass over depth-6 table: " << parents << " parents, " << childrenTotal
              << " children (" << gbOf(childrenTotal) / 1000.0 << " TB raw) in " << scanSeconds
              << "s (expand " << expandSeconds << "s, waiting on SSD " << readWaitSeconds << "s)\n";

        // Sort + de-dupe the sampled slices, two at a time to bound peak RAM.
        const u32 sortConcurrency = std::min(2U, sampleSlices);
        Timer sortTimer;
        std::vector<u64> rawCounts(sampleSlices, 0), uniqueCounts(sampleSlices, 0);
        parallelFor(sortConcurrency, sampleSlices, [&](u32, const std::size_t s) {
            std::vector<KeyedState> keyed;
            u64 raw = 0;
            for (u32 t = 0; t < threads; ++t) {
                raw += local[t][s].size();
            }
            keyed.reserve(raw);
            for (u32 t = 0; t < threads; ++t) {
                for (const B1B2& state : local[t][s]) {
                    keyed.push_back({keyOf(state), state});
                }
                std::vector<B1B2>{}.swap(local[t][s]);
            }
            std::sort(keyed.begin(), keyed.end());
            std::vector<B1B2> unique;
            unique.reserve(keyed.size());
            for (std::size_t i = 0; i < keyed.size(); ++i) {
                if (i == 0 || keyed[i].key != keyed[i - 1].key || !(keyed[i].state == keyed[i - 1].state)) {
                    unique.push_back(keyed[i].state);
                }
            }
            writeAll(sortedPath(d7Dir, static_cast<u32>(s)), unique.data(), unique.size());
            rawCounts[s] = raw;
            uniqueCounts[s] = unique.size();
        });
        const double sortSeconds = sortTimer.getSeconds();

        u64 rawSample = 0, uniqueSample = 0;
        for (u32 s = 0; s < sampleSlices; ++s) {
            rawSample += rawCounts[s];
            uniqueSample += uniqueCounts[s];
            tcout << "  slice " << s << ": raw " << rawCounts[s] << ", unique " << uniqueCounts[s] << '\n';
        }

        const double scale = static_cast<double>(D7_SLICES) / sampleSlices;
        const double uniqueTotal = static_cast<double>(uniqueSample) * scale;
        const double rawPerSlice = static_cast<double>(rawSample) / sampleSlices;
        const u32 slicesPerPass = std::max<u32>(1, static_cast<u32>(static_cast<double>(passBudgetStates) / rawPerSlice));
        const u32 passes = (D7_SLICES + slicesPerPass - 1) / slicesPerPass;
        const double sortPerSlice = sortSeconds * sortConcurrency / sampleSlices;
        const double passSeconds = scanSeconds + sortPerSlice * slicesPerPass / sortConcurrency;

        tcout << "sample: " << sampleSlices << "/" << D7_SLICES << " slices, raw " << rawSample
              << ", unique " << uniqueSample << ", sort+dedupe+write " << sortSeconds << "s\n";
        tcout << "PROJECTED full depth-7 table: ~" << static_cast<u64>(uniqueTotal) << " unique states, ~"
              << gbOf(static_cast<u64>(uniqueTotal)) << " GB on disk\n";
        tcout << "PROJECTED full build: " << passes << " passes of " << slicesPerPass
              << " slices (RAM budget " << passBudgetStates << " raw states/pass) x ~" << passSeconds
              << "s = ~" << passes * passSeconds / 3600.0 << " h\n";
    }

    std::vector<std::string> readSolutionLines(const std::string& file) {
        std::vector<std::string> lines;
        std::ifstream in(file);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                line.pop_back();
            }
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    // ------------------------------------------------------------------
    // Middles files (work units)
    // ------------------------------------------------------------------
    void writeMiddlesFile(const std::string& path, const Config& cfg, const u64 seedsInShard,
                          const std::vector<B1B2>& middles) {
        const std::string tmp = path + ".part";
        {
            std::ofstream out(tmp, std::ios::trunc);
            out << "# mindbender middles v1\n"
                << "puzzle " << cfg.puzzle << "\n"
                << "split " << cfg.seedDepth << " " << cfg.streamDepth << " " << cfg.tableDepth << "\n"
                << "reverse " << (cfg.reverse ? 1 : 0) << "\n"
                << "shard " << cfg.shardIndex << " " << cfg.shardCount << "\n"
                << "seeds " << seedsInShard << "\n"
                << "count " << middles.size() << "\n";
            out << std::hex;
            for (const B1B2& m : middles) {
                out << m.b1 << " " << m.b2 << "\n";
            }
            out << std::dec << "end\n";
            if (!out) {
                throw std::runtime_error("failed to write " + tmp);
            }
        }
        fs::rename(tmp, path);   // only a complete file ever has the final name
        tcout << "middles written: " << path << " (" << middles.size() << " middles, shard " << cfg.shardIndex
              << "/" << cfg.shardCount << ")\n";
    }

    // Reads shards 0..N-1 of one search (each exactly once) and returns the
    // union of their middles; refuses mixed puzzles, splits or incomplete files.
    std::vector<B1B2> readMiddlesFiles(const std::vector<std::string>& paths, const Config& cfg) {
        std::vector<B1B2> all;
        std::vector<int> seen;
        u32 shardCount = 0;
        for (const std::string& path : paths) {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("cannot open " + path);
            }
            std::string line, puzzle;
            u32 s0 = 0, s1 = 0, s2 = 0, rev = 0, k = 0, n = 0;
            u64 count = 0;
            std::getline(in, line);
            if (line.rfind("# mindbender middles v1", 0) != 0) {
                throw std::runtime_error(path + ": not a middles file");
            }
            std::string key;
            in >> key >> puzzle >> key >> s0 >> s1 >> s2 >> key >> rev >> key >> k >> n >> key >> line >> key >> count;
            if (!in || puzzle != cfg.puzzle || s0 != cfg.seedDepth || s1 != cfg.streamDepth || s2 != cfg.tableDepth
                || (rev != 0) != cfg.reverse) {
                throw std::runtime_error(path + ": a different puzzle or split than this run");
            }
            if (shardCount == 0) {
                shardCount = n;
                seen.assign(n, 0);
            }
            if (n != shardCount || k >= n || seen[k]++ != 0) {
                throw std::runtime_error(path + ": shard " + std::to_string(k) + "/" + std::to_string(n)
                                         + " does not fit the other files (each shard exactly once)");
            }
            std::vector<B1B2> mine;
            in >> std::hex;
            for (u64 i = 0; i < count; ++i) {
                B1B2 m;
                in >> m.b1 >> m.b2;
                mine.push_back(m);
            }
            in >> std::dec >> line;
            if (!in || line != "end" || mine.size() != count) {
                throw std::runtime_error(path + ": incomplete");
            }
            all.insert(all.end(), mine.begin(), mine.end());
            tcout << "  " << path << ": shard " << k << "/" << n << ", " << count << " middles\n";
        }
        for (u32 k = 0; k < shardCount; ++k) {
            if (seen[k] == 0) {
                throw std::runtime_error("shard " + std::to_string(k) + "/" + std::to_string(shardCount) + " is missing");
            }
        }
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        tcout << "merged " << paths.size() << " shard files: " << all.size() << " unique middles\n";
        return all;
    }

    // Middles implied by a complete optimal solution file: the board after the
    // first `prefixMoves` moves of every solution, applied to the true start.
    B1B2 middleOf(const BoardPair& pair, const std::string& line, const std::size_t prefixMoves) {
        const bool isFat = pair.getStartState().getFatBool();
        const std::vector<u8> moves = isFat ? Memory::parseFatMoveString(line)
                                            : Memory::parseNormMoveString(line);
        if (moves.size() < prefixMoves) {
            throw std::runtime_error("solution shorter than the middle depth: " + line);
        }
        Board board = pair.getStartState();
        for (std::size_t i = 0; i < prefixMoves; ++i) {
            allActStructList[moves[i]].action(board);
        }
        return board.asB1B2();
    }

    std::vector<B1B2> expectedMiddles(const BoardPair& pair,
                                      const std::vector<std::string>& solutions,
                                      const std::size_t prefixMoves) {
        std::vector<B1B2> out;
        for (const std::string& line : solutions) {
            out.push_back(middleOf(pair, line, prefixMoves));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    template<int TOTAL_LEFT_DEPTH, int TABLE_DEPTH>
    int recoverWithLeftDepth(BoardSolverFrontier& solver, const bool reverse,
                             const std::vector<B1B2>& middles, const u32 threads) {
        if (reverse) {
            return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, true>(middles, static_cast<int>(threads));
        }
        return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, false>(middles, static_cast<int>(threads));
    }

    template<int TABLE_DEPTH>
    int recoverWithTableDepth(BoardSolverFrontier& solver, const Config& cfg, const std::vector<B1B2>& middles) {
        switch (cfg.seedDepth + cfg.streamDepth) {
            case 4: return recoverWithLeftDepth<4, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 5: return recoverWithLeftDepth<5, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 6: return recoverWithLeftDepth<6, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 7: return recoverWithLeftDepth<7, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 8: return recoverWithLeftDepth<8, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 9: return recoverWithLeftDepth<9, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            default: throw std::runtime_error("unsupported left depth for recovery");
        }
    }

    int recoverSolutions(BoardSolverFrontier& solver, const Config& cfg, const std::vector<B1B2>& middles) {
        return cfg.tableDepth == 7 ? recoverWithTableDepth<7>(solver, cfg, middles)
                                   : recoverWithTableDepth<6>(solver, cfg, middles);
    }

    Config parseArgs(const int argc, char** argv) {
        Config cfg;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + a);
                }
                return argv[++i];
            };
            if (a == "--out") cfg.outRoot = next();
            else if (a == "--batch") cfg.batchStates = std::stoull(next());
            else if (a == "--threads") cfg.threads = static_cast<u32>(std::stoul(next()));
            else if (a == "--seed-depth") cfg.seedDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--reverse") cfg.reverse = true;
            else if (a == "--rebuild") cfg.rebuild = true;
            else if (a == "--no-recover") cfg.recover = false;
            else if (a == "--batched") cfg.batched = true;
            else if (a == "--ram-gb") cfg.ramBytes = static_cast<u64>(std::stod(next()) * 1e9);
            else if (a == "--no-large-pages") cfg.largePages = false;
            else if (a == "--solutions") cfg.solutionsFile = next();
            else if (a == "--d7-sample") cfg.d7Sample = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-pass-budget") cfg.d7PassBudget = std::stoull(next());
            else if (a == "--table-depth") cfg.tableDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-partitions") cfg.d7Partitions = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-first-partition") cfg.d7FirstPartition = static_cast<u32>(std::stoul(next()));
            else if (a == "--progress-file") cfg.progressFile = next();
            else if (a == "--progress-interval") cfg.progressInterval = std::stod(next());
            else if (a == "--no-calibrate") cfg.calibrate = false;
            else if (a == "--ram-build") cfg.ramBuild = true;
            else if (a == "--disk-table") cfg.diskTable = true;
            else if (a == "--calibrate-seconds") cfg.calibrateSeconds = std::stod(next());
            else if (a == "--bound-calibrate-seconds") cfg.boundCalibrateSeconds = std::stod(next());
            else if (a == "--bound-kind") {
                const std::string v = next();
                if (v != "mix" && v != "rotation") {
                    throw std::runtime_error("--bound-kind must be mix or rotation");
                }
                cfg.boundKind = v == "mix" ? colour_mix::Kind::Mix : colour_mix::Kind::Rotation;
            }
            else if (a == "--bound") {
                const std::string v = next();
                cfg.boundMode = v == "auto" ? -1 : v == "off" ? 0 : v == "leaf" ? 1 : v == "both" ? 2
                              : v == "pdb" ? 3 : v == "pdbleaf" ? 4 : v == "parent" ? 5 : v == "pdbparent" ? 6 : -2;
                if (cfg.boundMode == -2) {
                    throw std::runtime_error("--bound must be auto, off, leaf, both, pdb, pdbleaf, parent or pdbparent");
                }
                if (cfg.boundMode == 3 || cfg.boundMode == 4 || cfg.boundMode == 6) {
                    cfg.pdbMode = 1;
                }
            }
            else if (a == "--pdb") {
                const std::string v = next();
                cfg.pdbMode = v == "auto" ? -1 : v == "off" ? 0 : v == "on" ? 1 : -2;
                if (cfg.pdbMode == -2) {
                    throw std::runtime_error("--pdb must be auto, off or on");
                }
            }
            else if (a == "--pdb-max-mentries") cfg.pdbMaxEntries = std::stoull(next()) * 1'000'000ULL;
            else if (a == "--stream-depth") cfg.streamDepth = static_cast<u32>(std::stoul(next()));
            else if (a == "--shard") {
                const std::string v = next();
                const std::size_t slash = v.find('/');
                if (slash == std::string::npos) {
                    throw std::runtime_error("--shard must be k/N");
                }
                cfg.shardIndex = static_cast<u32>(std::stoul(v.substr(0, slash)));
                cfg.shardCount = static_cast<u32>(std::stoul(v.substr(slash + 1)));
                if (cfg.shardCount == 0 || cfg.shardIndex >= cfg.shardCount) {
                    throw std::runtime_error("--shard k/N needs 0 <= k < N");
                }
            }
            else if (a == "--middles-out") cfg.middlesOut = next();
            else if (a == "--recover-from") cfg.recoverFrom.push_back(next());
            else if (a == "--probe-variant") {
                const std::string v = next();
                const auto colon = v.find(':');
                cfg.forcedGroup = std::stoi(v.substr(0, colon));
                cfg.forcedPipeline = colon == std::string::npos ? 0 : std::stoi(v.substr(colon + 1));
            }
            else cfg.puzzle = a;
        }
        if (cfg.threads == 0) {
            cfg.threads = std::max(1U, std::thread::hardware_concurrency());
        }
        if (cfg.tableDepth != 6 && cfg.tableDepth != 7) {
            throw std::runtime_error("--table-depth must be 6 or 7");
        }
        if (cfg.tableDepth == 7 && (cfg.reverse || cfg.batched)) {
            throw std::runtime_error("--table-depth 7 supports forward fingerprint mode only");
        }
        cfg.d7FirstPartition = std::min<u32>(cfg.d7FirstPartition, PARTITION_COUNT - 1);
        cfg.d7Partitions = std::clamp<u32>(cfg.d7Partitions, 1, PARTITION_COUNT - cfg.d7FirstPartition);
        return cfg;
    }

} // namespace

int main(const int argc, char** argv) {
    const Config cfg = parseArgs(argc, argv);
    g_largePagesWanted = cfg.largePages;
    tcout << std::unitbuf;   // flush every line: a redirected log can be followed live
    g_progress.start(cfg.progressFile, cfg.progressInterval);
    const BoardPair* pair = BoardLookup::getBoardPair(cfg.puzzle);
    if (pair == nullptr) {
        tcout << "Unknown puzzle: " << cfg.puzzle << '\n';
        return 1;
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();
    const Board& leftRoot = cfg.reverse ? goal : start;
    const Board& rightRoot = cfg.reverse ? start : goal;
    g_rightRoot = rightRoot.asB1B2();
    const fs::path dir = cfg.outRoot / (cfg.puzzle + (cfg.reverse ? "_start_d6" : "_goal_d6"));
    const u32 leftDepth = cfg.seedDepth + cfg.streamDepth;
    const u32 totalDepth = leftDepth + cfg.tableDepth;

    if (cfg.streamDepth != LEFT_STREAM_DEPTH && cfg.batched) {
        throw std::runtime_error("--batched supports only --stream-depth 3");
    }
    tcout << "puzzle " << cfg.puzzle << " <" << cfg.seedDepth << "," << cfg.streamDepth << ","
          << cfg.tableDepth << "> = "
          << totalDepth << " moves, " << (cfg.reverse ? "reverse" : "forward")
          << ", threads " << cfg.threads << ", table dir " << dir.string() << '\n';

    const Timer total;
    if (cfg.batched && cfg.shardCount > 1) {
        throw std::runtime_error("--batched does not support --shard");
    }
    std::vector<B1B2> middles;
    double probeSeconds = 0.0;
    double buildSeconds = 0.0;
    if (!cfg.recoverFrom.empty()) {
        tcout << "recovering from " << cfg.recoverFrom.size() << " shard file(s); no search\n";
        middles = readMiddlesFiles(cfg.recoverFrom, cfg);
        tcout << "unique middle states (" << leftDepth << " from left, " << cfg.tableDepth << " from table side): "
              << middles.size() << " (merged shards)\n";
    } else {
    std::vector<u64> counts = cfg.rebuild ? std::vector<u64>{} : loadPartitionCounts(dir);
    // No table unless one is already on disk (with its index) or the run needs it.
    const bool noTable = cfg.tableDepth == 6 && !cfg.reverse && !cfg.batched && !cfg.diskTable && !cfg.ramBuild
                         && cfg.d7Sample == 0 && (counts.empty() || !fs::exists(fp16MetaPath(dir)));
    // Depth 7: no table unless a complete depth-7 index is already on disk (6-4).
    const bool noTable7 = cfg.tableDepth == 7 && !cfg.reverse && !cfg.diskTable && !cfg.d7Subset()
                          && cfg.d7Sample == 0 && !fs::exists(d7Files(dir).meta);
    if (noTable || noTable7) {
        tcout << "no-table mode: nothing is written to " << dir.string() << "\n";
    } else if (counts.empty()) {
        const Timer buildTimer;
        counts = cfg.ramBuild ? buildDepth6InRam(rightRoot, dir, cfg, cfg.threads)
                              : buildDepth6OnDisk(rightRoot, dir, cfg.threads);
        buildSeconds = buildTimer.getSeconds();
        tcout << "BUILD depth-6 table: " << buildSeconds << "s\n";
    } else {
        u64 n = 0;
        for (const u64 c : counts) n += c;
        tcout << "reusing depth-6 table: " << n << " states (" << gbOf(n) << " GB)\n";
    }

    if (cfg.d7Sample > 0) {
        sampleDepth7(rightRoot, dir, counts, cfg.threads, cfg.d7Sample, cfg.d7PassBudget);
        return 0;
    }

    const Timer probeTimer;
    if (cfg.d7Subset()) {
        tcout << "depth-7 TEST SUBSET: only key partitions [" << cfg.d7FirstPartition << "," << cfg.d7End()
              << ") of " << PARTITION_COUNT << "\n";
    }
    middles = noTable
            ? probeNoTable(leftRoot, rightRoot, cfg, cfg.threads)
            : noTable7
            ? probeNoTableDepth7(leftRoot, rightRoot, cfg, cfg.threads)
            : cfg.tableDepth == 7
            ? probeDepth7(leftRoot, rightRoot, dir, counts, cfg, cfg.threads)
            : cfg.batched
            ? probeAllBatches(leftRoot, dir, counts, cfg, cfg.threads)
            : probeFingerprint(leftRoot, dir, counts, cfg, cfg.threads);
    probeSeconds = probeTimer.getSeconds();
    tcout << "unique middle states (" << leftDepth << " from left, " << cfg.tableDepth << " from table side): "
          << middles.size() << '\n';
    tcout << "PROBE phase: " << probeSeconds << "s\n";
    if (!cfg.middlesOut.empty()) {
        writeMiddlesFile(cfg.middlesOut, cfg, 0, middles);
    }
    if (cfg.shardCount > 1) {
        tcout << "SHARD " << cfg.shardIndex << "/" << cfg.shardCount << " complete: " << middles.size()
              << " middles (a partial search: no solution validation)\n";
    }
    }

    int status = 0;
    std::vector<std::string> known;
    if (!cfg.solutionsFile.empty()) {
        known = readSolutionLines(cfg.solutionsFile);
        if (cfg.d7Subset()) {
            // Only solutions whose middle falls in the probed partitions can be found.
            std::vector<std::string> kept;
            std::vector<u32> middleParts;
            for (const std::string& line : known) {
                const u32 part = static_cast<u32>(keyOf(middleOf(*pair, line, leftDepth)) >> PARTITION_SHIFT);
                middleParts.push_back(part);
                if (part >= cfg.d7FirstPartition && part < cfg.d7End()) {
                    kept.push_back(line);
                }
            }
            std::sort(middleParts.begin(), middleParts.end());
            middleParts.erase(std::unique(middleParts.begin(), middleParts.end()), middleParts.end());
            tcout << "known middles lie in partitions:";
            for (const u32 part : middleParts) {
                tcout << ' ' << part;
            }
            tcout << "\ntest subset: " << kept.size() << " of " << known.size()
                  << " known solutions have their middle in partitions [" << cfg.d7FirstPartition << ","
                  << cfg.d7End() << ")\n";
            known = kept;
        }
        // The middle sits leftDepth moves from the left root; counted from the
        // true start that is leftDepth (forward) or the table depth (reverse) moves in.
        const std::vector<B1B2> expected = expectedMiddles(*pair, known, cfg.reverse ? cfg.tableDepth : leftDepth);
        std::vector<B1B2> missing, extra;
        std::set_difference(expected.begin(), expected.end(), middles.begin(), middles.end(),
                            std::back_inserter(missing));
        std::set_difference(middles.begin(), middles.end(), expected.begin(), expected.end(),
                            std::back_inserter(extra));
        tcout << "middle validation vs " << cfg.solutionsFile << ": expected " << expected.size()
              << ", missing " << missing.size() << ", extra " << extra.size()
              << (missing.empty() && extra.empty() ? "  -> EXACT MATCH\n" : "  -> MISMATCH\n");
        status = missing.empty() && extra.empty() ? 0 : 2;
    }

    double recoverSeconds = 0.0;
    if (cfg.recover && !middles.empty()) {
        g_progress.phase("recovery", middles.size(), "full solutions through the middles");
        const Timer recoverTimer;
        BoardSolverFrontier solver(pair);
        fs::create_directories(dir / "solutions");
        solver.setWriteDirectory((dir / "solutions").string());
        const int found = recoverSolutions(solver, cfg, middles);
        recoverSeconds = recoverTimer.getSeconds();
        tcout << "RECOVERY: " << found << " solutions of " << totalDepth << " moves in "
              << recoverSeconds << "s (written under " << (dir / "solutions" / "levels").string() << ")\n";

        if (!known.empty()) {
            const std::set<std::string> knownSet(known.begin(), known.end());
            const bool same = knownSet == solver.getExpandedSolutions();
            tcout << "solution validation: known " << knownSet.size() << ", recovered "
                  << solver.getExpandedSolutions().size()
                  << (same ? "  -> IDENTICAL SET\n" : "  -> SETS DIFFER\n");
            if (!same) {
                status = 2;
            }
        }
    }

    tcout << "TOTAL: " << total.getSeconds() << "s (build " << buildSeconds << "s, probe "
          << probeSeconds << "s, recovery " << recoverSeconds << "s)\n";
    g_progress.phase("done", 0, "exit " + std::to_string(status));
    g_progress.stop();
    return status;
}
