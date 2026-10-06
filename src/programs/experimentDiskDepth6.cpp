// programs/experimentDiskDepth6.cpp
//
// Experiment: <S, 3, T> meet-in-the-middle with the right frontier on disk
// (S = --seed-depth, default 3; T = --table-depth 6 (default) or 7).
// <3,3,6> = 12 moves, <4,3,6> = 13, <5,3,6> = 14, <5,3,7> = 15.
//
//   Build:  exact depth-5 frontier in RAM -> expand every state by one move ->
//           route each child to one of 256 partition files by the top 8 bits of
//           a mixed state key -> sort + de-dupe each partition independently.
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
//
// --progress-file: a background thread rewrites a small JSON status (phase,
// done/total units, elapsed) every few seconds for python/scheduler; workers
// only add to an atomic counter once per seed / partition. Log lines are
// flushed as they are written, so a redirected log can be followed live.
//                        [--d7-sample N [--d7-pass-budget STATES]]

#include "src/chuzzles/include.hpp"
#include "src/utils/progress_reporter.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <fstream>
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

        // Bloom block, then bucket offsets, then fingerprints; each stage
        // prefetches the next stage's lines for the whole group. Returns the
        // number of probes that passed the Bloom filter.
        u32 probeGroup(const B1B2* probes, const u64* keys, const u32 count,
                       std::vector<KeyedState>& candidates) const {
            const u64 bucketsPer = meta.bucketsPerPartition();
            const u64 dirPer = bucketsPer + 1;
            const u32 shift = meta.bucketShift();

            u64 h[PROBE_GROUP];
            const u32* blocks[PROBE_GROUP];
            for (u32 i = 0; i < count; ++i) {
                const u64 part = (keys[i] >> PARTITION_SHIFT) - pBegin;
                h[i] = bloomHash(keys[i]);
                blocks[i] = bloom.data()
                            + (part * meta.bloomBlocksPerPartition
                               + bloomBlockInPartition(h[i], meta.bloomBlocksPerPartition)) * 8;
                __builtin_prefetch(blocks[i]);
            }
            u32 pass[PROBE_GROUP];
            u32 passCount = 0;
            for (u32 i = 0; i < count; ++i) {
                if (bloomTest(blocks[i], h[i])) {
                    pass[passCount++] = i;
                }
            }

            const u32* dirSlot[PROBE_GROUP];
            u64 base[PROBE_GROUP], lo[PROBE_GROUP], hi[PROBE_GROUP];
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
    };

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

    // Streams the whole left side once per RAM-sized range of the index's
    // partitions and returns the unique fingerprint hits (key, state), sorted.
    std::vector<KeyedState> streamAndProbe(const Board& leftRoot,
                                           const IndexFiles& files,
                                           const FingerprintMeta& meta,
                                           const std::vector<u64>& counts,
                                           const Config& cfg,
                                           const u32 threads) {
        // Partition ranges whose index fits the RAM budget.
        const u32 pFirst = meta.partitionBegin, pLimit = meta.partitionLimit;
        // Seeds: exact depth-S frontier, ~ (25.4^S) states of 16 B (+ hashes while built).
        const u64 seedBytes = static_cast<u64>(std::pow(25.4, cfg.seedDepth) * 2 * sizeof(B1B2));
        const u64 budget = cfg.ramBytes != 0 ? cfg.ramBytes : defaultRamBudgetBytes(seedBytes);
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
            index.reserve(meta, maxStates, maxParts);
        }

        Timer seedTimer;
        JVec<B1B2> seeds;
        buildSeeds(leftRoot, cfg.seedDepth, seeds);
        tcout << "left seeds (exact depth " << cfg.seedDepth << "): " << seeds.size()
              << " in " << seedTimer.getSeconds() << "s\n";

        std::vector<KeyedState> candidates;
        double loadSeconds = 0.0, probeSeconds = 0.0;
        u64 streamedTotal = 0, inRangeTotal = 0, bloomPassTotal = 0;
        g_progress.phase("probe", static_cast<u64>(passes.size()) * seeds.size(),
                         "pass 1/" + std::to_string(passes.size()) + " loading index");

        for (std::size_t pass = 0; pass < passes.size(); ++pass) {
            const auto [pBegin, pEnd] = passes[pass];
            const std::string passLabel = "pass " + std::to_string(pass + 1) + "/" + std::to_string(passes.size());
            g_progress.detail(passLabel + " loading index");

            Timer loadTimer;
            index.load(files, meta, counts, pBegin, pEnd);
            const double passLoad = loadTimer.getSeconds();
            loadSeconds += passLoad;
            g_progress.detail(passLabel + " streaming");

            Timer probeTimer;
            std::mutex candidateLock;
            std::vector<u64> streamed(threads, 0), inRange(threads, 0), bloomPasses(threads, 0);
            parallelFor(threads, seeds.size(), [&](const u32 t, const std::size_t s) {
                struct Sink {
                    const FingerprintRange& index;
                    B1B2 groupStates[PROBE_GROUP];
                    u64 groupKeys[PROBE_GROUP];
                    u32 groupCount = 0;
                    std::vector<KeyedState> found;
                    u64 streamed = 0, inRange = 0, bloomPasses = 0;

                    void flush() {
                        bloomPasses += index.probeGroup(groupStates, groupKeys, groupCount, found);
                        groupCount = 0;
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
                            groupStates[groupCount] = chunkStates[i];
                            groupKeys[groupCount] = key;
                            if (++groupCount == PROBE_GROUP) {
                                flush();
                            }
                        }
                    }
                } sink{index, {}, {}, 0, {}, 0, 0, 0};

                const Board seedBoard = makeBoardFromState(seeds[s]);
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_STREAM_DEPTH>(seedBoard, sink, 4096);
                sink.flush();
                streamed[t] += sink.streamed;
                inRange[t] += sink.inRange;
                bloomPasses[t] += sink.bloomPasses;
                if (!sink.found.empty()) {
                    std::lock_guard<std::mutex> lock(candidateLock);
                    candidates.insert(candidates.end(), sink.found.begin(), sink.found.end());
                }
                g_progress.add();   // one relaxed atomic add per seed (~ms of work)
            });
            const double passProbe = probeTimer.getSeconds();
            probeSeconds += passProbe;

            u64 passStreamed = 0, passInRange = 0, passBloom = 0;
            for (u32 t = 0; t < threads; ++t) {
                passStreamed += streamed[t];
                passInRange += inRange[t];
                passBloom += bloomPasses[t];
            }
            streamedTotal += passStreamed;
            inRangeTotal += passInRange;
            bloomPassTotal += passBloom;
            tcout << "  pass " << (pass + 1) << "/" << passes.size() << " partitions [" << pBegin << ","
                  << pEnd << "): load " << index.bytes() / 1e9 << " GB ("
                  << (index.largePages() ? "large pages" : "normal pages") << ") in " << passLoad << "s, stream+probe "
                  << passStreamed << " left states in " << passProbe << "s ("
                  << static_cast<double>(passStreamed) / passProbe / 1e6 << " M/s), Bloom pass "
                  << 100.0 * static_cast<double>(passBloom) / static_cast<double>(std::max<u64>(1, passInRange))
                  << "%\n";
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
        if (!fs::exists(fp16MetaPath(dir)) || cfg.rebuild) {
            buildFingerprintIndex(dir, counts, threads);
        }
        const FingerprintMeta meta = readFingerprintMeta(fp16MetaPath(dir));
        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, d6Files(dir), meta, counts, cfg, threads);
        return verifyAgainstTable(dir, meta, candidates);
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
        switch (cfg.seedDepth + LEFT_STREAM_DEPTH) {
            case 4: return recoverWithLeftDepth<4, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 5: return recoverWithLeftDepth<5, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 6: return recoverWithLeftDepth<6, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 7: return recoverWithLeftDepth<7, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
            case 8: return recoverWithLeftDepth<8, TABLE_DEPTH>(solver, cfg.reverse, middles, cfg.threads);
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
    const fs::path dir = cfg.outRoot / (cfg.puzzle + (cfg.reverse ? "_start_d6" : "_goal_d6"));
    const u32 leftDepth = cfg.seedDepth + LEFT_STREAM_DEPTH;
    const u32 totalDepth = leftDepth + cfg.tableDepth;

    tcout << "puzzle " << cfg.puzzle << " <" << cfg.seedDepth << "," << LEFT_STREAM_DEPTH << ","
          << cfg.tableDepth << "> = "
          << totalDepth << " moves, " << (cfg.reverse ? "reverse" : "forward")
          << ", threads " << cfg.threads << ", table dir " << dir.string() << '\n';

    const Timer total;
    std::vector<u64> counts = cfg.rebuild ? std::vector<u64>{} : loadPartitionCounts(dir);
    double buildSeconds = 0.0;
    if (counts.empty()) {
        const Timer buildTimer;
        counts = buildDepth6OnDisk(rightRoot, dir, cfg.threads);
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
    const std::vector<B1B2> middles = cfg.tableDepth == 7
            ? probeDepth7(leftRoot, rightRoot, dir, counts, cfg, cfg.threads)
            : cfg.batched
            ? probeAllBatches(leftRoot, dir, counts, cfg, cfg.threads)
            : probeFingerprint(leftRoot, dir, counts, cfg, cfg.threads);
    const double probeSeconds = probeTimer.getSeconds();
    tcout << "unique middle states (" << leftDepth << " from left, " << cfg.tableDepth << " from table side): "
          << middles.size() << '\n';
    tcout << "PROBE phase: " << probeSeconds << "s\n";

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
