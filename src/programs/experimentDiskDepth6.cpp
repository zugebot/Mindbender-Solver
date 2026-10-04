// programs/experimentDiskDepth6.cpp
//
// Experiment: <S, 3, 6> meet-in-the-middle with the depth-6 right frontier on disk
// (S = --seed-depth, default 3: 12 moves; 4 gives 13 moves).
//
//   Build:  exact depth-5 frontier in RAM -> expand every state by one move ->
//           route each child to one of 256 partition files by the top 8 bits of
//           a mixed state key -> sort + de-dupe each partition independently.
//   Probe (default): load a compact fingerprint index of the whole table (u32
//           per state + bucket starts), stream the left side once, and confirm
//           fingerprint hits against the full states on disk.
//   Probe (--batched): load a contiguous range of sorted partitions that fits the RAM
//           budget, index it with a direct-address bucket table, then regenerate
//           the whole left side (exact depth-S seeds, canonical depth-3 stream)
//           and probe only states whose key falls in that range. Repeat per range.
//   Recover: full solutions through every middle via BoardSolverFrontier.
//
// All disk I/O is sequential. With --solutions, both the middle states and the
// recovered solution set are checked against a complete known solution file.
//
// Usage:
//   experimentDiskDepth6 <puzzle> [--out X:/mindbender_d6] [--batch 320000000]
//                        [--threads N] [--seed-depth S] [--reverse] [--rebuild]
//                        [--no-recover] [--batched] [--solutions file]
//                        [--d7-sample N [--d7-pass-budget STATES]]

#include "src/chuzzles/include.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
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
        u32 d7Sample = 0;
        u64 d7PassBudget = 375'000'000ULL;
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
    // Fingerprint mode: one probe pass over a compact in-RAM index
    // ------------------------------------------------------------------
    //
    // fp.bin    : u32 low 32 key bits of every table state, in key order
    // fpdir.bin : u64 start index of every bucket (bucket = top bits of key)
    // fpmeta.txt: "<bucketBits> <stateCount>"
    //
    // A probe whose bucket holds a matching fingerprint becomes a candidate;
    // candidates are confirmed against the full states on disk, so false
    // fingerprint matches (~bucket size / 2^32 per probe) cost one tiny read.

    struct FingerprintIndex {
        u32 bucketBits = 0;
        u32 shift = 0;
        u64 stateCount = 0;
        std::vector<u32> fingerprints;
        std::vector<u64> starts;   // bucketCount + 1

        // Prefetch bucket offsets, then fingerprint runs, then compare, so the
        // DRAM misses of a whole group overlap.
        void probeGroup(const B1B2* probes, const u64* keys, const u32 count,
                        std::vector<KeyedState>& candidates) const {
            u64 bucket[PROBE_GROUP], lo[PROBE_GROUP], hi[PROBE_GROUP];
            for (u32 i = 0; i < count; ++i) {
                bucket[i] = keys[i] >> shift;
                __builtin_prefetch(&starts[bucket[i]]);
            }
            for (u32 i = 0; i < count; ++i) {
                lo[i] = starts[bucket[i]];
                hi[i] = starts[bucket[i] + 1];
                if (lo[i] < hi[i]) {
                    __builtin_prefetch(&fingerprints[lo[i]]);
                }
            }
            for (u32 i = 0; i < count; ++i) {
                const u32 fp = static_cast<u32>(keys[i]);
                for (u64 j = lo[i]; j < hi[i]; ++j) {
                    if (fingerprints[j] == fp) {
                        candidates.push_back({keys[i], probes[i]});
                        break;
                    }
                }
            }
        }
    };

    fs::path fpPath(const fs::path& dir) { return dir / "fp.bin"; }
    fs::path fpDirPath(const fs::path& dir) { return dir / "fpdir.bin"; }
    fs::path fpMetaPath(const fs::path& dir) { return dir / "fpmeta.txt"; }

    void buildFingerprintIndex(const fs::path& dir, const std::vector<u64>& counts, const u32 threads) {
        Timer timer;
        u64 n = 0;
        for (const u64 c : counts) {
            n += c;
        }
        u32 bucketBits = PARTITION_BITS;
        while (bucketBits < 40 && (n >> bucketBits) > 8) {
            ++bucketBits;
        }
        const u32 shift = 64U - bucketBits;
        const u64 bucketCount = 1ULL << bucketBits;

        std::vector<u64> starts(bucketCount + 1, 0);
        std::FILE* fpOut = std::fopen(fpPath(dir).string().c_str(), "wb");
        if (fpOut == nullptr) {
            throw std::runtime_error("failed to open fp.bin");
        }

        u64 globalIndex = 0, nextBucket = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            const std::vector<B1B2> states = readAll<B1B2>(sortedPath(dir, p));
            std::vector<u64> keys(states.size());
            parallelFor(threads, (states.size() + 65535) / 65536, [&](u32, const std::size_t c) {
                const std::size_t end = std::min(states.size(), (c + 1) * 65536);
                for (std::size_t i = c * 65536; i < end; ++i) {
                    keys[i] = keyOf(states[i]);
                }
            });
            std::vector<u32> fps(states.size());
            for (std::size_t i = 0; i < states.size(); ++i) {
                const u64 bucket = keys[i] >> shift;
                while (nextBucket <= bucket) {
                    starts[nextBucket++] = globalIndex + i;
                }
                fps[i] = static_cast<u32>(keys[i]);
            }
            if (std::fwrite(fps.data(), sizeof(u32), fps.size(), fpOut) != fps.size()) {
                throw std::runtime_error("fp.bin write failed");
            }
            globalIndex += states.size();
        }
        while (nextBucket <= bucketCount) {
            starts[nextBucket++] = globalIndex;
        }
        std::fclose(fpOut);
        writeAll(fpDirPath(dir), starts.data(), starts.size());
        std::ofstream(fpMetaPath(dir)) << bucketBits << ' ' << n << '\n';

        tcout << "BUILD fingerprint index: " << n << " fingerprints + 2^" << bucketBits << " buckets ("
              << (n * 4.0 + (bucketCount + 1) * 8.0) / 1e9 << " GB) in " << timer.getSeconds() << "s\n";
    }

    FingerprintIndex loadFingerprintIndex(const fs::path& dir) {
        FingerprintIndex index;
        std::ifstream meta(fpMetaPath(dir));
        meta >> index.bucketBits >> index.stateCount;
        index.shift = 64U - index.bucketBits;
        index.fingerprints = readAll<u32>(fpPath(dir));
        index.starts = readAll<u64>(fpDirPath(dir));
        if (index.fingerprints.size() != index.stateCount
            || index.starts.size() != (1ULL << index.bucketBits) + 1) {
            throw std::runtime_error("fingerprint index files are inconsistent");
        }
        return index;
    }

    std::vector<B1B2> probeFingerprint(const Board& leftRoot,
                                       const fs::path& dir,
                                       const std::vector<u64>& counts,
                                       const Config& cfg,
                                       const u32 threads) {
        if (!fs::exists(fpMetaPath(dir)) || cfg.rebuild) {
            buildFingerprintIndex(dir, counts, threads);
        }

        Timer loadTimer;
        const FingerprintIndex index = loadFingerprintIndex(dir);
        const double gb = (index.fingerprints.size() * 4.0 + index.starts.size() * 8.0) / 1e9;
        const double loadSeconds = loadTimer.getSeconds();
        tcout << "loaded fingerprint index: " << gb << " GB in " << loadSeconds << "s ("
              << gb / loadSeconds * 1000.0 << " MB/s)\n";

        Timer seedTimer;
        JVec<B1B2> seeds;
        buildSeeds(leftRoot, cfg.seedDepth, seeds);
        tcout << "left seeds (exact depth " << cfg.seedDepth << "): " << seeds.size()
              << " in " << seedTimer.getSeconds() << "s\n";

        // Single pass: every left state is probed exactly once.
        Timer probeTimer;
        std::mutex candidateLock;
        std::vector<KeyedState> candidates;
        std::vector<u64> streamed(threads, 0);
        parallelFor(threads, seeds.size(), [&](const u32 t, const std::size_t s) {
            struct Sink {
                const FingerprintIndex& index;
                B1B2 groupStates[PROBE_GROUP];
                u64 groupKeys[PROBE_GROUP];
                u32 groupCount = 0;
                std::vector<KeyedState> found;
                u64 streamed = 0;

                void flush() {
                    index.probeGroup(groupStates, groupKeys, groupCount, found);
                    groupCount = 0;
                }

                void operator()(JVec<B1B2>& chunkStates, JVec<u64>& chunkHashes, const u32 count) {
                    streamed += count;
                    for (u32 i = 0; i < count; ++i) {
                        groupStates[groupCount] = chunkStates[i];
                        groupKeys[groupCount] = mixKey(chunkHashes[i]);
                        if (++groupCount == PROBE_GROUP) {
                            flush();
                        }
                    }
                }
            } sink{index, {}, {}, 0, {}, 0};

            const Board seedBoard = makeBoardFromState(seeds[s]);
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, LEFT_STREAM_DEPTH>(seedBoard, sink, 4096);
            sink.flush();
            streamed[t] += sink.streamed;
            if (!sink.found.empty()) {
                std::lock_guard<std::mutex> lock(candidateLock);
                candidates.insert(candidates.end(), sink.found.begin(), sink.found.end());
            }
        });
        const double probeSeconds = probeTimer.getSeconds();
        u64 streamedTotal = 0;
        for (const u64 n : streamed) {
            streamedTotal += n;
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](const KeyedState& a, const KeyedState& b) {
                                         return a.key == b.key && a.state == b.state;
                                     }),
                         candidates.end());
        tcout << "stream+probe: " << streamedTotal << " left states in " << probeSeconds << "s ("
              << static_cast<double>(streamedTotal) / probeSeconds / 1e6 << " M/s), "
              << candidates.size() << " unique candidates\n";

        // Confirm each candidate against the full states of its bucket on disk.
        Timer verifyTimer;
        std::vector<u64> partitionStart(PARTITION_COUNT + 1, 0);
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            partitionStart[p + 1] = partitionStart[p] + counts[p];
        }
        std::vector<B1B2> middles;
        u64 falseMatches = 0;
        std::size_t i = 0;
        while (i < candidates.size()) {
            const u32 p = static_cast<u32>(candidates[i].key >> PARTITION_SHIFT);
            std::ifstream in(sortedPath(dir, p), std::ios::binary);
            for (; i < candidates.size() && (candidates[i].key >> PARTITION_SHIFT) == p; ++i) {
                const u64 bucket = candidates[i].key >> index.shift;
                const u64 lo = index.starts[bucket];
                const u64 hi = index.starts[bucket + 1];
                std::vector<B1B2> run(hi - lo);
                in.seekg(static_cast<std::streamoff>((lo - partitionStart[p]) * sizeof(B1B2)));
                in.read(reinterpret_cast<char*>(run.data()), static_cast<std::streamsize>(run.size() * sizeof(B1B2)));
                if (!in) {
                    throw std::runtime_error("failed to read candidate bucket");
                }
                if (std::find(run.begin(), run.end(), candidates[i].state) != run.end()) {
                    middles.push_back(candidates[i].state);
                } else {
                    ++falseMatches;
                }
            }
        }
        std::sort(middles.begin(), middles.end());
        middles.erase(std::unique(middles.begin(), middles.end()), middles.end());
        tcout << "verify: " << middles.size() << " confirmed, " << falseMatches
              << " fingerprint false matches, " << verifyTimer.getSeconds() << "s\n";
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
    std::vector<B1B2> expectedMiddles(const BoardPair& pair,
                                      const std::vector<std::string>& solutions,
                                      const std::size_t prefixMoves) {
        std::vector<B1B2> out;
        const bool isFat = pair.getStartState().getFatBool();
        for (const std::string& line : solutions) {
            const std::vector<u8> moves = isFat ? Memory::parseFatMoveString(line)
                                                : Memory::parseNormMoveString(line);
            if (moves.size() < prefixMoves) {
                throw std::runtime_error("solution shorter than the middle depth: " + line);
            }
            Board board = pair.getStartState();
            for (std::size_t i = 0; i < prefixMoves; ++i) {
                allActStructList[moves[i]].action(board);
            }
            out.push_back(board.asB1B2());
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    template<int TOTAL_LEFT_DEPTH>
    int recoverWithLeftDepth(BoardSolverFrontier& solver, const bool reverse,
                             const std::vector<B1B2>& middles, const u32 threads) {
        constexpr int TABLE_DEPTH = 6;
        if (reverse) {
            return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, true>(middles, static_cast<int>(threads));
        }
        return solver.recoverSolutionsFromMiddles<TOTAL_LEFT_DEPTH, TABLE_DEPTH, false>(middles, static_cast<int>(threads));
    }

    int recoverSolutions(BoardSolverFrontier& solver, const Config& cfg, const std::vector<B1B2>& middles) {
        switch (cfg.seedDepth + LEFT_STREAM_DEPTH) {
            case 4: return recoverWithLeftDepth<4>(solver, cfg.reverse, middles, cfg.threads);
            case 5: return recoverWithLeftDepth<5>(solver, cfg.reverse, middles, cfg.threads);
            case 6: return recoverWithLeftDepth<6>(solver, cfg.reverse, middles, cfg.threads);
            case 7: return recoverWithLeftDepth<7>(solver, cfg.reverse, middles, cfg.threads);
            case 8: return recoverWithLeftDepth<8>(solver, cfg.reverse, middles, cfg.threads);
            default: throw std::runtime_error("unsupported left depth for recovery");
        }
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
            else if (a == "--solutions") cfg.solutionsFile = next();
            else if (a == "--d7-sample") cfg.d7Sample = static_cast<u32>(std::stoul(next()));
            else if (a == "--d7-pass-budget") cfg.d7PassBudget = std::stoull(next());
            else cfg.puzzle = a;
        }
        if (cfg.threads == 0) {
            cfg.threads = std::max(1U, std::thread::hardware_concurrency());
        }
        return cfg;
    }

} // namespace

int main(const int argc, char** argv) {
    const Config cfg = parseArgs(argc, argv);
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
    const u32 totalDepth = leftDepth + 6;

    tcout << "puzzle " << cfg.puzzle << " <" << cfg.seedDepth << "," << LEFT_STREAM_DEPTH << ",6> = "
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
    const std::vector<B1B2> middles = cfg.batched
            ? probeAllBatches(leftRoot, dir, counts, cfg, cfg.threads)
            : probeFingerprint(leftRoot, dir, counts, cfg, cfg.threads);
    const double probeSeconds = probeTimer.getSeconds();
    tcout << "unique middle states (" << leftDepth << " from left, 6 from table side): "
          << middles.size() << '\n';
    tcout << "PROBE phase: " << probeSeconds << "s\n";

    int status = 0;
    std::vector<std::string> known;
    if (!cfg.solutionsFile.empty()) {
        known = readSolutionLines(cfg.solutionsFile);
        // The middle sits leftDepth moves from the left root; counted from the
        // true start that is leftDepth (forward) or 6 (reverse) moves in.
        const std::vector<B1B2> expected = expectedMiddles(*pair, known, cfg.reverse ? 6 : leftDepth);
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
    return status;
}
