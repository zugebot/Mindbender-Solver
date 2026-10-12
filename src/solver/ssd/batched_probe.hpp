#pragma once
// src/solver/ssd/batched_probe.hpp
//
// --batched: the oldest probe, the <S,3> left side against RAM-sized
// batches of on-disk partitions (exact states, no fingerprints).

#include "src/solver/ssd/common.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // Probe: <S, 3> left side against batches of partitions
    // ------------------------------------------------------------------

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

    inline std::vector<B1B2> probeAllBatches(const Board& leftRoot,
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

} // namespace ssd
