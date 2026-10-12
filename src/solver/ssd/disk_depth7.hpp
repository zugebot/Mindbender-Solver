#pragma once
// src/solver/ssd/disk_depth7.hpp
//
// The index-only depth-7 table on disk (6-4's), its probe, and
// --d7-sample (projects a full depth-7 build from one pass).

#include "src/solver/ssd/probe.hpp"
#include "src/solver/ssd/disk_depth6.hpp"

namespace ssd {

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
    inline void mergeD7Partition(const u64* keys, const std::vector<std::pair<u64, u64>>& runs,
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

    inline void buildDepth7Index(const Board& root, const fs::path& dir, const Config& cfg, const u32 threads) {
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
    inline std::vector<B1B2> probeDepth7(const Board& leftRoot, const Board& rightRoot, const fs::path& dir,
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
    inline void sampleDepth7(const Board& root,
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

} // namespace ssd
