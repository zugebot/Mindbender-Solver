#pragma once
// src/solver/ssd/no_table_d7.hpp
//
// No-table depth 7 (the default for --table-depth 7 without an index on
// disk): a count pass, then per pass the canonical 2-move stream from the
// depth-5 frontier fills the pass's buckets (keys routed to their owner
// thread and grouped, no atomics); hits confirmed by regeneration.
//
// Goal-side filter (normal boards, forward, bound on; --no-goal-filter turns it
// off): a meeting point is within L = seed + stream depth moves of the start,
// so depth-7 states whose rotation bound to the start exceeds L are dropped
// everywhere (count, fill, verify); exact. The bucket counts (2 B per bucket)
// go to a temp file on the system drive after the count pass and each pass
// reads only its partitions' slice, so they do not come out of the index
// budget (--d7-counts-ram keeps them in RAM). --d7-partitions/--d7-first-partition
// restrict the whole run to some key partitions (exactness tests).

#include "src/solver/ssd/no_table_d6.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // No-table depth-7 probe (default for --table-depth 7 without an index on disk)
    // ------------------------------------------------------------------
    // The canonical 2-move stream from each frontier state (the left side's
    // generator: no same-lane pairs, one order per commuting pair). Every
    // state at exact depth 7 is a depth-5 state plus such a pair.
    constexpr u32 NT7_STREAM_DEPTH = 2;

    // Set for the run by probeNoTableDepth7: drop streamed states whose bound
    // to the left root exceeds `limit` (no meeting point can be among them).
    struct GoalFilter {
        const lane_bound::LaneTables* tables = nullptr;
        int limit = 0;
    };
    inline GoalFilter g_goalFilter;

    template<typename Fn>
    struct StreamCallbackSink {
        Fn& fn;
        void operator()(JVec<B1B2>& states, JVec<u64>& hashes, const u32 count) {
            const GoalFilter f = g_goalFilter;
            for (u32 i = 0; i < count; ++i) {
                if (f.tables != nullptr && lane_bound::bound(states[i], *f.tables) > f.limit) {
                    continue;
                }
                fn(states[i], mixKey(hashes[i]));
            }
        }
    };

    // The 2-move descendants of frontier[i], the first move following the order
    // rules after the state's own last move (g_goalMoves, from buildLastMoves;
    // empty: every first move). Every exact depth-7 state is still produced.
    template<typename Fn>
    FORCEINLINE void streamTwo(const JVec<B1B2>& frontier, const std::size_t i, Fn&& fn) {
        StreamCallbackSink<Fn> sink{fn};
        const u8 prev = g_goalMoves.empty() ? perm_stream_detail::NO_PREV_MOVE : g_goalMoves[i];
        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, NT7_STREAM_DEPTH>(
                makeBoardFromState(frontier[i]), sink, 2048, prev);
    }

    // fn(thread, descendant, key) for every streamed descendant of `frontier`.
    template<typename Fn>
    void expandFrontierTwice(const JVec<B1B2>& frontier, const bool, const u32 threads, Fn&& fn) {
        const std::size_t n = frontier.size();
        constexpr std::size_t CHUNK = 1U << 10;
        parallelFor(threads, (n + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
            const std::size_t end = std::min(n, (c + 1) * CHUNK);
            for (std::size_t i = c * CHUNK; i < end; ++i) {
                streamTwo(frontier, i, [&](const B1B2& s, const u64 key) { fn(t, s, key); });
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
                    streamTwo(frontier, i, [&](const B1B2&, const u64 key) {
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

    inline std::vector<B1B2> probeNoTableDepth7(const Board& leftRoot, const Board& rightRoot, const Config& cfg,
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
        struct GoalMovesReset {
            ~GoalMovesReset() { std::vector<u8>().swap(g_goalMoves); }
        } goalMovesReset;
        if (cfg.seedLastMove) {
            Timer moveTimer;
            const u64 missing = buildLastMoves(rightRoot, 5, depth5, g_goalMoves, threads);
            tcout << "goal last moves: " << depth5.size() - missing << " of " << depth5.size() << " found in "
                  << moveTimer.getSeconds() << "s (the 2-move stream's first move follows the order rules)\n";
            if (missing != 0) {
                tcout << "  dropped " << dropUncoded(depth5, g_goalMoves)
                      << " goal states at depth < 5 (no descendant at exact depth 7)\n";
            }
        }

        static lane_bound::LaneTables leftTables;
        struct FilterReset {
            ~FilterReset() { g_goalFilter = GoalFilter{}; }
        } filterReset;
        if (cfg.goalFilter && !isFat && !cfg.reverse && cfg.boundMode != 0) {
            lane_bound::buildLaneTables(leftRoot.asB1B2(), leftTables, lane_bound::Kind::Rotation);
            g_goalFilter = GoalFilter{&leftTables, static_cast<int>(cfg.seedDepth + cfg.streamDepth)};
            tcout << "goal-side filter: depth-7 states with rotation h(start) > " << g_goalFilter.limit
                  << " dropped (no meeting point among them)\n";
        }
        const u32 P0 = cfg.d7FirstPartition, P1 = cfg.d7End();

        // Size the buckets from a sample of the stream's fan-out.
        double fanOut = 0.0;
        {
            const std::size_t step = std::max<std::size_t>(1, depth5.size() / 4096);
            u64 got = 0, parents = 0;
            for (std::size_t i = 0; i < depth5.size(); i += step, ++parents) {
                streamTwo(depth5, i, [&](const B1B2&, u64) { ++got; });
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
        BigArray<u16> bucketCounts;   // partitions [P0, P1), bucketsPer each
        bucketCounts.allocate(static_cast<u64>(P1 - P0) * bucketsPer);   // zero-filled
        tcout << "bucket counts: " << bucketCounts.size() * 2 / 1e9 << " GB, large pages "
              << (bucketCounts.largePages() ? "yes" : "no") << "\n";
        std::vector<std::vector<u64>> sample(threads);
        expandTwiceRouted(depth5, threads, P0, P1, fanOut, NT7_COUNT_BLOCK_KEYS, 6, [&](const u32 owner, const u64 key) {
            ++bucketCounts[((key >> PARTITION_SHIFT) - P0) * bucketsPer + ((key >> shift) & (bucketsPer - 1))];
            // 1/4096 of the keys (the top bits: partition 0's; mid bits when partition 0 is not probed)
            if (P0 == 0 ? (key >> (64 - 12)) == 0 : ((key >> 20) & 4095) == 0) {
                sample[owner].push_back(key);
            }
        });
        std::vector<u64> raw(PARTITION_COUNT, 0);
        u64 maxBucket = 0;
        parallelFor(threads, P1 - P0, [&](u32, const std::size_t i) {
            u64 sum = 0;
            for (u64 b = 0; b < bucketsPer; ++b) {
                sum += bucketCounts[i * bucketsPer + b];
            }
            raw[P0 + i] = sum;
        });
        u64 rawTotal = 0, rawMax = 0;
        for (u32 p = P0; p < P1; ++p) {
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
        meta.partitionBegin = P0;
        meta.partitionLimit = P1;

        // Spill the counts to the system drive; each pass reads its slice back.
        struct CountsFile {
            fs::path path;
            ~CountsFile() {
                std::error_code ec;
                if (!path.empty()) fs::remove(path, ec);
            }
        } countsFile;
        const u64 countsBytesPerPartition = bucketsPer * sizeof(u16);
        if (!cfg.d7CountsInRam) {
            Timer spillTimer;
            countsFile.path = fs::temp_directory_path()
                              / ("mb_d7counts_" + cfg.puzzle + "_" + std::to_string(GetCurrentProcessId()) + ".bin");
            FILE* f = std::fopen(countsFile.path.string().c_str(), "wb");
            if (f == nullptr) {
                throw std::runtime_error("cannot write " + countsFile.path.string());
            }
            const u64 bytes = bucketCounts.size() * sizeof(u16);
            const char* src = reinterpret_cast<const char*>(bucketCounts.data());
            for (u64 off = 0; off < bytes;) {
                const std::size_t n = static_cast<std::size_t>(std::min<u64>(bytes - off, 1ULL << 28));
                if (std::fwrite(src + off, 1, n, f) != n) {
                    std::fclose(f);
                    throw std::runtime_error("short write to " + countsFile.path.string());
                }
                off += n;
            }
            std::fclose(f);
            bucketCounts.reset();
            tcout << "bucket counts spilled to " << countsFile.path.string() << " (" << bytes / 1e9 << " GB in "
                  << spillTimer.getSeconds() << "s); each pass reads its partitions' slice\n";
        }
        BigArray<u16> passCounts;
        // counts of partition p (in [p0, p1) of the current pass)
        auto loadPassCounts = [&](const u32 p0, const u32 p1) -> const u16* {
            if (cfg.d7CountsInRam) {
                return bucketCounts.data() + static_cast<u64>(p0 - P0) * bucketsPer;
            }
            passCounts.ensure(static_cast<u64>(p1 - p0) * bucketsPer);
            FILE* f = std::fopen(countsFile.path.string().c_str(), "rb");
            if (f == nullptr || _fseeki64(f, static_cast<long long>(static_cast<u64>(p0 - P0) * countsBytesPerPartition), SEEK_SET) != 0) {
                if (f) std::fclose(f);
                throw std::runtime_error("cannot read " + countsFile.path.string());
            }
            const u64 bytes = static_cast<u64>(p1 - p0) * countsBytesPerPartition;
            char* dst = reinterpret_cast<char*>(passCounts.data());
            for (u64 off = 0; off < bytes;) {
                const std::size_t n = static_cast<std::size_t>(std::min<u64>(bytes - off, 1ULL << 28));
                if (std::fread(dst + off, 1, n, f) != n) {
                    std::fclose(f);
                    throw std::runtime_error("short read from " + countsFile.path.string());
                }
                off += n;
            }
            std::fclose(f);
            return passCounts.data();
        };
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
            const u16* passC = loadPassCounts(p0, p1);
            parallelFor(threads, parts, [&](u32, const std::size_t i) {
                const u16* c = passC + i * bucketsPer;
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
                const u16* c = passC + i * bucketsPer;
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

        // Keep back the fill's key blocks and, with spilled counts, the largest pass's count slice
        // (estimated from the budget: counts are ~2% of a partition's index).
        u64 reserve = 5'000'000'000ULL;
        if (!cfg.d7CountsInRam) {
            const u64 perPart = std::max<u64>(1, meta.indexBytes(rawTotal / std::max<u32>(1, P1 - P0), 1));
            const u64 guess = cfg.ramBytes != 0 ? cfg.ramBytes : defaultRamBudgetBytes(0);
            const u64 parts = std::min<u64>(P1 - P0, guess / perPart + 2);
            reserve += parts * countsBytesPerPartition;
        }
        const std::vector<KeyedState> candidates = streamAndProbe(leftRoot, IndexFiles{}, meta, raw, cfg, threads,
                                                                  &builder, reserve);
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

} // namespace ssd
