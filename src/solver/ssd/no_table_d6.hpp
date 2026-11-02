#pragma once
// src/solver/ssd/no_table_d6.hpp
//
// No-table depth 6 (the default for --table-depth 6): the depth-5 frontier
// stays in RAM, each pass builds its index slice from regenerated depth-6 keys,
// and hits are confirmed by regenerating depth 6 once more. Nothing is written.
// With --gpu (normal boards) the GPU counts the children per partition and, per
// group of partitions, generates, sorts and de-duplicates the keys and writes the
// group's fingerprints, bucket offsets and Bloom blocks straight into the index
// arrays (mbgpu_build_index: the same layout as the CPU loop below).

#include "src/solver/ssd/probe.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // No-table depth-6 probe (default)
    // ------------------------------------------------------------------
    // fn(thread, child, key) for every one-move child of `frontier` that the
    // canonical stream would visit first after the state's own last move
    // (g_goalMoves, from buildLastMoves; empty: every child). Every exact depth-6
    // state is produced either way; the first-move rule only drops children that
    // another depth-5 state produces in the allowed order (~60 -> ~41 per state).
    template<typename Fn>
    void expandFrontier(const JVec<B1B2>& frontier, const u32 threads, Fn&& fn) {
        const std::size_t n = frontier.size();
        constexpr std::size_t CHUNK = 1U << 14;
        const std::vector<u8>& moves = g_goalMoves;
        parallelFor(threads, (n + CHUNK - 1) / CHUNK, [&](const u32 t, const std::size_t c) {
            const std::size_t end = std::min(n, (c + 1) * CHUNK);
            for (std::size_t i = c * CHUNK; i < end; ++i) {
                const u8 prev = moves.empty() ? perm_stream_detail::NO_PREV_MOVE : moves[i];
                perm_stream_detail::forEachFirstMove(frontier[i], prev, [&](u8, const B1B2& child) {
                    fn(t, child, mixKey(StateHash::computeHash(child)));
                });
            }
        });
    }

    inline std::vector<B1B2> probeNoTable(const Board& leftRoot, const Board& rightRoot, const Config& cfg,
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
        struct GoalMovesReset {
            ~GoalMovesReset() { std::vector<u8>().swap(g_goalMoves); }
        } goalMovesReset;
        if (cfg.seedLastMove) {
            Timer moveTimer;
            const u64 missing = buildLastMoves(rightRoot, 5, depth5, g_goalMoves, threads);
            tcout << "goal last moves: " << depth5.size() - missing << " of " << depth5.size() << " found in "
                  << moveTimer.getSeconds() << "s (depth-6 children follow the order rules)\n";
            if (missing != 0) {
                tcout << "  dropped " << dropUncoded(depth5, g_goalMoves)
                      << " goal states at depth < 5 (no child at exact depth 6)\n";
            }
        }

        const bool gpuKeys = !cfg.gpuDevice.empty() && !isFat;
        GpuModule* gpu = gpuKeys ? &ensureGpu(cfg) : nullptr;
        const auto* d5States = reinterpret_cast<const uint64_t*>(depth5.data());
        const u8* d5Moves = g_goalMoves.empty() ? nullptr : g_goalMoves.data();

        // Exact raw child count per partition.
        std::vector<u64> raw(PARTITION_COUNT, 0);
        if (gpuKeys) {
            if (gpu->childCounts(d5States, d5Moves, depth5.size(), raw.data()) != 0) {
                throw std::runtime_error("GPU: " + gpu->error());
            }
        } else {
            std::vector<std::vector<u64>> rawPer(threads, std::vector<u64>(PARTITION_COUNT, 0));
            expandFrontier(depth5, threads, [&](const u32 t, const B1B2&, const u64 key) {
                ++rawPer[t][key >> PARTITION_SHIFT];
            });
            for (u32 p = 0; p < PARTITION_COUNT; ++p) {
                for (u32 th = 0; th < threads; ++th) {
                    raw[p] += rawPer[th][p];
                }
            }
        }
        u64 rawTotal = 0, rawMax = 0;
        for (u32 p = 0; p < PARTITION_COUNT; ++p) {
            rawTotal += raw[p];
            rawMax = std::max(rawMax, raw[p]);
        }

        // De-dupe ratio from partition 0 (partitions are near-uniform).
        double ratio = 1.0;
        if (gpuKeys) {
            std::vector<u64> keys0(std::max<u64>(1, raw[0]));
            u64 u0 = 0;
            if (gpu->childKeys(d5States, d5Moves, depth5.size(), 0, 1, keys0.data(), keys0.size(), &u0) != 0) {
                throw std::runtime_error("GPU: " + gpu->error());
            }
            ratio = static_cast<double>(u0) / static_cast<double>(std::max<u64>(1, raw[0]));
        } else {
            std::vector<u64> keys0;
            keys0.reserve(raw[0]);
            std::mutex m0;
            std::vector<std::vector<u64>> local(threads);
            expandFrontier(depth5, threads, [&](const u32 t, const B1B2&, const u64 key) {
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
            const u64 avail = gpuKeys ? gpu->childKeyCapacity() * sizeof(u64) * 9 / 10
                            : cfg.ramBytes != 0 ? keyReserve
                                                : std::max(keyReserve, defaultRamBudgetBytes(indexBytes));
            u64 fill = 0;   // fingerprints written so far in this range
            double keySeconds = 0.0;   // generating + sorting keys (GPU or CPU); the rest writes the index
            for (u32 s0 = p0; s0 < p1;) {
                u32 s1 = s0;
                u64 bytes = 0;
                while (s1 < p1 && (s1 == s0 || bytes + raw[s1] * sizeof(u64) <= avail)) {
                    bytes += raw[s1++] * sizeof(u64);
                }
                const u32 gN = s1 - s0;
                if (s0 == p0) {
                    tcout << "  index build" << (gpuKeys ? " (GPU)" : "") << ": " << avail / 1e9
                          << " GB for keys -> about " << (parts + gN - 1) / gN << " regeneration(s) of depth 6\n";
                }
                // sorted unique keys of each partition of the group: sortedKeys[i], uniq[i]
                std::vector<const u64*> sortedKeys(gN, nullptr);
                std::vector<u64> uniq(gN, 0);
                std::vector<std::unique_ptr<u64[]>> keys(gN);
                Timer keyTimer;
                if (gpuKeys) {
                    const u64 part = s0 - p0;
                    if (gpu->buildIndex(d5States, d5Moves, depth5.size(), s0, s1, meta.bucketBits,
                                        meta.bloomBlocksPerPartition, idx.fingerprints.data() + fill, cap - fill,
                                        idx.starts.data() + part * dirPer, idx.bloom.data() + part * bloomPer,
                                        uniq.data()) != 0) {
                        throw std::runtime_error("GPU: " + gpu->error());
                    }
                } else {
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
                expandFrontier(depth5, threads, [&](const u32 t, const B1B2&, const u64 key) {
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
                parallelFor(threads, gN, [&](u32, const std::size_t i) {
                    u64* a = keys[i].get();
                    const u64 n = got[i].load();
                    std::sort(a, a + n);
                    uniq[i] = static_cast<u64>(std::unique(a, a + n) - a);
                    sortedKeys[i] = a;
                });
                }
                keySeconds += keyTimer.getSeconds();
                std::vector<u64> offset(gN, 0);
                for (u32 i = 0; i < gN; ++i) {
                    offset[i] = fill;
                    fill += uniq[i];
                    if (fill > cap) {
                        throw std::runtime_error("no-table index estimate too small");
                    }
                    idx.partitionFirst.push_back(fill);
                }
                parallelFor(threads, gpuKeys ? 0 : gN, [&](u32, const std::size_t i) {   // GPU: already written
                    const u64* a = sortedKeys[i];
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
                  << " unique states in " << timer.getSeconds() << "s (keys " << keySeconds << "s"
                  << (gpuKeys ? " on the GPU, with the index" : "") << ", index writing "
                  << timer.getSeconds() - keySeconds << "s)\n";
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
            expandFrontier(depth5, threads, [&](u32, const B1B2& child, const u64 key) {
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

} // namespace ssd
