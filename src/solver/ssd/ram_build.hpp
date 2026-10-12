#pragma once
// src/solver/ssd/ram_build.hpp
//
// --ram-build: the depth-6 disk table built with in-RAM sorting (no raw
// spill files), writing the sorted partitions and their fingerprint index.

#include "src/solver/ssd/disk_depth6.hpp"
#include "src/solver/ssd/fingerprint_index.hpp"

namespace ssd {

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
    inline std::vector<u64> buildDepth6InRam(const Board& root, const fs::path& dir, const Config& cfg, const u32 threads) {
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

} // namespace ssd
