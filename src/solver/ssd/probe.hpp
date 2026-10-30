#pragma once
// src/solver/ssd/probe.hpp
//
// The probe: stream every left seed against the goal-side index, once per
// RAM-sized range of partitions, with the lane bound / pattern databases
// pruning the stream (audited and timed per run), then confirm the hits against
// the disk table (probeFingerprint) or hand them back (no-table modes).

#include "src/solver/ssd/probe_sink.hpp"
#include "src/solver/bounds/audit.hpp"
#include "src/solver/bounds/calibrate.hpp"
#include "src/solver/bounds/parent_prune.hpp"

namespace ssd {

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
        std::vector<u8> seedMoves;   // each seed's smallest-path last move (buildLastMoves, common.hpp)
        if (cfg.seedLastMove) {
            Timer moveTimer;
            const u64 missing = buildLastMoves(leftRoot, cfg.seedDepth, seeds, seedMoves, threads);
            tcout << "seed last moves: " << seeds.size() - missing << " of " << seeds.size() << " found in "
                  << moveTimer.getSeconds() << "s (first streamed move follows the order rules)\n";
            if (missing != 0) {
                // the uncoded seeds are shallower than the seed depth: not on any solution of this length
                tcout << "  dropped " << dropUncoded(seeds, seedMoves) << " seeds at depth < " << cfg.seedDepth
                      << " (never the depth-" << cfg.seedDepth << " state of a shortest solution)\n";
            }
        }

        // Colour-mix bound: eligible modes after the audit (see header).
        // boundMode: 0 off, 1 leaf, 2 leaf + parent; -1 = decide by timing.
        static lane_bound::LaneTables boundTables;   // read by bounds::LanePrune
        const int boundLimit = static_cast<int>(cfg.tableDepth);
        int boundMode = 0;
        std::vector<int> boundModes;   // modes to time (auto)
        static const char* BOUND_NAMES[7] = {"off", "leaf", "both", "pdb", "pdbleaf", "parent", "pdbparent"};
        if (cfg.boundMode != 0 && !cfg.reverse) {
            if (g_rightRoot.getFatBool()) {
                tcout << "bound: off (2x2 board; the halved bound prunes nothing)\n";
            } else {
                Timer auditTimer;
                lane_bound::buildLaneTables(g_rightRoot, boundTables, cfg.boundKind);
                const bounds::BoundAudit audit = bounds::auditLaneBound<STREAM_DEPTH>(seeds, boundTables, boundLimit, threads, g_rightRoot);
                tcout << "bound (" << lane_bound::kindName(cfg.boundKind) << "): audit " << (audit.ok ? "ok" : "FAILED (" + audit.why + ")") << ": " << audit.nodes
                      << " nodes, " << audit.leaves << " leaves, " << audit.edges << " edges (|dh| <= 1), "
                      << audit.streamChecks << " wanted leaves identical pruned vs full; sampled skip: leaf "
                      << 100.0 * audit.leafSkip << "%, whole last level " << 100.0 * audit.parentSkip << "%; tables "
                      << boundTables.bytes << " B, " << auditTimer.getSeconds() << "s\n";
                if (audit.ok) {
                    bounds::g_laneTables = &boundTables;
                    bounds::g_laneLimit = boundLimit;
                    bool pdbReady = false;
                    const bool wantPdb = cfg.pdbMode == 1
                                         || (cfg.pdbMode == -1 && cfg.seedDepth >= 5 && audit.parentSkip < 0.9);
                    if (wantPdb) {
                        static std::vector<pattern_db::Table> pdbTables;
                        pdbTables.clear();
                        Timer pdbTimer;
                        for (const u8 mask : pattern_db::chooseClasses(g_rightRoot, cfg.pdbMaxEntries, 3)) {
                            pdbTables.push_back(pattern_db::build(g_rightRoot, mask, boundLimit, threads));
                            const pattern_db::Table& tb = pdbTables.back();
                            tcout << "pdb: table " << tb.name() << ": " << tb.cells << " cells, " << tb.entries / 1e6
                                  << " M entries (" << tb.bits.size() * 8 / 1e6 << " MB), max distance " << tb.maxDist
                                  << ", built in " << tb.buildSeconds << "s\n";
                        }
                        pattern_db::g_tableCount = static_cast<u32>(pdbTables.size());
                        for (u32 i = 0; i < pattern_db::g_tableCount; ++i) {
                            pattern_db::g_tables[i] = &pdbTables[i];
                        }
                        if (pattern_db::g_tableCount != 0) {
                            const bounds::BoundAudit pa = bounds::auditPatternDb<STREAM_DEPTH>(seeds, boundTables, boundLimit, threads, g_rightRoot);
                            tcout << "pdb: audit " << (pa.ok ? "ok" : "FAILED (" + pa.why + ")") << ": " << pa.nodes
                                  << " nodes, " << pa.edges << " edges below far nodes, " << pa.streamChecks
                                  << " wanted leaves identical pruned vs full; whole last level skipped "
                                  << 100.0 * pa.leafSkip << "% -> " << 100.0 * pa.parentSkip << "% with the tables ("
                                  << pdbTimer.getSeconds() << "s total)\n";
                            pdbReady = pa.ok;
                            if (!pa.ok) {
                                pattern_db::g_tableCount = 0;
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
                        if constexpr (STREAM_DEPTH >= 5) {
                            // "off" and "leaf" generate the whole 5-move tree (a timing block would
                            // take many minutes): time only the per-level modes.
                            std::erase_if(boundModes, [](const int m) { return m == 0 || m == 1 || m == 3 || m == 4; });
                            if (boundModes.size() < 2) {
                                boundMode = boundModes.empty() ? 2 : boundModes[0];
                                tcout << "bound: " << BOUND_NAMES[boundMode] << " (5-move stream: per-level pruning only)\n";
                            }
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
        // and how much the bounds prune differs a lot between its regions.
        // Every seed is still visited exactly once; the order does not
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
                const u8 prevMove = seedMoves.empty() ? perm_stream_detail::NO_PREV_MOVE : seedMoves[s];
                if (mode == 0 || mode == 5 || mode == 6) {
                    ProbeSinkT<false> sink(index, variant);
                    if (mode == 5) {
                        PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, bounds::LanePrune>(
                                seedBoard, sink, 4096, prevMove);
                    } else if (mode == 6) {
                        PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, bounds::LaneAndPatternPrune>(
                                seedBoard, sink, 4096, prevMove);
                    } else {
                        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, STREAM_DEPTH>(seedBoard, sink, 4096, prevMove);
                    }
                    return finishSeed(t, sink);
                }
                if (mode == 4) {
                    ProbeSinkT<true, true> sink(index, variant, &boundTables, boundLimit);
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, bounds::LaneAndPatternPrune>(
                            seedBoard, sink, 4096, prevMove);
                    return finishSeed(t, sink);
                }
                ProbeSinkT<true> sink(index, variant, &boundTables, boundLimit);
                if (mode == 3) {
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, bounds::LaneAndPatternPrune>(
                            seedBoard, sink, 4096, prevMove);
                } else if (mode == 2) {
                    // leaves over the limit are dropped in the stream, before their hash; the sink's
                    // own leaf check then only sees survivors ("left states streamed" counts those)
                    PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, STREAM_DEPTH, bounds::LanePruneLeaves>(
                            seedBoard, sink, 4096, prevMove);
                } else {
                    PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, STREAM_DEPTH>(seedBoard, sink, 4096, prevMove);
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
                // Time every eligible mode on real seeds and keep the fastest
                // (src/solver/bounds/calibrate.hpp).
                if (cfg.calibrate && seedPositions - firstSeed >= bounds::calibrationPositionsNeeded(threads, boundModes.size())) {
                    g_progress.detail(passLabel + " timing the lower-bound modes");
                    std::vector<const char*> names;
                    for (const int m : boundModes) {
                        names.push_back(BOUND_NAMES[m]);
                    }
                    const bounds::Calibration c = bounds::calibrateModes(
                            names, threads, firstSeed, firstSeed + (seedPositions - firstSeed) / 4,
                            static_cast<double>(seeds.size()), cfg.boundCalibrateSeconds,
                            [&](const u32 t, const std::size_t pos, const std::size_t m) {
                                processSeed(t, pos, chosenVariant, boundModes[m]);
                            });
                    boundMode = boundModes[c.best];
                    tcout << "bound calibrate: " << c.report << '\n';
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

    inline std::vector<KeyedState> streamAndProbe(const Board& leftRoot,
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
            case 5: return streamAndProbeT<5>(leftRoot, files, meta, counts, cfg, threads, builder, builderReserveBytes);
            default: throw std::runtime_error("--stream-depth must be 3, 4 or 5");
        }
    }

    // Confirms key-sorted candidates against the full depth-6 states on disk.
    // A partition with many candidates is read once sequentially; otherwise
    // each candidate reads just its bucket. Returns the confirmed states, sorted.
    inline std::vector<B1B2> verifyAgainstTable(const fs::path& dir, const FingerprintMeta& meta,
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

    inline std::vector<B1B2> probeFingerprint(const Board& leftRoot,
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

} // namespace ssd
