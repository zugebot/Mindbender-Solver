// programs/experimentDiskDepth6.cpp
//
// Experiment: <3, 3, 6> meet-in-the-middle with the depth-6 right frontier on disk.
//
//   Build:  exact depth-5 frontier in RAM -> expand every state by one move ->
//           route each child to one of 256 partition files by the top 8 bits of
//           a mixed state key -> sort + de-dupe each partition independently.
//   Probe:  load a contiguous range of sorted partitions that fits the RAM
//           budget, index it with RightFrontierIndexB1B2, then regenerate the
//           whole left side (exact depth-3 seeds, canonical depth-3 stream) and
//           probe only states whose key falls in that range. Repeat per range.
//
// All disk I/O is sequential. Found middle states are optionally validated
// against a complete solution file: the state after the first 6 moves of every
// optimal 12-move solution must be exactly the set of middles found.
//
// Usage:
//   experimentDiskDepth6 <puzzle> [--out X:/mindbender_d6] [--batch 160000000]
//                        [--threads N] [--reverse] [--rebuild] [--solutions file]

#include "src/chuzzles/include.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

    constexpr u32 PARTITION_BITS = 8;
    constexpr u32 PARTITION_COUNT = 1U << PARTITION_BITS;
    constexpr u32 PARTITION_SHIFT = 64U - PARTITION_BITS;
    constexpr std::size_t WRITE_BUFFER_STATES = 8192;
    constexpr std::size_t PROBE_BUFFER_STATES = 1U << 16;

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
        u64 batchStates = 160'000'000ULL;
        u32 threads = 0;
        bool reverse = false;
        bool rebuild = false;
        std::string solutionsFile;
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
    // Probe: <3, 3> left side against batches of partitions
    // ------------------------------------------------------------------
    std::vector<B1B2> probeAllBatches(const Board& leftRoot,
                                      const fs::path& dir,
                                      const std::vector<u64>& counts,
                                      const Config& cfg,
                                      const u32 threads) {
        Timer seedTimer;
        JVec<B1B2> seeds;
        JVec<u64> seedHashes;
        buildUniqueNoneDepthFrontierB1B2<3>(leftRoot, seeds, seedHashes, false);
        tcout << "left seeds (exact depth 3): " << seeds.size() << " in " << seedTimer.getSeconds() << "s\n";

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
            JVec<B1B2> states;
            JVec<u64> keys;
            states.resize(batchSize);
            keys.resize(batchSize);
            u64 offset = 0;
            for (u32 p = pBegin; p < pEnd; ++p) {
                std::FILE* f = std::fopen(sortedPath(dir, p).string().c_str(), "rb");
                if (f == nullptr || std::fread(states.data() + offset, sizeof(B1B2), counts[p], f) != counts[p]) {
                    throw std::runtime_error("failed to read sorted partition");
                }
                std::fclose(f);
                offset += counts[p];
            }
            const double readSeconds = loadTimer.getSeconds();
            parallelFor(threads, (batchSize + 65535) / 65536, [&](u32, const std::size_t c) {
                const u64 end = std::min<u64>(batchSize, (c + 1) * 65536);
                for (u64 i = c * 65536; i < end; ++i) {
                    keys[i] = keyOf(states[i]);
                }
            });
            loadSeconds += loadTimer.getSeconds();

            Timer indexTimer;
            RightFrontierIndexB1B2 index;
            index.buildFromUniqueStates<false>(std::move(states), std::move(keys), leftRoot);
            indexSeconds += indexTimer.getSeconds();

            Timer probeTimer;
            std::mutex middleLock;
            std::vector<u64> streamed(threads, 0), inRange(threads, 0);
            parallelFor(threads, seeds.size(), [&](const u32 t, const std::size_t s) {
                struct Sink {
                    const RightFrontierIndexB1B2& index;
                    const u32 pBegin, pEnd;
                    JVec<B1B2> probeStates, matches;
                    JVec<u64> probeKeys, matchKeys;
                    std::vector<B1B2> found;
                    u64 streamed = 0, inRange = 0;

                    void probe() {
                        if (probeStates.size() == 0) {
                            return;
                        }
                        index.collectMatches<6, false>(probeStates, probeKeys, matches, matchKeys, nullptr);
                        found.insert(found.end(), matches.begin(), matches.end());
                        probeStates.resize(0);
                        probeKeys.resize(0);
                    }

                    void operator()(JVec<B1B2>& chunkStates, JVec<u64>& chunkHashes, const u32 count) {
                        streamed += count;
                        for (u32 i = 0; i < count; ++i) {
                            const u64 key = mixKey(chunkHashes[i]);
                            const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                            if (p < pBegin || p >= pEnd) {
                                continue;
                            }
                            const std::size_t n = probeStates.size();
                            probeStates.resize(n + 1);
                            probeKeys.resize(n + 1);
                            probeStates[n] = chunkStates[i];
                            probeKeys[n] = key;
                            ++inRange;
                            if (n + 1 == PROBE_BUFFER_STATES) {
                                probe();
                            }
                        }
                    }
                } sink{index, pBegin, pEnd, {}, {}, {}, {}, {}, 0, 0};
                sink.probeStates.reserve(PROBE_BUFFER_STATES);
                sink.probeKeys.reserve(PROBE_BUFFER_STATES);

                const Board seedBoard = makeBoardFromState(seeds[s]);
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 3>(seedBoard, sink, 4096);
                sink.probe();

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
                  << ", index " << indexTimer.getSeconds() << "s"
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

    // Middles expected from a complete 12-move solution file: the state after
    // the first 6 moves of every solution (from the true start board).
    std::vector<B1B2> expectedMiddles(const BoardPair& pair, const std::string& file) {
        std::vector<B1B2> out;
        std::ifstream in(file);
        std::string line;
        const bool isFat = pair.getStartState().getFatBool();
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }
            const std::vector<u8> moves = isFat ? Memory::parseFatMoveString(line)
                                                : Memory::parseNormMoveString(line);
            if (moves.size() != 12) {
                throw std::runtime_error("expected 12-move solutions in " + file);
            }
            Board board = pair.getStartState();
            for (std::size_t i = 0; i < 6; ++i) {
                allActStructList[moves[i]].action(board);
            }
            out.push_back(board.asB1B2());
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
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
            else if (a == "--reverse") cfg.reverse = true;
            else if (a == "--rebuild") cfg.rebuild = true;
            else if (a == "--solutions") cfg.solutionsFile = next();
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

    tcout << "puzzle " << cfg.puzzle << " <3,3,6>, " << (cfg.reverse ? "reverse" : "forward")
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

    const Timer probeTimer;
    const std::vector<B1B2> middles = probeAllBatches(leftRoot, dir, counts, cfg, cfg.threads);
    const double probeSeconds = probeTimer.getSeconds();

    tcout << "unique middle states at depth 6: " << middles.size() << '\n';
    tcout << "PROBE phase: " << probeSeconds << "s, TOTAL: " << total.getSeconds()
          << "s (build " << buildSeconds << "s)\n";

    if (!cfg.solutionsFile.empty()) {
        const std::vector<B1B2> expected = expectedMiddles(*pair, cfg.solutionsFile);
        std::vector<B1B2> missing, extra;
        std::set_difference(expected.begin(), expected.end(), middles.begin(), middles.end(),
                            std::back_inserter(missing));
        std::set_difference(middles.begin(), middles.end(), expected.begin(), expected.end(),
                            std::back_inserter(extra));
        tcout << "validation vs " << cfg.solutionsFile << ": expected " << expected.size()
              << ", missing " << missing.size() << ", extra " << extra.size()
              << (missing.empty() && extra.empty() ? "  -> EXACT MATCH\n" : "  -> MISMATCH\n");
        return missing.empty() && extra.empty() ? 0 : 2;
    }
    return 0;
}
