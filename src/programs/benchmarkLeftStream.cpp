// programs/benchmarkLeftStream.cpp
//
// Measures raw left-side generation speed: exact depth-S seeds, then the
// canonical PermStream (ASCENDING, depth 3) from a sample of them, with a sink
// that only mixes each state's hash into a key (the per-state work every probe
// needs) and folds it into a checksum.
//
// Usage: benchmarkLeftStream <puzzle> [--goal] [--seed-depth 4] [--seeds 200000] [--threads N]

#include "src/chuzzles/include.hpp"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace {

    FORCEINLINE u64 mixKey(u64 x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    struct ChecksumSink {
        u64 count = 0;
        u64 checksum = 0;

        void operator()(JVec<B1B2>&, JVec<u64>& hashes, const u32 n) {
            for (u32 i = 0; i < n; ++i) {
                checksum += mixKey(hashes[i]);
            }
            count += n;
        }
    };

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

} // namespace

int main(const int argc, char** argv) {
    std::string puzzle = "9-3";
    bool goalSide = false;
    u32 seedDepth = 4;
    u64 sampleSeeds = 200'000;
    u32 threads = std::max(1U, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--goal") goalSide = true;
        else if (a == "--seed-depth") seedDepth = static_cast<u32>(std::stoul(argv[++i]));
        else if (a == "--seeds") sampleSeeds = std::stoull(argv[++i]);
        else if (a == "--threads") threads = static_cast<u32>(std::stoul(argv[++i]));
        else puzzle = a;
    }

    const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
    if (pair == nullptr) {
        tcout << "Unknown puzzle: " << puzzle << '\n';
        return 1;
    }
    const Board root = goalSide ? pair->getEndState() : pair->getStartState();

    JVec<B1B2> seeds;
    JVec<u64> seedHashes;
    switch (seedDepth) {
        case 2: buildUniqueNoneDepthFrontierB1B2<2>(root, seeds, seedHashes, false); break;
        case 3: buildUniqueNoneDepthFrontierB1B2<3>(root, seeds, seedHashes, false); break;
        case 4: buildUniqueNoneDepthFrontierB1B2<4>(root, seeds, seedHashes, false); break;
        default: tcout << "seed depth must be 2..4\n"; return 1;
    }
    const u64 stride = std::max<u64>(1, seeds.size() / sampleSeeds);
    std::vector<B1B2> sample;
    for (u64 i = 0; i < seeds.size(); i += stride) {
        sample.push_back(seeds[i]);
    }

    std::vector<u64> counts(threads, 0), sums(threads, 0);
    Timer timer;
    parallelFor(threads, sample.size(), [&](const u32 t, const std::size_t s) {
        ChecksumSink sink;
        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 3>(makeBoardFromState(sample[s]), sink, 4096);
        counts[t] += sink.count;
        sums[t] += sink.checksum;
    });
    const double seconds = timer.getSeconds();

    u64 total = 0, checksum = 0;
    for (u32 t = 0; t < threads; ++t) {
        total += counts[t];
        checksum += sums[t];
    }
    tcout << puzzle << (goalSide ? " goal" : " start") << ", " << (root.getFatBool() ? "fat" : "normal")
          << ": " << sample.size() << " of " << seeds.size() << " depth-" << seedDepth << " seeds, "
          << total << " states in " << seconds << "s on " << threads << " threads = "
          << static_cast<double>(total) / seconds / 1e6 << " M states/s ("
          << static_cast<double>(total) / seconds / 1e6 / threads << " M/s/thread), checksum "
          << checksum << '\n';
    return 0;
}
