// src/gpu/test_gpu_stream.cu
//
// Checks the GPU left stream against the CPU's (vectors from exportGpuVectors):
// for every case and every seed, the leaf count and the sum and xor of the leaf
// keys must be identical. Then a filter check: with an empty Bloom filter nothing
// may pass, with a full one every in-range leaf must. Prints the GPU time next to
// the CPU's (single thread) for scale.
//
// Usage: test_gpu_stream <vectors.bin> [--device 5060] [--vram-gb 12.5]
//        test_gpu_stream <bench.bin> --bench   (a file from exportGpuVectors --bench: times
//        the full stream of every seed, no CPU comparison)

#include "src/gpu/mbgpu.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

    struct Case {
        char name[16];
        uint64_t goalB1, goalB2;
        int limit, depth, seedDepth, lastMoves;
        std::vector<uint64_t> b1, b2, cnt, sum, xr, states;   // states: b1, b2 interleaved (mbgpu.h)
        std::vector<uint8_t> prev;
        double cpuSeconds;
    };

    template<typename T>
    bool get(std::FILE* f, T& v) {
        return std::fread(&v, sizeof(T), 1, f) == 1;
    }

    template<typename T>
    bool getN(std::FILE* f, std::vector<T>& v, const uint64_t n) {
        v.resize(n);
        return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
    }

    bool readCases(const char* path, std::vector<Case>& cases) {
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) {
            return false;
        }
        char magic[8];
        uint32_t count = 0;
        bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "MBGV0001", 8) == 0 && get(f, count);
        for (uint32_t i = 0; ok && i < count; ++i) {
            Case c{};
            uint64_t n = 0;
            ok = std::fread(c.name, 1, 16, f) == 16 && get(f, c.goalB1) && get(f, c.goalB2) && get(f, c.limit) &&
                 get(f, c.depth) && get(f, c.seedDepth) && get(f, c.lastMoves) && get(f, n) && getN(f, c.b1, n) &&
                 getN(f, c.b2, n) && getN(f, c.prev, n) && getN(f, c.cnt, n) && getN(f, c.sum, n) &&
                 getN(f, c.xr, n) && get(f, c.cpuSeconds);
            c.states.resize(2 * n);
            for (uint64_t s = 0; s < n && ok; ++s) {
                c.states[2 * s] = c.b1[s];
                c.states[2 * s + 1] = c.b2[s];
            }
            cases.push_back(std::move(c));
        }
        std::fclose(f);
        return ok;
    }

    struct CountSink {
        uint64_t n = 0;
        static void fn(void* ctx, const uint64_t*, const uint64_t*, const uint64_t*, const uint32_t*, const uint64_t c) {
            static_cast<CountSink*>(ctx)->n += c;
        }
    };

}  // namespace

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_gpu_stream <vectors.bin> [--device NAME] [--vram-gb G]\n");
        return 2;
    }
    std::string device = "5060";
    double vramGb = 12.5;
    bool bench = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        if (i + 1 < argc && std::string(argv[i]) == "--device") device = argv[i + 1];
        if (i + 1 < argc && std::string(argv[i]) == "--vram-gb") vramGb = std::stod(argv[i + 1]);
    }
    std::vector<Case> cases;
    if (!readCases(argv[1], cases)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    char info[512];
    if (mbgpu_open(device.c_str(), static_cast<uint64_t>(vramGb * 1e9), info, sizeof(info)) != 0) {
        std::fprintf(stderr, "mbgpu_open: %s\n", mbgpu_last_error());
        return 1;
    }
    std::printf("%s\n", info);

    if (bench) {
        for (const Case& c : cases) {
            CountSink sink;
            mbgpu_stats st{};
            if (mbgpu_set_target(c.goalB1, c.goalB2, c.limit) != 0 || mbgpu_set_filter(nullptr, 0, 0, 256) != 0 ||
                mbgpu_stream(c.states.data(), c.prev.data(), c.b1.size(), c.depth, CountSink::fn, &sink, &st) != 0) {
                std::printf("ERROR: %s\n", mbgpu_last_error());
                return 1;
            }
            uint64_t nodes = 0;
            std::printf("%s <%d,%d,%d>: %llu seeds, %.2f s, %llu launches\n", c.name, c.seedDepth, c.depth, c.limit,
                        static_cast<unsigned long long>(c.b1.size()), st.seconds,
                        static_cast<unsigned long long>(st.launches));
            for (int d = 0; d <= c.depth; ++d) {
                const uint64_t v = d < c.depth ? st.nodes[d] : st.leaves;
                nodes += d > 0 ? v : 0;
                std::printf("  level %d: %llu %s\n", d, static_cast<unsigned long long>(v),
                            d == 0 ? "seeds" : (d < c.depth ? "nodes kept" : "leaves kept"));
            }
            std::printf("  %.2f M seeds/s, %.1f M kept nodes/s; leaves handed to the sink: %llu\n",
                        c.b1.size() / st.seconds / 1e6, nodes / st.seconds / 1e6,
                        static_cast<unsigned long long>(sink.n));
        }
        mbgpu_close();
        return 0;
    }

    int failures = 0;
    for (const Case& c : cases) {
        const uint64_t n = c.b1.size();
        std::vector<uint64_t> cnt(n), sum(n), xr(n);
        mbgpu_stats st{};
        if (mbgpu_set_target(c.goalB1, c.goalB2, c.limit) != 0 ||
            mbgpu_stream_checksums(c.states.data(), c.prev.data(), n, c.depth, cnt.data(), sum.data(), xr.data(),
                                   &st) != 0) {
            std::printf("%-5s ERROR: %s\n", c.name, mbgpu_last_error());
            ++failures;
            continue;
        }
        uint64_t bad = 0, leaves = 0, firstBad = n;
        for (uint64_t s = 0; s < n; ++s) {
            leaves += c.cnt[s];
            if (cnt[s] != c.cnt[s] || sum[s] != c.sum[s] || xr[s] != c.xr[s]) {
                if (bad++ == 0) firstBad = s;
            }
        }
        std::printf("%-5s seeds d%d %-7s stream %d limit %2d: %6llu seeds %12llu leaves  %s",
                    c.name, c.seedDepth, c.lastMoves ? "lastmv" : "noprev", c.depth, c.limit,
                    static_cast<unsigned long long>(n), static_cast<unsigned long long>(leaves),
                    bad == 0 ? "MATCH" : "DIFFER");
        std::printf("  (GPU %.3f s, %llu launches; CPU 1 thread %.2f s)\n", st.seconds,
                    static_cast<unsigned long long>(st.launches), c.cpuSeconds);
        if (bad != 0) {
            ++failures;
            const uint64_t s = firstBad;
            std::printf("      %llu seeds differ; first: seed %llu prev %u: CPU count %llu, GPU %llu\n",
                        static_cast<unsigned long long>(bad), static_cast<unsigned long long>(s), c.prev[s],
                        static_cast<unsigned long long>(c.cnt[s]), static_cast<unsigned long long>(cnt[s]));
        }
    }

    // Filter: an all-zero Bloom filter passes nothing, an all-ones one passes every leaf.
    {
        const Case& c = cases.back();
        const uint64_t blocks = 64;
        std::vector<uint32_t> bloom(256 * blocks * 8, 0);
        CountSink none, all;
        mbgpu_stats s0{}, s1{};
        bool ok = mbgpu_set_target(c.goalB1, c.goalB2, c.limit) == 0 &&
                  mbgpu_set_filter(bloom.data(), blocks, 0, 256) == 0 &&
                  mbgpu_stream(c.states.data(), c.prev.data(), c.b1.size(), c.depth, CountSink::fn, &none, &s0) == 0;
        std::fill(bloom.begin(), bloom.end(), 0xFFFFFFFFu);
        ok = ok && mbgpu_set_filter(bloom.data(), blocks, 0, 256) == 0 &&
             mbgpu_stream(c.states.data(), c.prev.data(), c.b1.size(), c.depth, CountSink::fn, &all, &s1) == 0;
        uint64_t leaves = 0;
        for (const uint64_t v : c.cnt) leaves += v;
        const bool pass = ok && none.n == 0 && all.n == leaves && s1.leaves == leaves;
        std::printf("filter: empty passes %llu, full passes %llu of %llu leaves  %s\n",
                    static_cast<unsigned long long>(none.n), static_cast<unsigned long long>(all.n),
                    static_cast<unsigned long long>(leaves), pass ? "OK" : (ok ? "WRONG" : mbgpu_last_error()));
        failures += pass ? 0 : 1;
        mbgpu_set_filter(nullptr, 0, 0, 256);
    }

    mbgpu_close();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
