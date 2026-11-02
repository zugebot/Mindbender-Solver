#pragma once
// src/solver/ssd/gpu_stream.hpp
//
// --gpu: work on the GPU (src/gpu, mbgpu.dll). The DLL is built with nvcc + MSVC,
// so it is loaded at run time and only plain C types cross (src/gpu/mbgpu.h).
// - The probe: per pass the GPU gets the pass's Bloom blocks, streams every seed
//   with the canonical rules and the rotation lane bound ("both"), and hands back
//   the leaves that pass the Bloom filter; the CPU finishes those with the same
//   index code as the CPU probe (FingerprintRange::probeGroup), so the candidates
//   are exactly the CPU's.
// - The bound audit (gpuAuditBound), in place of bounds::auditLaneBound.
// - The no-table depth-6 index: child counts and sorted unique child keys per
//   partition range (no_table_d6.hpp); the CPU writes the index from them.

#include "src/solver/ssd/fingerprint_index.hpp"
#include "src/gpu/mbgpu.h"

namespace ssd {

    static_assert(sizeof(B1B2) == 16, "mbgpu reads seeds as 2 x u64 per B1B2");

    struct GpuModule {
        HMODULE dll = nullptr;
        decltype(&mbgpu_open) open = nullptr;
        decltype(&mbgpu_set_target) setTarget = nullptr;
        decltype(&mbgpu_set_filter) setFilter = nullptr;
        decltype(&mbgpu_stream) stream = nullptr;
        decltype(&mbgpu_stream_checksums) streamChecksums = nullptr;
        decltype(&mbgpu_audit_stream) auditStream = nullptr;
        decltype(&mbgpu_bounds) bounds = nullptr;
        decltype(&mbgpu_child_counts) childCounts = nullptr;
        decltype(&mbgpu_child_keys) childKeys = nullptr;
        decltype(&mbgpu_child_key_capacity) childKeyCapacity = nullptr;
        decltype(&mbgpu_build_index) buildIndex = nullptr;
        decltype(&mbgpu_last_error) lastError = nullptr;
        decltype(&mbgpu_close) close = nullptr;
        std::string path;

        MUND bool ready() const { return dll != nullptr; }

        MUND std::string error() const { return lastError != nullptr ? lastError() : "mbgpu.dll not loaded"; }

        // mbgpu.dll from `explicitPath`, else next to the exe, else <exe dir>/../build_gpu.
        bool load(const std::string& explicitPath, std::string& why) {
            std::vector<std::string> tries;
            if (!explicitPath.empty()) {
                tries.push_back(explicitPath);
            } else {
                char exe[MAX_PATH] = {};
                GetModuleFileNameA(nullptr, exe, MAX_PATH);
                const fs::path dir = fs::path(exe).parent_path();
                tries.push_back((dir / "mbgpu.dll").string());
                tries.push_back((dir.parent_path() / "build_gpu" / "mbgpu.dll").string());
            }
            for (const std::string& p : tries) {
                dll = LoadLibraryA(p.c_str());
                if (dll != nullptr) {
                    path = p;
                    break;
                }
            }
            if (dll == nullptr) {
                why = "mbgpu.dll not found (tried " + tries.front() + (tries.size() > 1 ? ", " + tries.back() : "") + ")";
                return false;
            }
            auto get = [&](auto& fn, const char* name) {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
                        reinterpret_cast<void*>(GetProcAddress(dll, name)));
                return fn != nullptr;
            };
            if (!(get(open, "mbgpu_open") && get(setTarget, "mbgpu_set_target") && get(setFilter, "mbgpu_set_filter") &&
                  get(stream, "mbgpu_stream") && get(streamChecksums, "mbgpu_stream_checksums") &&
                  get(auditStream, "mbgpu_audit_stream") && get(bounds, "mbgpu_bounds") &&
                  get(childCounts, "mbgpu_child_counts") && get(childKeys, "mbgpu_child_keys") &&
                  get(childKeyCapacity, "mbgpu_child_key_capacity") && get(buildIndex, "mbgpu_build_index") &&
                  get(lastError, "mbgpu_last_error") &&
                  get(close, "mbgpu_close"))) {
                why = path + " lacks an mbgpu_* export";
                FreeLibrary(dll);
                dll = nullptr;
                return false;
            }
            return true;
        }
    };

    // One module per run, opened by the first part that needs it (--gpu NAME).
    inline GpuModule g_gpu;
    inline bool g_gpuOpen = false;

    inline GpuModule& ensureGpu(const Config& cfg) {
        if (g_gpuOpen) {
            return g_gpu;
        }
        std::string why;
        if (!g_gpu.ready() && !g_gpu.load(cfg.gpuDll, why)) {
            throw std::runtime_error("--gpu: " + why);
        }
        char info[512] = {};
        if (g_gpu.open(cfg.gpuDevice.c_str(), static_cast<uint64_t>(cfg.gpuVramGb * 1e9), info, sizeof(info)) != 0) {
            throw std::runtime_error("--gpu: " + g_gpu.error());
        }
        g_gpuOpen = true;
        tcout << "gpu: " << info << "; " << g_gpu.path << "\n";
        return g_gpu;
    }

    inline void closeGpu() {
        if (g_gpuOpen) {
            g_gpu.close();
            g_gpuOpen = false;
        }
    }

    // The GPU's bound, audited on this run's seeds before the run trusts it (the GPU computes
    // h from its definition, not from the CPU's lane tables, so it is checked on its own):
    // (1) it equals lane_bound::referenceBound on the sampled seeds and their one- and
    // two-move children; (2) on every edge of the full (unpruned) stream of the sampled
    // seeds h changes by at most 1; (3) per sampled seed the pruned stream keeps exactly the
    // leaves with h <= limit of the full stream (count, key sum and xor).
    struct GpuAudit {
        bool ok = true;
        std::string why;
        mbgpu_audit_stats stats{};
        u64 seeds = 0, refStates = 0;
        double seconds = 0.0;
    };

    inline GpuAudit gpuAuditBound(GpuModule& gpu, const JVec<B1B2>& seeds, const std::vector<u8>& moves,
                                  const i32 depth, const int limit, const B1B2& target) {
        GpuAudit a;
        Timer timer;
        if (gpu.setTarget(target.b1, target.b2, limit) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        const std::size_t want = std::min<std::size_t>(seeds.size(), depth >= 5 ? 256 : 2048);
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, want));
        std::vector<u64> states;
        std::vector<u8> prev;
        for (std::size_t i = 0; i < seeds.size() && prev.size() < want; i += step) {
            states.push_back(seeds[i].b1);
            states.push_back(seeds[i].b2);
            prev.push_back(moves.empty() ? perm_stream_detail::NO_PREV_MOVE : moves[i]);
        }
        const std::size_t n = prev.size();
        a.seeds = n;
        std::vector<u64> fc(n), fs(n), fx(n), pc(n), ps(n), px(n);
        if (gpu.auditStream(states.data(), prev.data(), n, depth, fc.data(), fs.data(), fx.data(), &a.stats) != 0 ||
            gpu.streamChecksums(states.data(), prev.data(), n, depth, pc.data(), ps.data(), px.data(), nullptr) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        if (a.stats.violations != 0) {
            a.ok = false;
            a.why = std::to_string(a.stats.violations) + " edges changed h by more than 1";
        }
        for (std::size_t s = 0; s < n && a.ok; ++s) {
            if (fc[s] != pc[s] || fs[s] != ps[s] || fx[s] != px[s]) {
                a.ok = false;
                a.why = "pruned stream lost or added a wanted leaf";
            }
        }
        // (1) against the reference definition
        struct Collect {
            std::vector<B1B2>* out;
            void operator()(JVec<B1B2>& chunk, JVec<u64>&, const u32 count) {
                out->insert(out->end(), chunk.begin(), chunk.begin() + count);
            }
        };
        std::vector<B1B2> ref;
        for (std::size_t s = 0; s < n; ++s) {
            ref.emplace_back(states[2 * s], states[2 * s + 1]);
        }
        Collect collect{&ref};
        for (std::size_t s = 0; s < std::min<std::size_t>(n, 64); ++s) {
            const Board b = makeBoardFromState(B1B2(states[2 * s], states[2 * s + 1]));
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 1>(b, collect, 4096);
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 2>(b, collect, 4096);
        }
        std::vector<int32_t> h(ref.size());
        if (gpu.bounds(reinterpret_cast<const uint64_t*>(ref.data()), ref.size(), h.data()) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        for (std::size_t i = 0; i < ref.size() && a.ok; ++i) {
            if (h[i] != lane_bound::referenceBound(ref[i], target, lane_bound::Kind::Rotation)) {
                a.ok = false;
                a.why = "GPU bound != reference definition";
            }
        }
        a.refStates = ref.size();
        a.seconds = timer.getSeconds();
        return a;
    }

    // The leaves the GPU passes back (Bloom survivors; rare, so plain vectors).
    struct GpuLeaves {
        std::vector<B1B2> states;
        std::vector<u64> keys;

        static void sink(void* ctx, const uint64_t* b1, const uint64_t* b2, const uint64_t* keys, const uint32_t*,
                         const uint64_t count) {
            auto* self = static_cast<GpuLeaves*>(ctx);
            for (uint64_t i = 0; i < count; ++i) {
                self->states.emplace_back(b1[i], b2[i]);
                self->keys.push_back(keys[i]);
            }
        }
    };

    struct GpuPassResult {
        mbgpu_stats stats{};
        u64 bloomPasses = 0;
    };

    // One probe pass on the GPU: Bloom blocks of [pBegin, pEnd) to the device, every
    // seed streamed, survivors finished on the CPU into `candidates`.
    inline GpuPassResult gpuProbePass(GpuModule& gpu, const FingerprintRange& index, const JVec<B1B2>& seeds,
                                      const std::vector<u8>& seedMoves, const i32 streamDepth, const u32 threads,
                                      std::vector<KeyedState>& candidates) {
        GpuPassResult r;
        if (gpu.setFilter(index.bloom.data(), index.meta.bloomBlocksPerPartition, index.pBegin, index.pEnd) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        GpuLeaves leaves;
        const auto* states = reinterpret_cast<const uint64_t*>(seeds.data());
        if (gpu.stream(states, seedMoves.empty() ? nullptr : seedMoves.data(), seeds.size(), streamDepth,
                       &GpuLeaves::sink, &leaves, &r.stats) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        gpu.setFilter(nullptr, 0, 0, PARTITION_COUNT);   // the Bloom blocks' VRAM back before the next pass
        // Buckets and fingerprints on the CPU (the Bloom test is repeated there; it passes).
        const std::size_t n = leaves.states.size();
        constexpr std::size_t GROUP = PROBE_GROUP;
        std::vector<std::vector<KeyedState>> found(threads);
        std::vector<u64> passes(threads, 0);
        parallelFor(threads, (n + GROUP - 1) / GROUP, [&](const u32 t, const std::size_t g) {
            const std::size_t lo = g * GROUP, cnt = std::min(GROUP, n - lo);
            passes[t] += index.probeGroup(&leaves.states[lo], &leaves.keys[lo], static_cast<u32>(cnt), found[t]);
        });
        for (u32 t = 0; t < threads; ++t) {
            r.bloomPasses += passes[t];
            candidates.insert(candidates.end(), found[t].begin(), found[t].end());
        }
        return r;
    }

} // namespace ssd
