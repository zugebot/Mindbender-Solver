#pragma once
// src/solver/ssd/gpu_stream.hpp
//
// --gpu: the left stream of a probe pass on the GPU (src/gpu, mbgpu.dll). The
// DLL is built with nvcc + MSVC, so it is loaded at run time and only plain C
// types cross (src/gpu/mbgpu.h). Per pass the GPU gets the pass's Bloom blocks,
// streams every seed with the canonical rules and the rotation lane bound ("both"),
// and hands back the leaves that pass the Bloom filter; the CPU finishes those
// with the same index code as the CPU probe (FingerprintRange::probeGroup), so
// the candidates are exactly the CPU's.

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
                  get(stream, "mbgpu_stream") && get(lastError, "mbgpu_last_error") && get(close, "mbgpu_close"))) {
                why = path + " lacks an mbgpu_* export";
                FreeLibrary(dll);
                dll = nullptr;
                return false;
            }
            return true;
        }
    };

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
