#pragma once
// src/solver/ssd/gpu_stream.hpp
//
// --gpu: work on the GPU. Two modules with the same C interface (src/gpu/mbgpu.h), loaded at
// run time so only plain C types cross: mbgpu.dll (CUDA, src/gpu, nvcc + MSVC) and mbvk.dll
// (Vulkan, any vendor, src/gpu/vulkan). --gpu [cuda:|vulkan:]NAME: without a prefix CUDA is
// tried first, then Vulkan, so NVIDIA cards keep the CUDA kernels. NAME "auto" lets the module
// choose (CUDA: its first device, the fastest; Vulkan: every discrete GPU that drives no
// display, see mbvk.cpp). --gpu-list prints the Vulkan devices without opening any.
// - The probe: per pass the GPU gets the pass's Bloom blocks, streams every seed
//   with the canonical rules and the rotation lane bound ("both"), and hands back
//   the leaves that pass the Bloom filter; CPU threads finish those with the same
//   index code as the CPU probe (FingerprintRange::probeGroup) while the GPU
//   streams on (ProbeQueue), so the candidates are exactly the CPU's.
// - The bound audit (gpuAuditBound), in place of bounds::auditLaneBound.
// - The no-table depth-6 index (no_table_d6.hpp): child counts per partition, then
//   per group of partitions the index itself (fingerprints, bucket offsets, Bloom
//   blocks) written by the GPU; and the candidate check (mbgpu_match_children).

#include "src/solver/ssd/fingerprint_index.hpp"
#include "src/gpu/mbgpu.h"

#include <condition_variable>
#include <functional>
#include <deque>
#include <mutex>
#include <thread>

namespace ssd {

    static_assert(sizeof(B1B2) == 16, "mbgpu reads seeds as 2 x u64 per B1B2");

    struct GpuModule {
        void* dll = nullptr;
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
        decltype(&mbgpu_match_children) matchChildren = nullptr;
        decltype(&mbgpu_vram_left) vramLeft = nullptr;
        decltype(&mbgpu_last_error) lastError = nullptr;
        decltype(&mbgpu_close) close = nullptr;
        std::string path;

        MUND bool ready() const { return dll != nullptr; }

        MUND std::string error() const { return lastError != nullptr ? lastError() : "mbgpu.dll not loaded"; }

        bool vulkan = false;   // mbvk.dll (it exports mbvk_list_devices)

        // `file` as a path when it has a directory, else <exe dir>/file, then <exe dir>/../build_gpu/file.
        bool load(const std::string& file, std::string& why) {
            std::vector<std::string> tries;
            if (file.find_first_of("/\\") != std::string::npos) {
                tries.push_back(file);
            } else {
                const fs::path dir = fs::path(platform::exePath()).parent_path();
                tries.push_back((dir / file).string());
                tries.push_back((dir.parent_path() / "build_gpu" / file).string());
            }
            for (const std::string& p : tries) {
                dll = platform::loadLibrary(p);
                if (dll != nullptr) {
                    path = p;
                    break;
                }
            }
            if (dll == nullptr) {
                why = file + " not found (tried " + tries.front() + (tries.size() > 1 ? ", " + tries.back() : "") + ")";
                return false;
            }
            vulkan = platform::librarySymbol(dll, "mbvk_list_devices") != nullptr;
            auto get = [&](auto& fn, const char* name) {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(platform::librarySymbol(dll, name));
                return fn != nullptr;
            };
            if (!(get(open, "mbgpu_open") && get(setTarget, "mbgpu_set_target") && get(setFilter, "mbgpu_set_filter") &&
                  get(stream, "mbgpu_stream") && get(streamChecksums, "mbgpu_stream_checksums") &&
                  get(auditStream, "mbgpu_audit_stream") && get(bounds, "mbgpu_bounds") &&
                  get(childCounts, "mbgpu_child_counts") && get(childKeys, "mbgpu_child_keys") &&
                  get(childKeyCapacity, "mbgpu_child_key_capacity") && get(buildIndex, "mbgpu_build_index") &&
                  get(matchChildren, "mbgpu_match_children") && get(vramLeft, "mbgpu_vram_left") &&
                  get(lastError, "mbgpu_last_error") &&
                  get(close, "mbgpu_close"))) {
                why = path + " lacks an mbgpu_* export";
                unload();
                return false;
            }
            return true;
        }

        void unload() {
            if (dll != nullptr) {
                platform::freeLibrary(dll);
            }
            *this = GpuModule{};
        }
    };

    // --gpu [cuda:|vulkan:]NAME -> (modules to try in order, NAME)
    inline std::pair<std::vector<std::string>, std::string> gpuModules(const Config& cfg) {
        std::string name = cfg.gpuDevice, backend;
        const std::size_t colon = name.find(':');
        if (colon != std::string::npos) {
            const std::string prefix = name.substr(0, colon);
            if (prefix == "cuda" || prefix == "vulkan" || prefix == "vk") {
                backend = prefix == "cuda" ? "cuda" : "vulkan";
                name = name.substr(colon + 1);
            }
        }
        std::vector<std::string> files;
        if (!cfg.gpuDll.empty()) {
            files.push_back(cfg.gpuDll);
        } else {
            if (backend != "vulkan") files.emplace_back(platform::libraryFile("mbgpu"));
            if (backend != "cuda") files.emplace_back(platform::libraryFile("mbvk"));
        }
        return {files, name};
    }

    // One module per run, opened by the first part that needs it (--gpu NAME).
    inline GpuModule g_gpu;
    inline bool g_gpuOpen = false;

    inline GpuModule& ensureGpu(const Config& cfg) {
        if (g_gpuOpen) {
            return g_gpu;
        }
        const auto [files, name] = gpuModules(cfg);
        std::string whys;
        for (const std::string& file : files) {
            GpuModule m;
            std::string why;
            if (!m.load(file, why)) {
                whys += "; " + why;
                continue;
            }
            // CUDA "auto": its first device (CUDA orders them fastest first); Vulkan picks itself
            const std::string open = !m.vulkan && name == "auto" ? std::string() : name;
            char info[2048] = {};
            if (m.open(open.c_str(), static_cast<uint64_t>(cfg.gpuVramGb * 1e9), info, sizeof(info)) != 0) {
                const std::string err = m.error();
                whys += "; " + m.path + ": " + err;
                m.unload();
                // the next module only when this one has no such device: a CUDA device that
                // fails to open is an error, not a reason to run the same card on Vulkan
                if (err.rfind("no CUDA device", 0) == 0 || err.rfind("no Vulkan device", 0) == 0 ||
                    err.rfind("no usable Vulkan device", 0) == 0 || err.rfind("vulkan-1.dll", 0) == 0 || err.rfind("libvulkan", 0) == 0) {
                    continue;
                }
                throw std::runtime_error("--gpu " + cfg.gpuDevice + ": " + whys.substr(2));
            }
            g_gpu = m;
            g_gpuOpen = true;
            tcout << "gpu: " << info << "; " << g_gpu.path << "\n";
            return g_gpu;
        }
        throw std::runtime_error("--gpu " + cfg.gpuDevice + ": no GPU module could open it (" +
                                 (whys.empty() ? std::string("none tried") : whys.substr(2)) + ")");
    }

    // --gpu-list: the Vulkan devices (mbvk.dll asks the driver; nothing is opened) and whether
    // the CUDA module is there (opening it would allocate VRAM, so it is not opened).
    inline void listGpus() {
        GpuModule cuda, vk;
        std::string why;
        if (cuda.load(platform::libraryFile("mbgpu"), why) && !cuda.vulkan) {
            tcout << "CUDA module: " << cuda.path << " (NVIDIA GPUs; --gpu auto tries it first)\n";
        } else {
            tcout << "CUDA module: none (" << why << ")\n";
        }
        cuda.unload();
        why.clear();
        if (!vk.load(platform::libraryFile("mbvk"), why) || !vk.vulkan) {
            tcout << "Vulkan module: none (" << why << ")\n";
            vk.unload();
            return;
        }
        using ListFn = int (*)(char*, int);
        const auto list = reinterpret_cast<ListFn>(platform::librarySymbol(vk.dll, "mbvk_list_devices"));
        std::vector<char> text(1 << 16);
        tcout << "Vulkan module: " << vk.path << "\n";
        if (list(text.data(), static_cast<int>(text.size())) != 0) {
            tcout << "  " << vk.error() << "\n";
        } else {
            tcout << text.data();
        }
        vk.unload();
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
        // seeds streamed in full (unpruned): ~115 M nodes each at depth 5, ~30x that at depth 6
        const std::size_t want = std::min<std::size_t>(seeds.size(), depth >= 6 ? 16 : depth >= 5 ? 256 : 2048);
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

    // Leaves that passed the GPU's Bloom test, waiting for the CPU probe. The sink (on the
    // thread driving the GPU) copies each batch in, split into pieces, and waits while more
    // than `capLeaves` are queued; probe threads take pieces while the GPU streams on.
    class ProbeQueue {
    public:
        struct Piece {
            std::vector<B1B2> states;
            std::vector<u64> keys;
        };

        explicit ProbeQueue(const u64 capLeaves) : cap_(capLeaves) {}

        static void sink(void* ctx, const uint64_t* b1, const uint64_t* b2, const uint64_t* keys, const uint32_t*,
                         const uint64_t count) {
            auto* self = static_cast<ProbeQueue*>(ctx);
            constexpr uint64_t PIECE = 1ULL << 16;
            for (uint64_t lo = 0; lo < count; lo += PIECE) {
                const uint64_t n = std::min(PIECE, count - lo);
                Piece p;
                p.states.resize(n);
                p.keys.assign(keys + lo, keys + lo + n);
                for (uint64_t i = 0; i < n; ++i) {
                    p.states[i] = B1B2(b1[lo + i], b2[lo + i]);
                }
                std::unique_lock<std::mutex> lock(self->m_);
                self->space_.wait(lock, [&] { return self->queued_ < self->cap_; });
                self->queued_ += n;
                self->q_.push_back(std::move(p));
                self->ready_.notify_one();
            }
        }

        // false when the stream has finished and nothing is left
        bool pop(Piece& out) {
            std::unique_lock<std::mutex> lock(m_);
            ready_.wait(lock, [&] { return !q_.empty() || done_; });
            if (q_.empty()) {
                return false;
            }
            out = std::move(q_.front());
            q_.pop_front();
            queued_ -= out.states.size();
            space_.notify_one();
            return true;
        }

        void finish() {
            std::lock_guard<std::mutex> lock(m_);
            done_ = true;
            ready_.notify_all();
        }

    private:
        std::mutex m_;
        std::condition_variable ready_, space_;
        std::deque<Piece> q_;
        u64 queued_ = 0;
        u64 cap_;
        bool done_ = false;
    };

    struct GpuPassResult {
        mbgpu_stats stats{};
        u64 bloomPasses = 0;
        double probeWaitSeconds = 0;   // the probe threads still busy after the GPU finished
    };

    // One probe pass on the GPU: Bloom blocks of [pBegin, pEnd) to the device, every
    // seed streamed; the leaves that pass the Bloom test are finished on the CPU into
    // `candidates` by `threads` probe threads while the GPU streams. `progress` gets the
    // number of seeds done: once at the end, or per slice of seeds on 6-move streams
    // (hours per pass), each slice one mbgpu_stream call.
    inline GpuPassResult gpuProbePass(GpuModule& gpu, const FingerprintRange& index, const JVec<B1B2>& seeds,
                                      const std::vector<u8>& seedMoves, const i32 streamDepth, const u32 threads,
                                      std::vector<KeyedState>& candidates,
                                      const std::function<void(u64)>& progress = {}) {
        GpuPassResult r;
        if (gpu.setFilter(index.bloom.data(), index.meta.bloomBlocksPerPartition, index.pBegin, index.pEnd) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        ProbeQueue queue(64ULL << 20);   // ~2 GB of leaves at most
        std::vector<std::vector<KeyedState>> found(threads);
        std::vector<u64> passes(threads, 0);
        std::vector<std::thread> workers;
        for (u32 t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                ProbeQueue::Piece p;
                while (queue.pop(p)) {
                    const std::size_t n = p.states.size();
                    for (std::size_t lo = 0; lo < n; lo += PROBE_GROUP) {
                        const u32 cnt = static_cast<u32>(std::min<std::size_t>(PROBE_GROUP, n - lo));
                        passes[t] += index.probeGroup(&p.states[lo], &p.keys[lo], cnt, found[t]);
                    }
                }
            });
        }
        const auto* states = reinterpret_cast<const uint64_t*>(seeds.data());
        const u64 total = seeds.size();
        // 6-move streams: 64 slices (minutes each on 17-move runs); shorter streams: one call
        const u64 slice = streamDepth >= 6 ? std::max<u64>(1, (total + 63) / 64) : std::max<u64>(1, total);
        int rc = 0;
        for (u64 off = 0; off < total && rc == 0; off += slice) {
            const u64 n = std::min(slice, total - off);
            mbgpu_stats st{};
            rc = gpu.stream(states + 2 * off, seedMoves.empty() ? nullptr : seedMoves.data() + off, n, streamDepth,
                            &ProbeQueue::sink, &queue, &st);
            for (int d = 0; d <= MBGPU_MAX_DEPTH; ++d) {
                r.stats.nodes[d] += st.nodes[d];
            }
            r.stats.leaves += st.leaves;
            r.stats.inRange += st.inRange;
            r.stats.passed += st.passed;
            r.stats.launches += st.launches;
            r.stats.seconds += st.seconds;
            r.stats.retries += st.retries;
            if (rc == 0 && progress) {
                progress(n);
            }
        }
        Timer wait;
        queue.finish();
        for (std::thread& w : workers) {
            w.join();
        }
        r.probeWaitSeconds = wait.getSeconds();
        if (rc != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        gpu.setFilter(nullptr, 0, 0, PARTITION_COUNT);   // the Bloom blocks' VRAM back before the next pass
        for (u32 t = 0; t < threads; ++t) {
            r.bloomPasses += passes[t];
            candidates.insert(candidates.end(), found[t].begin(), found[t].end());
        }
        return r;
    }

} // namespace ssd
