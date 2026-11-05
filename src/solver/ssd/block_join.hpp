#pragma once
// src/solver/ssd/block_join.hpp
//
// --block (the free-block join, docs/changes-2026-11-04.md section 15): confirming candidates.
// A candidate is a left leaf Y (or, in GPU key mode 2, a parent standing for its leaves, marked
// with class_key::PARENT_FLAG) whose row- or column-class key matched the goal side's index.
// The GPU regenerates the goal side's states with both class keys (key mode 1) and hands back
// those whose key is a candidate's; a goal state W of the same class exactly one lane from Y
// (two from a parent) is a middle: Y, one move, W, then the goal side's depth to the goal.
// The hits are checked as they arrive, since one class can hold thousands of goal states.

#include "src/solver/ssd/class_key.hpp"
#include "src/solver/ssd/gpu_stream.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

namespace ssd {

    // `cand` sorted by key. `depth`: the goal side's stream below `states` (1: depth 6 from the
    // depth-5 frontier, 2: depth 7). Leaves the GPU in key mode 0.
    inline std::vector<B1B2> confirmBlockJoin(GpuModule& gpu, const uint64_t* states, const u8* prevMoves,
                                              const u64 count, const int depth, const std::vector<KeyedState>& cand) {
        Timer timer;
        std::vector<u64> keys(cand.size());
        for (std::size_t i = 0; i < cand.size(); ++i) {
            keys[i] = cand[i].key;
        }
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

        struct Sink {
            const std::vector<KeyedState>* cands;
            std::vector<u8> used;   // candidates one (two) lanes from some goal state
            std::vector<B1B2> middles;
            u64 hits = 0;

            static void sink(void* ctx, const uint64_t* b1, const uint64_t* b2, const uint64_t* hitKeys,
                             const uint32_t*, const uint64_t n) {
                auto* self = static_cast<Sink*>(ctx);
                self->hits += n;
                for (uint64_t i = 0; i < n; ++i) {
                    auto it = std::lower_bound(self->cands->begin(), self->cands->end(), hitKeys[i],
                                               [](const KeyedState& c, const u64 k) { return c.key < k; });
                    for (; it != self->cands->end() && it->key == hitKeys[i]; ++it) {
                        const bool parent = (it->state.b2 & class_key::PARENT_FLAG) != 0;
                        const u64 y1 = it->state.b1, y2 = it->state.b2 & ~class_key::PARENT_FLAG;
                        for (const bool cols : {false, true}) {
                            u64 ya = 0, yb = 0, wa = 0, wb = 0;
                            class_key::classCode(y1, y2, cols, ya, yb);
                            class_key::classCode(b1[i], b2[i], cols, wa, wb);
                            if (ya == wa && yb == wb
                                && class_key::differingLanes(y1, y2, b1[i], b2[i], cols) == (parent ? 2u : 1u)) {
                                self->middles.emplace_back(b1[i], b2[i]);
                                self->used[static_cast<std::size_t>(it - self->cands->begin())] = 1;
                            }
                        }
                    }
                }
            }
        } s{&cand, std::vector<u8>(cand.size(), 0), {}};

        if (!keys.empty()) {
            // the regenerated goal states themselves, both classes of every one
            if (gpu.setKeyMode == nullptr || gpu.setKeyMode(1) != 0 ||
                gpu.matchChildren(states, prevMoves, count, depth, keys.data(), keys.size(), &Sink::sink, &s,
                                  nullptr) != 0) {
                throw std::runtime_error("GPU: " + gpu.error());
            }
        }
        gpu.setKeyMode(0);
        std::sort(s.middles.begin(), s.middles.end());
        s.middles.erase(std::unique(s.middles.begin(), s.middles.end()), s.middles.end());
        u64 used = 0;
        for (const u8 u : s.used) used += u;
        tcout << "verify (block join): " << s.hits << " goal states of a candidate's class, " << used << " of "
              << cand.size() << " candidates one lane (a parent: two) from one, " << s.middles.size() << " middles, "
              << timer.getSeconds() << "s\n";
        return s.middles;
    }

    // --block: switch the GPU to class keys (mode 2) before the goal side is counted.
    inline void startBlockJoin(GpuModule* gpu) {
        if (gpu == nullptr || gpu->setKeyMode == nullptr || gpu->setKeyMode(2) != 0) {
            throw std::runtime_error("--block needs --gpu with the CUDA module (mbgpu_set_key_mode)");
        }
        tcout << "block join: leaves and goal states keyed by row and column class (each class once per parent)\n";
    }

} // namespace ssd
