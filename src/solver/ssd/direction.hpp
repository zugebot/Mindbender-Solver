#pragma once
// src/solver/ssd/direction.hpp
//
// --direction auto|forward|reverse (2026-11-04). A search can stream from the start toward the
// goal (forward) or from the goal toward the start (reverse: the roots swap, the goal side is
// built around the start, and recovery turns the paths around). Both find the same solutions;
// what differs is how well the lane bound prunes toward the other root. A structured target
// (16-3's anti-diagonal stripes) keeps the bound low over a large region, so the stream toward
// it is bushy; toward a random-looking board it prunes like other puzzles. And the other way
// round: 15-1 visits 1.75x more nodes in reverse (changes-2026-10-16.md section 12).
//
// auto (the default with --gpu on normal boards): before anything is built, stream a strided
// sample of each direction's seeds on the GPU with the run's bound and stream depth, count the
// nodes it expands, scale by each direction's seed count, and keep the cheaper direction. The
// count is deterministic (the same seeds, the same canonical stream), unlike timing. 2x2 boards
// have no bound that prunes, so both directions cost about the same: forward.

#include "src/solver/ssd/gpu_stream.hpp"

#include <algorithm>
#include <string>

namespace ssd {

    struct DirectionSample {
        u64 seeds = 0;        // the direction's seeds (exact depth S, after the first-move rule)
        u64 sampled = 0;
        double parents = 0;   // nodes the sample expanded (levels 0 .. stream depth - 1)
        double leaves = 0;    // leaves it kept
        double seconds = 0;
        // expanded nodes of the whole stream, estimated from the sample
        MUND double estimate() const {
            return sampled == 0 ? 0 : (parents + leaves) * static_cast<double>(seeds) / static_cast<double>(sampled);
        }
    };

    inline DirectionSample sampleDirection(const Board& left, const Board& right, const Config& cfg, const int limit) {
        DirectionSample r;
        JVec<B1B2> seeds;
        buildSeeds(left, cfg.seedDepth, seeds);
        std::vector<u8> moves;
        if (cfg.seedLastMove) {
            if (buildLastMoves(left, cfg.seedDepth, seeds, moves, cfg.threads) != 0) {
                dropUncoded(seeds, moves);
            }
        }
        r.seeds = seeds.size();
        GpuModule& gpu = ensureGpu(cfg);
        const B1B2 target = right.asB1B2();
        struct Discard {
            static void sink(void*, const uint64_t*, const uint64_t*, const uint64_t*, const uint32_t*, const uint64_t) {}
        };
        if (gpu.setTarget(target.b1, target.b2, limit) != 0 || gpu.setFilter(nullptr, 0, 0, 1) != 0) {
            throw std::runtime_error("GPU: " + gpu.error());
        }
        // A strided sample, grown 4x until it takes a few seconds of GPU time (a deep stream costs
        // millions of nodes per seed) or reaches ~1/2000 of the seeds (at most 65536).
        const u64 cap = std::max<u64>(std::min<u64>(r.seeds, 256), std::min<u64>(r.seeds / 2000, 65536));
        mbgpu_stats st{};
        for (u64 want = std::min<u64>(r.seeds, 256);; want = std::min(cap, want * 4)) {
            const u64 step = std::max<u64>(1, r.seeds / std::max<u64>(1, want));
            std::vector<u64> states;
            std::vector<u8> prev;
            for (u64 i = 0; i < r.seeds && prev.size() < want; i += step) {
                states.push_back(seeds[i].b1);
                states.push_back(seeds[i].b2);
                prev.push_back(moves.empty() ? perm_stream_detail::NO_PREV_MOVE : moves[i]);
            }
            st = mbgpu_stats{};
            if (gpu.stream(states.data(), prev.data(), prev.size(), static_cast<int>(cfg.streamDepth), &Discard::sink,
                           nullptr, &st) != 0) {
                throw std::runtime_error("GPU: " + gpu.error());
            }
            r.sampled = prev.size();
            if (st.seconds >= 3.0 || want >= cap) {
                break;
            }
        }
        r.parents = 0;
        for (u32 d = 0; d < cfg.streamDepth && d <= MBGPU_MAX_DEPTH; ++d) {
            r.parents += static_cast<double>(st.nodes[d]);
        }
        r.leaves = static_cast<double>(st.leaves);
        r.seconds = st.seconds;
        gpu.setFilter(nullptr, 0, 0, PARTITION_COUNT);
        return r;
    }

    // Decides cfg.reverse for --direction auto; prints both estimates.
    inline bool chooseReverse(const Board& start, const Board& goal, const Config& cfg) {
        const int limit = static_cast<int>(cfg.tableDepth + (cfg.blockJoin ? 1 : 0));
        Timer timer;
        const DirectionSample fwd = sampleDirection(start, goal, cfg, limit);
        const DirectionSample rev = sampleDirection(goal, start, cfg, limit);
        const bool reverse = rev.estimate() < fwd.estimate();
        auto line = [](const char* name, const DirectionSample& s) {
            tcout << "  " << name << ": " << s.seeds << " seeds, " << s.sampled << " sampled: " << s.parents
                  << " nodes expanded, " << s.leaves << " leaves kept (" << s.seconds << "s) -> ~" << s.estimate()
                  << " for the whole stream\n";
        };
        tcout << "direction (auto, a sampled GPU stream each way, " << timer.getSeconds() << "s):\n";
        line("forward (start -> goal)", fwd);
        line("reverse (goal -> start)", rev);
        const double a = std::max(1.0, std::min(fwd.estimate(), rev.estimate()));
        tcout << "  -> " << (reverse ? "reverse" : "forward") << " ("
              << std::max(fwd.estimate(), rev.estimate()) / a << "x fewer nodes than the other way)\n";
        return reverse;
    }

} // namespace ssd
