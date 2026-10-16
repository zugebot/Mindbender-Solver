#pragma once
// src/solver/bounds/audit.hpp
//
// Run-start audits of the pruning bounds on the run's own seeds: the fast
// tables equal the reference definition, the bound changes by at most 1 per
// move, and pruned streams keep exactly the wanted leaves of full streams.
// Any failure turns the bound off for the run. Also measures skip shares.

#include "src/chuzzles/include.hpp"
#include "src/utils/parallel_for.hpp"
#include "src/solver/bounds/parent_prune.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace bounds {

    struct BoundAudit {
        bool ok = true;
        std::string why;
        u64 nodes = 0, leaves = 0, edges = 0;
        double leafSkip = 0.0, parentSkip = 0.0;   // sampled shares
        u64 streamChecks = 0;
    };

    template<typename Fn>
    struct StateCallbackSink {
        Fn& fn;
        void operator()(JVec<B1B2>& states, JVec<u64>&, const u32 count) {
            for (u32 i = 0; i < count; ++i) {
                fn(states[i]);
            }
        }
    };

    template<i32 DEPTH, typename PRUNE, typename Fn>
    void streamStates(const B1B2& root, Fn&& fn) {
        StateCallbackSink<Fn> sink{fn};
        if constexpr (std::is_same_v<PRUNE, perm_stream_detail::NoParentPrune>) {
            PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, DEPTH>(makeBoardFromState(root), sink, 4096);
        } else {
            PermStream<B1B2>::streamDepthPruned<eSequenceDir::ASCENDING, DEPTH, PRUNE>(makeBoardFromState(root), sink, 4096);
        }
    }

    // On ~256 seeds spread over the seed list: (1) the fast lane tables give
    // the reference bound on every node and leaf; (2) on every sampled edge
    // (node -> child) h changes by at most 1, which the parent prune relies
    // on; (3) on 32 seeds the pruned stream keeps exactly the leaves with
    // h <= limit that the full stream has. Also measures the skip shares.
    template<i32 STREAM_DEPTH>
    BoundAudit auditLaneBound(const JVec<B1B2>& seeds, const lane_bound::LaneTables& tabs, const int limit,
                              const u32 threads, const B1B2& target) {
        BoundAudit a;
        const std::size_t samples = std::min<std::size_t>(STREAM_DEPTH >= 5 ? 8 : 256, seeds.size());
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, samples));
        std::atomic<u64> nodes{0}, leaves{0}, edges{0}, nodesSkip{0}, leavesSkip{0}, leavesUnderSkip{0}, bad{0};
        std::mutex whyLock;
        std::string firstWhy;
        auto fail = [&](const std::string& why) {
            bad.fetch_add(1);
            std::lock_guard<std::mutex> lock(whyLock);
            if (firstWhy.empty()) {
                firstWhy = why;
            }
        };
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            const B1B2& seed = seeds[k * step];
            // nodes one move above the leaves of the STREAM_DEPTH stream
            streamStates<STREAM_DEPTH - 1, perm_stream_detail::NoParentPrune>(seed, [&](const B1B2& node) {
                const int hn = lane_bound::bound(node, tabs);
                if (hn != lane_bound::referenceBound(node, target, tabs.kind)) {
                    fail("fast bound != reference on a node");
                }
                nodes.fetch_add(1, std::memory_order_relaxed);
                const bool skipNode = hn > limit + 1;
                nodesSkip.fetch_add(skipNode, std::memory_order_relaxed);
                streamStates<1, perm_stream_detail::NoParentPrune>(node, [&](const B1B2& leaf) {
                    const int hl = lane_bound::bound(leaf, tabs);
                    if (hl != lane_bound::referenceBound(leaf, target, tabs.kind)) {
                        fail("fast bound != reference on a leaf");
                    }
                    if (hl > hn + 1 || hl < hn - 1) {
                        fail("bound changed by more than 1 in one move");
                    }
                    leaves.fetch_add(1, std::memory_order_relaxed);
                    edges.fetch_add(1, std::memory_order_relaxed);
                    leavesSkip.fetch_add(hl > limit, std::memory_order_relaxed);
                    leavesUnderSkip.fetch_add(skipNode, std::memory_order_relaxed);
                });
            });
        });
        // (3) pruned vs full stream, wanted leaves identical
        const std::size_t streamSeeds = std::min<std::size_t>(STREAM_DEPTH >= 5 ? 2 : 32, seeds.size());
        const std::size_t sstep = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, streamSeeds));
        g_laneTables = &tabs;
        g_laneLimit = limit;
        std::atomic<u64> checks{0};
        parallelFor(threads, (seeds.size() + sstep - 1) / sstep, [&](u32, const std::size_t k) {
            std::vector<B1B2> full, pruned;
            streamStates<STREAM_DEPTH, perm_stream_detail::NoParentPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (lane_bound::bound(s, tabs) <= limit) {
                    full.push_back(s);
                }
            });
            streamStates<STREAM_DEPTH, LanePrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (lane_bound::bound(s, tabs) <= limit) {
                    pruned.push_back(s);
                }
            });
            std::sort(full.begin(), full.end());
            std::sort(pruned.begin(), pruned.end());
            if (full != pruned) {
                fail("pruned stream lost or added a wanted leaf");
            }
            checks.fetch_add(full.size(), std::memory_order_relaxed);
        });
        a.nodes = nodes.load();
        a.leaves = leaves.load();
        a.edges = edges.load();
        a.streamChecks = checks.load();
        a.leafSkip = static_cast<double>(leavesSkip.load()) / static_cast<double>(std::max<u64>(1, a.leaves));
        a.parentSkip = static_cast<double>(leavesUnderSkip.load()) / static_cast<double>(std::max<u64>(1, a.leaves));
        a.ok = bad.load() == 0;
        a.why = firstWhy;
        return a;
    }

    // Grouped pattern databases on this run's seeds: zero at the target; on
    // sampled real edges, every child of a "> limit + 1" node is "> limit";
    // the pruned stream (colour-mix + tables at parents) keeps exactly the
    // leaves the full stream has that neither bound rules out. Also measures
    // the extra share of whole last levels the tables skip.
    template<i32 STREAM_DEPTH>
    BoundAudit auditPatternDb(const JVec<B1B2>& seeds, const lane_bound::LaneTables& tabs, const int limit,
                              const u32 threads, const B1B2& target) {
        BoundAudit a;
        std::atomic<u64> nodes{0}, edges{0}, nodeSkipMix{0}, nodeSkipAny{0}, bad{0}, checks{0};
        std::mutex whyLock;
        std::string firstWhy;
        auto fail = [&](const std::string& why) {
            bad.fetch_add(1);
            std::lock_guard<std::mutex> lock(whyLock);
            if (firstWhy.empty()) {
                firstWhy = why;
            }
        };
        for (u32 i = 0; i < pattern_db::g_tableCount; ++i) {
            if (pattern_db::g_tables[i]->code(target) != 0) {
                fail("table " + pattern_db::g_tables[i]->name() + " is not 0 at the target");
            }
        }
        auto anyFar = [](const B1B2& s, const bool node) {
            for (u32 i = 0; i < pattern_db::g_tableCount; ++i) {
                if (node ? pattern_db::g_tables[i]->farNode(s) : pattern_db::g_tables[i]->farLeaf(s)) {
                    return true;
                }
            }
            return false;
        };
        const std::size_t samples = std::min<std::size_t>(STREAM_DEPTH >= 5 ? 8 : 256, seeds.size());
        const std::size_t step = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, samples));
        parallelFor(threads, (seeds.size() + step - 1) / step, [&](u32, const std::size_t k) {
            streamStates<STREAM_DEPTH - 1, perm_stream_detail::NoParentPrune>(seeds[k * step], [&](const B1B2& node) {
                nodes.fetch_add(1, std::memory_order_relaxed);
                const bool mixFar = lane_bound::bound(node, tabs) > limit + 1;
                const bool pdbFar = anyFar(node, true);
                nodeSkipMix.fetch_add(mixFar, std::memory_order_relaxed);
                nodeSkipAny.fetch_add(mixFar || pdbFar, std::memory_order_relaxed);
                if (pdbFar) {
                    streamStates<1, perm_stream_detail::NoParentPrune>(node, [&](const B1B2& leaf) {
                        edges.fetch_add(1, std::memory_order_relaxed);
                        if (!anyFar(leaf, false)) {
                            fail("a far node has a child within the limit");
                        }
                    });
                }
            });
        });
        const std::size_t streamSeeds = std::min<std::size_t>(STREAM_DEPTH >= 5 ? 2 : 32, seeds.size());
        const std::size_t sstep = std::max<std::size_t>(1, seeds.size() / std::max<std::size_t>(1, streamSeeds));
        parallelFor(threads, (seeds.size() + sstep - 1) / sstep, [&](u32, const std::size_t k) {
            std::vector<B1B2> full, pruned;
            auto wanted = [&](const B1B2& s) { return lane_bound::bound(s, tabs) <= limit && !anyFar(s, false); };
            streamStates<STREAM_DEPTH, perm_stream_detail::NoParentPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (wanted(s)) {
                    full.push_back(s);
                }
            });
            streamStates<STREAM_DEPTH, LaneAndPatternPrune>(seeds[k * sstep], [&](const B1B2& s) {
                if (wanted(s)) {
                    pruned.push_back(s);
                }
            });
            std::sort(full.begin(), full.end());
            std::sort(pruned.begin(), pruned.end());
            if (full != pruned) {
                fail("pruned stream lost or added a wanted leaf");
            }
            checks.fetch_add(full.size(), std::memory_order_relaxed);
        });
        a.nodes = nodes.load();
        a.edges = edges.load();
        a.streamChecks = checks.load();
        a.parentSkip = static_cast<double>(nodeSkipAny.load()) / static_cast<double>(std::max<u64>(1, a.nodes));
        a.leafSkip = static_cast<double>(nodeSkipMix.load()) / static_cast<double>(std::max<u64>(1, a.nodes));
        a.ok = bad.load() == 0;
        a.why = firstWhy;
        return a;
    }

} // namespace bounds
