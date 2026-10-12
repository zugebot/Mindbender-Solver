#pragma once
// src/solver/bounds/parent_prune.hpp
//
// The PARENT_PRUNE hooks for PermStream (perm_stream.hpp): asked per lane
// (5 sibling nodes) at every level above the leaves, they mark the nodes
// whose whole subtree cannot reach the table. LanePrune uses the lane bound;
// LaneAndPatternPrune adds the pattern databases (up to 2 levels up).

#include "src/solver/bounds/lane_bound.hpp"
#include "src/solver/bounds/pattern_db.hpp"

namespace bounds {

    // PermStream's parent-level prune (perm_stream.hpp): a node with BELOW
    // moves still to come is dropped when h > limit + BELOW (h changes by at
    // most 1 per move, so no descendant can be within `limit`). Static so the
    // stream's template signature is unchanged; set once per run, read-only.
    inline const lane_bound::LaneTables* g_laneTables = nullptr;
    inline int g_laneLimit = 0;

    struct LanePrune {
        static constexpr bool ENABLED = true;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5]) {
            u32 m = 0;
            for (u32 k = 0; k < 5; ++k) {
                m |= static_cast<u32>(lane_bound::bound(nodes[k], *g_laneTables) > g_laneLimit + BELOW) << k;
            }
            return m;
        }
    };

    // Lane bound, then the grouped pattern databases.
    // Per lane (5 nodes): every table entry still needed is located and
    // prefetched first, then tested, so the cache misses overlap instead of
    // waiting one after another.
    struct LaneAndPatternPrune {
        static constexpr bool ENABLED = true;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5]) {
            u32 m = LanePrune::skipMask<BELOW>(nodes);
            const u32 nt = pattern_db::g_tableCount;
            if (m == 31U || nt == 0 || BELOW > 2) {   // the tables encode up to limit + 2
                return m;
            }
            u64 idx[4][5];
            for (u32 i = 0; i < nt; ++i) {
                for (u32 k = 0; k < 5; ++k) {
                    if ((m & (1U << k)) == 0) {
                        idx[i][k] = pattern_db::g_tables[i]->indexOf(nodes[k]);
                        pattern_db::g_tables[i]->prefetchAt(idx[i][k]);
                    }
                }
            }
            for (u32 i = 0; i < nt; ++i) {
                for (u32 k = 0; k < 5; ++k) {
                    if ((m & (1U << k)) == 0 && pattern_db::g_tables[i]->codeAt(idx[i][k]) > static_cast<u32>(BELOW)) {
                        m |= 1U << k;
                    }
                }
            }
            return m;
        }
    };

} // namespace bounds
