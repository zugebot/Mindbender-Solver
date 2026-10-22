#pragma once
// src/solver/bounds/parent_prune.hpp
//
// The PARENT_PRUNE hooks for PermStream (perm_stream.hpp): asked per lane
// (5 sibling nodes) at every level above the leaves, they mark the nodes
// whose whole subtree cannot reach the table. LanePrune uses the lane bound;
// LaneAndPatternPrune adds the pattern databases (up to 2 levels up).
//
// LanePrune evaluates the five siblings cheaply, with exactly the same result
// as bound() > limit (LanePruneRef, kept for tests and A/B timing):
// - the moved lane's own half is the same for all five and for their parent
//   (shifting a row rotates it, and a row's distance is the same for all its
//   rotations, colour mix and rotation kind alike; the other rows do not move),
//   so it is computed once;
// - the other half is read lane by lane and a node is dropped as soon as the
//   part read so far already puts h over the limit (h only grows with more
//   lanes). Almost every node at the last levels is dropped, usually by the
//   first lane read, so the search starts at the lane that dropped the
//   previous sibling. Kept nodes read all six lanes, as before.
//
// LanePruneLeaves also asks it for the leaves (BELOW = 0): the stream then
// drops leaves with h > limit before hashing them, which the "both" probe mode
// otherwise did per leaf in its sink after the hash (the same leaves survive).

#include "src/solver/bounds/lane_bound.hpp"
#include "src/solver/bounds/pattern_db.hpp"

namespace bounds {

    // PermStream's parent-level prune (perm_stream.hpp): a node with BELOW
    // moves still to come is dropped when h > limit + BELOW (h changes by at
    // most 1 per move, so no descendant can be within `limit`). Static so the
    // stream's template signature is unchanged; set once per run, read-only.
    inline const lane_bound::LaneTables* g_laneTables = nullptr;
    inline int g_laneLimit = 0;

    // The plain definition: bound() of every node.
    struct LanePruneRef {
        static constexpr bool ENABLED = true;
        static constexpr bool LEAVES = false;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5], bool) {
            u32 m = 0;
            for (u32 k = 0; k < 5; ++k) {
                m |= static_cast<u32>(lane_bound::bound(nodes[k], *g_laneTables) > g_laneLimit + BELOW) << k;
            }
            return m;
        }
    };

    // nodes = one lane's five shifts; colLane: the shifted lane is a column.
    struct LanePrune {
        static constexpr bool ENABLED = true;
        static constexpr bool LEAVES = false;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5], const bool colLane) {
            const lane_bound::LaneTables& t = *g_laneTables;
            const int limit = g_laneLimit + BELOW;
            // the moved lane's half: rows for a row shift, columns for a column shift
            u32 keptBits = 0;
            for (u32 i = 0; i < 6; ++i) {
                keptBits |= colLane ? lane_bound::colLaneBits(nodes[0], t, i) : lane_bound::rowLaneBits(nodes[0], t, i);
            }
            const int kept = lane_bound::maxDistOf(keptBits);
            if (lane_bound::combineHalves(kept, 0, t.fat) > limit) {
                return 31U;
            }
            u32 m = 0, first = 0;
            for (u32 k = 0; k < 5; ++k) {
                u32 bits = 0;
                for (u32 j = 0; j < 6; ++j) {
                    const u32 i = first + j < 6 ? first + j : first + j - 6;
                    bits |= colLane ? lane_bound::rowLaneBits(nodes[k], t, i) : lane_bound::colLaneBits(nodes[k], t, i);
                    if (lane_bound::combineHalves(kept, lane_bound::maxDistOf(bits), t.fat) > limit) {
                        m |= 1U << k;
                        first = i;
                        break;
                    }
                }
            }
            return m;
        }
    };

    // LanePrune at every level, the leaves included.
    struct LanePruneLeaves : LanePrune {
        static constexpr bool LEAVES = true;
    };

    // Lane bound, then the grouped pattern databases.
    // Per lane (5 nodes): every table entry still needed is located and
    // prefetched first, then tested, so the cache misses overlap instead of
    // waiting one after another.
    struct LaneAndPatternPrune {
        static constexpr bool ENABLED = true;
        static constexpr bool LEAVES = false;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5], const bool colLane) {
            u32 m = LanePrune::skipMask<BELOW>(nodes, colLane);
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
