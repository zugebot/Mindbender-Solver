#pragma once
// src/solver/colour_mix_bound.hpp
//
// Admissible lower bound on the moves between a state and a fixed target
// (the "colour mix" bound), for skipping meet-in-the-middle work.
//
// A row move rotates its row, so every row keeps its colour multiset, and
// changes exactly one cell of every column; a column move is the mirror
// image. So the column moves still needed are at least R = the largest, over
// rows, of how many cells of the row's colour multiset differ from the target
// row's, and the row moves needed are at least C (same over columns):
//     h = R + C
// On 2x2 boards one move can shift two lanes, so h = ceil(R/2) + ceil(C/2).
// One move changes h by at most 1, so if a node one move above the leaves has
// h > limit + 1, none of its children can be within `limit` of the target.
//
// Fast evaluation uses per-target lane tables built at run time (see
// LaneTables); `referenceBound` is the plain definition, used to audit them.
// Checked on every known optimal solution (experimentBoundSample --check).

#include "src/chuzzles/board.hpp"
#include "src/utils/intrinsics/pext_u64.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace colour_mix {

    // ---------------- reference definition ----------------

    struct LaneMixRef {
        u32 rows[6];
        u32 cols[6];
    };

    // Colour counts of every row and column: 8 colours x 4-bit counters.
    inline LaneMixRef laneMixRef(const B1B2& b) {
        LaneMixRef m{};
        for (u32 y = 0; y < 6; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const u32 c = b.getColor(static_cast<u64>(x), static_cast<u64>(y));
                m.rows[y] += 1U << (4 * c);
                m.cols[x] += 1U << (4 * c);
            }
        }
        return m;
    }

    inline int mixDistanceRef(const u32 a, const u32 b) {
        int common = 0;
        for (u32 c = 0; c < 8; ++c) {
            common += static_cast<int>(std::min((a >> (4 * c)) & 15U, (b >> (4 * c)) & 15U));
        }
        return 6 - common;
    }

    inline int referenceBound(const B1B2& state, const LaneMixRef& target, const bool fat) {
        const LaneMixRef s = laneMixRef(state);
        int r = 0, c = 0;
        for (u32 i = 0; i < 6; ++i) {
            r = std::max(r, mixDistanceRef(s.rows[i], target.rows[i]));
            c = std::max(c, mixDistanceRef(s.cols[i], target.cols[i]));
        }
        return fat ? (r + 1) / 2 + (c + 1) / 2 : r + c;
    }

    // ---------------- fast evaluation ----------------
    //
    // b1 holds rows 0-2 and b2 rows 3-5; row i of a half is 18 bits at
    // (2 - i) * 18, cell x of a row at bit 15 - 3x within it. A lane's 6 cells
    // are two triples (9 bits each). A lane's distance is 6 - sum over colours
    // of min(count, target count); for the two triples only each colour's
    // count up to its target count matters, and colours absent from the
    // target lane don't matter. Per lane L:
    //   first[L][triple]  -> offset of L's table + id * ids_L   (u16)
    //   second[L][triple] -> id                                 (u8)
    //   dist[first + second] -> 1 << the lane's distance         (one flat u8 array)
    // where id numbers the triple's capped colour counts. A lane is 3 loads and
    // an add; the max over 6 lanes is 5 ORs and one bit scan. 22-30 KB per
    // target on the puzzles measured (stays in L1).

    constexpr u64 colMask(const u32 x) {
        return (7ULL << (36 + 15 - 3 * x)) | (7ULL << (18 + 15 - 3 * x)) | (7ULL << (15 - 3 * x));
    }

    inline constexpr u64 COL_MASK[6] = {colMask(0), colMask(1), colMask(2), colMask(3), colMask(4), colMask(5)};

    struct LaneTables {
        alignas(64) u16 first[12][512];     // lanes 0-5 rows, 6-11 columns
        alignas(64) u8 second[12][512];
        std::vector<u8> dist;
        bool fat = false;
        u64 bytes = 0;
    };

    inline void buildLaneTables(const B1B2& target, LaneTables& t) {
        t.fat = target.getFatBool();
        u8 cells[12][6];   // target lane L, cell k
        for (u32 y = 0; y < 6; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const u8 c = target.getColor(static_cast<u64>(x), static_cast<u64>(y));
                cells[y][x] = c;
                cells[6 + x][y] = c;
            }
        }
        t.dist.clear();
        for (u32 lane = 0; lane < 12; ++lane) {
            int classOf[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            u32 goalCount[6] = {};
            u32 m = 0;
            for (u32 k = 0; k < 6; ++k) {
                if (classOf[cells[lane][k]] < 0) {
                    classOf[cells[lane][k]] = static_cast<int>(m++);
                }
                ++goalCount[classOf[cells[lane][k]]];
            }
            std::map<std::vector<u32>, u32> idOf;
            std::vector<std::vector<u32>> countsOf;
            u32 idOfTriple[512];
            for (u32 v = 0; v < 512; ++v) {
                std::vector<u32> counts(m, 0);
                for (u32 k = 0; k < 3; ++k) {
                    const int c = classOf[(v >> (3 * k)) & 7];
                    if (c >= 0) {
                        counts[c] = std::min(counts[c] + 1, goalCount[c]);
                    }
                }
                auto it = idOf.find(counts);
                if (it == idOf.end()) {
                    it = idOf.emplace(counts, static_cast<u32>(countsOf.size())).first;
                    countsOf.push_back(counts);
                }
                idOfTriple[v] = it->second;
            }
            const u32 ids = static_cast<u32>(countsOf.size());
            const std::size_t base = t.dist.size();
            if (base + static_cast<std::size_t>(ids) * ids > 65535) {
                throw std::runtime_error("colour-mix lane tables exceed 16-bit offsets");
            }
            t.dist.resize(base + static_cast<std::size_t>(ids) * ids);
            for (u32 a = 0; a < ids; ++a) {
                for (u32 b = 0; b < ids; ++b) {
                    u32 common = 0;
                    for (u32 c = 0; c < m; ++c) {
                        common += std::min(countsOf[a][c] + countsOf[b][c], goalCount[c]);
                    }
                    t.dist[base + a * ids + b] = static_cast<u8>(1U << (6 - common));
                }
            }
            for (u32 v = 0; v < 512; ++v) {
                t.first[lane][v] = static_cast<u16>(base + idOfTriple[v] * ids);
                t.second[lane][v] = static_cast<u8>(idOfTriple[v]);
            }
        }
        t.bytes = sizeof(t.first) + sizeof(t.second) + t.dist.size();
    }

    FORCEINLINE int bound(const B1B2& s, const LaneTables& t) {
        const u8* d = t.dist.data();
        // Unsigned 64-bit indices (no sign extension); each lane finished and
        // OR-ed in before the next starts, so few values are live at once.
        auto lane = [&](const u32 L, const u64 a9, const u64 b9) -> u32 {
            return d[static_cast<u64>(t.first[L][a9]) + static_cast<u64>(t.second[L][b9])];
        };
        u64 b1 = s.b1, b2 = s.b2;
        u32 rowBits = lane(0, (b1 >> 36) & 511U, (b1 >> 45) & 511U);
        rowBits |= lane(1, (b1 >> 18) & 511U, (b1 >> 27) & 511U);
        rowBits |= lane(2, b1 & 511U, (b1 >> 9) & 511U);
        rowBits |= lane(3, (b2 >> 36) & 511U, (b2 >> 45) & 511U);
        rowBits |= lane(4, (b2 >> 18) & 511U, (b2 >> 27) & 511U);
        rowBits |= lane(5, b2 & 511U, (b2 >> 9) & 511U);
        u32 colBits = 0;
        for (u32 x = 0; x < 6; ++x) {
            // Empty asm: b1/b2 "change" here, so GCC cannot hoist all twelve
            // pexts to the top (it did, and spilled them to the stack).
            asm volatile("" : "+r"(b1), "+r"(b2));
            colBits |= lane(6 + x, my_pext_u64(b1, COL_MASK[x]), my_pext_u64(b2, COL_MASK[x]));
        }
        const u32 r = 31U - static_cast<u32>(__builtin_clz(rowBits));
        const u32 c = 31U - static_cast<u32>(__builtin_clz(colBits));
        return t.fat ? static_cast<int>((r + 1) / 2 + (c + 1) / 2) : static_cast<int>(r + c);
    }

    // ---------------- stream hook ----------------
    //
    // PermStream's parent-level prune (perm_stream.hpp): skip a node one move
    // above the leaves when no child can be within `limit`. Static so the
    // stream's template signature is unchanged; set once per run, read-only.
    inline const LaneTables* g_parentTables = nullptr;
    inline int g_parentLimit = 0;

    // A node with BELOW moves still to come is dropped when h > limit + BELOW.
    struct ParentPrune {
        static constexpr bool ENABLED = true;
        template<int BELOW>
        static FORCEINLINE u32 skipMask(const B1B2 (&nodes)[5]) {
            u32 m = 0;
            for (u32 k = 0; k < 5; ++k) {
                m |= static_cast<u32>(bound(nodes[k], *g_parentTables) > g_parentLimit + BELOW) << k;
            }
            return m;
        }
    };

} // namespace colour_mix
