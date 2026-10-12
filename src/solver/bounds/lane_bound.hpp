#pragma once
// src/solver/bounds/lane_bound.hpp
//
// Admissible lower bounds on the moves between a state and a fixed target,
// for skipping meet-in-the-middle work. Two kinds, both of the form
//     h = R + C,   R = max over rows of d(row, target row),
//                  C = max over columns of d(column, target column):
//
// - colour mix: d = how many cells of the lane's colour multiset differ from
//   the target lane's (6 - the common count).
// - rotation (default in the solver, never lower): d = the fewest cells that
//   differ from the target lane over the lane's 6 rotations.
//
// Why it holds: a row move rotates its row (so that row's d is unchanged:
// the multiset and the set of rotations stay the same; other rows are not
// touched) and changes exactly one cell of every column (each column's d
// changes by at most 1). A column move is the mirror image. So the column
// moves still needed are >= R, the row moves needed >= C, and every move
// changes h by at most 1. On 2x2 boards one move can shift two lanes, so
// h = ceil(R/2) + ceil(C/2).
// Since h changes by at most 1 per move, a node with r moves still to come
// below it and h > limit + r has no descendant within `limit` of the target.
//
// Example (target row R R G G B B, row R G B R G B): colour mix 0 (same
// colours), rotation 4 (every rotation has 4 cells wrong).
//
// Fast evaluation uses per-target lane tables built at run time (see
// LaneTables); `referenceBound` is the plain definition, used to audit them
// (bounds/audit.hpp). The PermStream hook that prunes with it is
// bounds/parent_prune.hpp.
// Checked on every known optimal solution (experimentBoundSample --check).

#include "src/chuzzles/board.hpp"
#include "src/utils/intrinsics/pext_u64.hpp"
#include "src/utils/intrinsics/clz.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace lane_bound {

    enum class Kind : u8 { Mix = 0, Rotation = 1 };

    inline const char* kindName(const Kind k) { return k == Kind::Mix ? "colour-mix" : "rotation"; }

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

    // Colour-mix bound from colour counts (kept for experimentBoundSample).
    inline int referenceBound(const B1B2& state, const LaneMixRef& target, const bool fat) {
        const LaneMixRef s = laneMixRef(state);
        int r = 0, c = 0;
        for (u32 i = 0; i < 6; ++i) {
            r = std::max(r, mixDistanceRef(s.rows[i], target.rows[i]));
            c = std::max(c, mixDistanceRef(s.cols[i], target.cols[i]));
        }
        return fat ? (r + 1) / 2 + (c + 1) / 2 : r + c;
    }

    // Lanes 0-5 = rows (cell k = column k), 6-11 = columns (cell k = row k).
    inline void laneCells(const B1B2& b, u8 (&out)[12][6]) {
        for (u32 y = 0; y < 6; ++y) {
            for (u32 x = 0; x < 6; ++x) {
                const u8 c = b.getColor(static_cast<u64>(x), static_cast<u64>(y));
                out[y][x] = c;
                out[6 + x][y] = c;
            }
        }
    }

    inline int rotationDistanceRef(const u8 (&a)[6], const u8 (&t)[6]) {
        int best = 6;
        for (u32 k = 0; k < 6; ++k) {
            int d = 0;
            for (u32 i = 0; i < 6; ++i) {
                d += a[(i + k) % 6] != t[i];
            }
            best = std::min(best, d);
        }
        return best;
    }

    inline int mixDistanceCellsRef(const u8 (&a)[6], const u8 (&t)[6]) {
        int common = 0;
        for (u32 c = 0; c < 8; ++c) {
            int na = 0, nt = 0;
            for (u32 i = 0; i < 6; ++i) {
                na += a[i] == c;
                nt += t[i] == c;
            }
            common += std::min(na, nt);
        }
        return 6 - common;
    }

    // Either kind, from the boards' cells (the audit's reference).
    inline int referenceBound(const B1B2& state, const B1B2& target, const Kind kind) {
        u8 s[12][6], t[12][6];
        laneCells(state, s);
        laneCells(target, t);
        int r = 0, c = 0;
        for (u32 L = 0; L < 12; ++L) {
            const int d = kind == Kind::Mix ? mixDistanceCellsRef(s[L], t[L]) : rotationDistanceRef(s[L], t[L]);
            (L < 6 ? r : c) = std::max(L < 6 ? r : c, d);
        }
        return target.getFatBool() ? (r + 1) / 2 + (c + 1) / 2 : r + c;
    }

    // ---------------- fast evaluation ----------------
    //
    // b1 holds rows 0-2 and b2 rows 3-5; row i of a half is 18 bits at
    // (2 - i) * 18, cell x of a row at bit 15 - 3x within it. A lane's 6 cells
    // are two triples (9 bits each, 3-bit groups j = 0, 1, 2 from the low end):
    //   row lanes:    first = cells 5, 4, 3   second = cells 2, 1, 0
    //   column lanes: first = pext of b1 = rows 2, 1, 0   second = pext of b2 = rows 5, 4, 3
    // Per lane L, the triple is mapped to an id that keeps what the distance
    // needs, then
    //   first[L][triple]  -> offset of L's table + id * ids_L
    //   second[L][triple] -> id
    //   dist[first + second] -> 1 << the lane's distance   (one flat u8 array)
    // Colour mix: id = the triple's colour counts capped at the target lane's
    // (order does not matter; 22-30 KB in all). Rotation: id = the triple's
    // cells as symbols, base S, where the symbols are the target lane's m
    // distinct colours plus one for any other colour (S = m + 1, S^6 entries
    // per lane; only equality with target colours matters). A lane is 3 loads
    // and an add; the max over 6 lanes is 5 ORs and one bit scan.

    constexpr u64 colMask(const u32 x) {
        return (7ULL << (36 + 15 - 3 * x)) | (7ULL << (18 + 15 - 3 * x)) | (7ULL << (15 - 3 * x));
    }

    inline constexpr u64 COL_MASK[6] = {colMask(0), colMask(1), colMask(2), colMask(3), colMask(4), colMask(5)};

    struct LaneTables {
        alignas(64) u32 first[12][512];     // lanes 0-5 rows, 6-11 columns
        alignas(64) u16 second[12][512];
        std::vector<u8> dist;
        bool fat = false;
        Kind kind = Kind::Mix;
        u64 bytes = 0;
    };

    inline void buildLaneTables(const B1B2& target, LaneTables& t, const Kind kind = Kind::Mix) {
        t.fat = target.getFatBool();
        t.kind = kind;
        u8 cells[12][6];   // target lane L, cell k
        laneCells(target, cells);
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
            u32 idOfTriple[512];
            u32 ids = 0;
            std::vector<u8> table;
            if (kind == Kind::Mix) {
                std::map<std::vector<u32>, u32> idOf;
                std::vector<std::vector<u32>> countsOf;
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
                ids = static_cast<u32>(countsOf.size());
                table.resize(static_cast<std::size_t>(ids) * ids);
                for (u32 a = 0; a < ids; ++a) {
                    for (u32 b = 0; b < ids; ++b) {
                        u32 common = 0;
                        for (u32 c = 0; c < m; ++c) {
                            common += std::min(countsOf[a][c] + countsOf[b][c], goalCount[c]);
                        }
                        table[a * ids + b] = static_cast<u8>(1U << (6 - common));
                    }
                }
            } else {
                const u32 S = m + 1;
                const u32 pw[3] = {1, S, S * S};
                ids = S * S * S;
                auto sym = [&](const u32 colour) { return classOf[colour] >= 0 ? static_cast<u32>(classOf[colour]) : m; };
                for (u32 v = 0; v < 512; ++v) {
                    idOfTriple[v] = sym(v & 7) + sym((v >> 3) & 7) * pw[1] + sym((v >> 6) & 7) * pw[2];
                }
                u8 goalSym[6];
                for (u32 k = 0; k < 6; ++k) {
                    goalSym[k] = static_cast<u8>(classOf[cells[lane][k]]);
                }
                table.resize(static_cast<std::size_t>(ids) * ids);
                for (u32 a = 0; a < ids; ++a) {
                    for (u32 b = 0; b < ids; ++b) {
                        u8 s[6];
                        for (u32 j = 0; j < 3; ++j) {
                            const u8 sa = static_cast<u8>(a / pw[j] % S), sb = static_cast<u8>(b / pw[j] % S);
                            s[lane < 6 ? 5 - j : 2 - j] = sa;   // first triple's cell positions
                            s[lane < 6 ? 2 - j : 5 - j] = sb;   // second triple's
                        }
                        table[a * ids + b] = static_cast<u8>(1U << rotationDistanceRef(s, goalSym));
                    }
                }
            }
            const std::size_t base = t.dist.size();
            if (base + table.size() > 0xFFFFFFFFULL) {
                throw std::runtime_error("lane tables exceed 32-bit offsets");
            }
            t.dist.insert(t.dist.end(), table.begin(), table.end());
            for (u32 v = 0; v < 512; ++v) {
                t.first[lane][v] = static_cast<u32>(base + static_cast<std::size_t>(idOfTriple[v]) * ids);
                t.second[lane][v] = static_cast<u16>(idOfTriple[v]);
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
        const u32 r = 31U - static_cast<u32>(my_clz(static_cast<int>(rowBits)));   // rowBits != 0: one bit per lane
        const u32 c = 31U - static_cast<u32>(my_clz(static_cast<int>(colBits)));
        return t.fat ? static_cast<int>((r + 1) / 2 + (c + 1) / 2) : static_cast<int>(r + c);
    }

} // namespace lane_bound
