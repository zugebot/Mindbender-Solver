#pragma once
// src/solver/group_pdb.hpp
//
// Grouped pattern databases: an admissible lower bound from an exactly
// solved simpler puzzle, for skipping meet-in-the-middle work.
//
// Pick a class of colours (e.g. colours 0, 4 and 7 merged). The abstract
// board has 1 in every cell whose colour is in the class and 0 elsewhere,
// with the 2x2 piece's position bits kept. Moves only permute cells, so the
// real move code runs on abstract boards unchanged, and every real solution
// is also an abstract one: the exact abstract distance (BFS over every
// placement of the class's cells) never exceeds the real distance. It also
// changes by at most 1 per move (BFS distances are consistent).
//
// Index: colex rank of the class's cell positions (or of the blank cells
// when the class has more than 18), x 25 piece positions on 2x2 boards.
// Each table keeps 2 bits per entry: code = min(3, max(0, distance - limit)),
// so a node with BELOW (<= 2) moves still to come below it can be dropped
// when code > BELOW. A 10-cell class: C(36,10) = 254 M entries, 64 MB.
//
// Measured (experimentBoundSample --pdb2, 14 moves, share of leaves |
// whole last levels skippable, colour-mix alone -> with 3 tables):
// 16-4 40% | 8% -> 98% | 55%; 17-5 26% | 4% -> 91% | 14%; 2x2 boards ~0 at
// the last level.

#include "src/chuzzles/board.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/utils/intrinsics/pext_u64.hpp"

#include <algorithm>
#include <atomic>
#include <immintrin.h>
#include <string>
#include <thread>
#include <vector>

namespace group_pdb {

    constexpr u64 LOW_BITS = 0111111111111111111ULL;   // low bit of each of 18 3-bit cells
    constexpr u64 MASK36 = (1ULL << 36) - 1;

    struct Binomials {
        u64 v[37][37] = {};
        Binomials() {
            for (u32 n = 0; n <= 36; ++n) {
                v[n][0] = 1;
                for (u32 k = 1; k <= 36; ++k) {
                    v[n][k] = n == 0 ? 0 : v[n - 1][k - 1] + v[n - 1][k];
                }
            }
        }
    };
    inline const Binomials BINOM;

    FORCEINLINE u64 colourFlags(const u64 half, const u32 c) {
        const u64 x = ~(half ^ (static_cast<u64>(c) * LOW_BITS));
        return x & (x >> 1) & (x >> 2) & LOW_BITS;
    }

    FORCEINLINE u64 colexRank(u64 m) {
        u64 r = 0;
        for (u32 i = 1; m != 0; ++i, m &= m - 1) {
            r += BINOM.v[__builtin_ctzll(m)][i];
        }
        return r;
    }

    FORCEINLINE u64 colexUnrank(u64 r, const u32 k) {
        u64 m = 0;
        u32 p = 36;
        for (u32 i = k; i >= 1; --i) {
            do {
                --p;
            } while (BINOM.v[p][i] > r);
            r -= BINOM.v[p][i];
            m |= 1ULL << p;
        }
        return m;
    }

    struct Table {
        u8 classMask = 0;           // bit c: colour c is in the class
        u32 cells = 0;              // cells of the class
        u32 rankK = 0;              // bits ranked (cells, or 36 - cells)
        bool complement = false;
        bool fat = false;
        int limit = 0;
        u64 entries = 0;
        u32 maxDist = 0;
        double buildSeconds = 0.0;
        std::vector<u64> bits;      // 2 bits per entry (see header)

        MUND std::string name() const {
            std::string s = "{";
            for (u32 c = 0; c < 8; ++c) {
                if (classMask & (1U << c)) {
                    s += std::to_string(c);
                }
            }
            return s + "}";
        }

        FORCEINLINE u64 indexOfMask(u64 m, const u32 fatXY) const {
            if (complement) {
                m = ~m & MASK36;
            }
            const u64 r = colexRank(m);
            return fat ? r * 25 + fatXY : r;
        }

        // Real state -> entry
        FORCEINLINE u64 indexOf(const B1B2& s) const {
            u64 f1 = 0, f2 = 0;
            for (u32 c = 0; c < 8; ++c) {
                if (classMask & (1U << c)) {
                    f1 |= colourFlags(s.b1, c);
                    f2 |= colourFlags(s.b2, c);
                }
            }
            return indexOfMask((my_pext_u64(f1, LOW_BITS) << 18) | my_pext_u64(f2, LOW_BITS), s.getFatXY());
        }

        // Abstract board (cells 0/1) -> entry
        FORCEINLINE u64 indexOfAbstract(const B1B2& a) const {
            return indexOfMask((my_pext_u64(a.b1, LOW_BITS) << 18) | my_pext_u64(a.b2, LOW_BITS), a.getFatXY());
        }

        B1B2 abstractOfIndex(u64 idx, const B1B2& meta) const {
            u32 fatXY = 0;
            if (fat) {
                fatXY = static_cast<u32>(idx % 25);
                idx /= 25;
            }
            u64 m = colexUnrank(idx, rankK);
            if (complement) {
                m = ~m & MASK36;
            }
            B1B2 x((meta.b1 & ~((1ULL << 54) - 1)) | _pdep_u64(m >> 18, LOW_BITS), _pdep_u64(m & ((1ULL << 18) - 1), LOW_BITS));
            if (fat) {
                x.setFatXY(fatXY / 5, fatXY % 5);
            }
            return x;
        }

        FORCEINLINE u32 codeAt(const u64 i) const {
            return static_cast<u32>(bits[i >> 5] >> ((i & 31) * 2)) & 3U;
        }
        FORCEINLINE void prefetchAt(const u64 i) const { __builtin_prefetch(&bits[i >> 5]); }
        FORCEINLINE u32 code(const B1B2& s) const { return codeAt(indexOf(s)); }
        FORCEINLINE bool farLeaf(const B1B2& s) const { return code(s) >= 1U; }   // > limit
        FORCEINLINE bool farNode(const B1B2& s) const { return code(s) >= 2U; }   // > limit + 1
    };

    template<typename Fn>
    inline void parallelChunks(const u32 threads, const u64 count, const u64 chunk, Fn&& fn) {
        std::atomic<u64> next{0};
        std::vector<std::thread> pool;
        for (u32 t = 0; t < threads; ++t) {
            pool.emplace_back([&] {
                for (u64 c; (c = next.fetch_add(1)) * chunk < count;) {
                    fn(c * chunk, std::min(count, (c + 1) * chunk));
                }
            });
        }
        for (auto& th : pool) {
            th.join();
        }
    }

    // Exact BFS from the target's abstract board, layer by layer over a u8
    // distance array (no frontier kept), then packed to 2 bits per entry.
    inline Table build(const B1B2& target, const u8 classMask, const int limit, const u32 threads) {
        Timer timer;
        Table t;
        t.classMask = classMask;
        t.fat = target.getFatBool();
        t.limit = limit;
        u64 f1 = 0, f2 = 0;
        for (u32 c = 0; c < 8; ++c) {
            if (classMask & (1U << c)) {
                f1 |= colourFlags(target.b1, c);
                f2 |= colourFlags(target.b2, c);
            }
        }
        const B1B2 goal((target.b1 & ~((1ULL << 54) - 1)) | f1, f2);
        t.cells = static_cast<u32>(__builtin_popcountll(f1) + __builtin_popcountll(f2));
        t.complement = t.cells > 18;
        t.rankK = t.complement ? 36 - t.cells : t.cells;
        t.entries = BINOM.v[36][t.rankK] * (t.fat ? 25 : 1);
        std::vector<u8> dist(t.entries, 255);
        dist[t.indexOfAbstract(goal)] = 0;
        for (u32 d = 0;; ++d) {
            std::atomic<u64> added{0};
            parallelChunks(threads, t.entries, 1ULL << 16, [&](const u64 lo, const u64 hi) {
                B1B2 children[60];
                u64 hashes[60];
                u64 local = 0;
                for (u64 i = lo; i < hi; ++i) {
                    if (dist[i] != d) {
                        continue;
                    }
                    const B1B2 x = t.abstractOfIndex(i, goal);
                    const std::size_t n = t.fat
                            ? frontier_recovery_detail::emitNoneChildren<true>(x, children, hashes)
                            : frontier_recovery_detail::emitNoneChildren<false>(x, children, hashes);
                    for (std::size_t k = 0; k < n; ++k) {
                        u8* slot = &dist[t.indexOfAbstract(children[k])];
                        u8 expected = 255;
                        if (*slot == 255 && __atomic_compare_exchange_n(slot, &expected, static_cast<u8>(d + 1), false,
                                                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            ++local;
                        }
                    }
                }
                added.fetch_add(local);
            });
            if (added.load() == 0) {
                break;
            }
            t.maxDist = d + 1;
        }
        t.bits.assign((t.entries + 31) / 32, 0);
        parallelChunks(threads, (t.entries + 31) / 32, 1ULL << 14, [&](const u64 lo, const u64 hi) {
            for (u64 w = lo; w < hi; ++w) {
                u64 word = 0;
                for (u64 j = 0; j < 32 && w * 32 + j < t.entries; ++j) {
                    const int dd = dist[w * 32 + j];
                    // unreachable (255) counts as far; it cannot occur for real states
                    const u64 c = static_cast<u64>(std::min(3, std::max(0, dd - limit)));
                    word |= c << (2 * j);
                }
                t.bits[w] = word;
            }
        });
        t.buildSeconds = timer.getSeconds();
        return t;
    }

    // Up to `maxTables` colour classes for `target`: every union of 1-4
    // colours whose table fits `maxEntries`, largest first, skipping a class
    // whose colours are all already covered by the chosen ones.
    inline std::vector<u8> chooseClasses(const B1B2& target, const u64 maxEntries, const u32 maxTables) {
        u32 counts[8] = {};
        for (u64 y = 0; y < 6; ++y) {
            for (u64 x = 0; x < 6; ++x) {
                ++counts[target.getColor(x, y)];
            }
        }
        const u64 mul = target.getFatBool() ? 25 : 1;
        struct Cand { u8 mask; u64 entries; };
        std::vector<Cand> cands;
        for (u32 mask = 1; mask < 256; ++mask) {
            u32 cells = 0, colours = 0;
            bool valid = true;
            for (u32 c = 0; c < 8; ++c) {
                if (mask & (1U << c)) {
                    valid &= counts[c] != 0;
                    cells += counts[c];
                    ++colours;
                }
            }
            if (!valid || colours > 4 || cells == 36) {
                continue;
            }
            const u32 k = std::min(cells, 36 - cells);
            const u64 entries = BINOM.v[36][k] * mul;
            if (k >= 7 && entries <= maxEntries) {
                cands.push_back({static_cast<u8>(mask), entries});
            }
        }
        std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.entries > b.entries; });
        std::vector<u8> chosen;
        u8 covered = 0;
        for (const Cand& c : cands) {
            if (chosen.size() >= maxTables) {
                break;
            }
            if ((c.mask & ~covered) == 0) {
                continue;
            }
            chosen.push_back(c.mask);
            covered |= c.mask;
        }
        return chosen;
    }

    // ---------------- stream hook ----------------
    inline const Table* g_tables[4] = {};
    inline u32 g_tableCount = 0;

} // namespace group_pdb
