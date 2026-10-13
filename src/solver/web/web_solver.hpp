#pragma once
// src/solver/web/web_solver.hpp
//
// The browser solver (web/): every shortest solution of up to 10 moves, on
// normal and 2x2 boards, in a few hundred MB. It is separate from the native
// solvers because those need BMI2 and threads and WebAssembly has neither, so
// this is plain C++ (web/build_wasm.ps1 compiles it) and the page runs one copy
// per Web Worker. testWebSolver checks it against build/levels_final.
//
// Search for D moves: exact distance layers S_0..S_4 from the start and
// G_0..G_4 from the goal (layer k = the neighbours of layer k-1 that are in
// neither k-1 nor k-2). D <= 8: S_a meets G_b with a + b = D. D = 9 / 10:
// every state of S_4 is streamed 1 / 2 more moves and looked up in G_4. Each
// worker owns a hash partition of S_4 (the work splits with no shared memory)
// and builds G_4 itself.
// Every optimal path crosses S_a at move a and G_b at move D - b, so all of
// them are listed as: the paths start -> s (walking the S layers back), the
// middle moves, the paths y -> goal (walking the G layers forward). The 2-move
// middle stream visits one order of each commuting pair (two rows, or two
// columns) and lists both orders when it hits. On normal boards the rotation
// lane bound (src/solver/bounds/lane_bound.hpp) prunes the middle stream.

#include "src/utils/intrinsics/popcount.hpp"
#include "src/utils/processor.hpp"

#include <algorithm>
#include <string>
#include <vector>


namespace web_solver {

    static constexpr u64 LANE = 0x3FFFFULL;                // six 3-bit cells
    static constexpr u64 CELLS_HALF = (1ULL << 54) - 1;     // three rows
    static constexpr u32 FAT_X_SHIFT = 54;
    static constexpr u32 FAT_Y_SHIFT = 57;

    // a = rows 0-2, b = rows 3-5; row r at bit 18 * (r % 3), cell c at +3c.
    // 2x2 boards keep the piece's column (x) and row (y) of its top-left cell
    // in a's bits 54-59; normal boards leave them 0.
    struct State {
        u64 a = 0;
        u64 b = 0;

        bool operator==(const State& o) const { return a == o.a && b == o.b; }
        bool operator!=(const State& o) const { return !(*this == o); }
        bool operator<(const State& o) const { return a < o.a || (a == o.a && b < o.b); }
    };

    FORCEINLINE u64 hashState(const State& s) {
        u64 h = s.a * 0x9E3779B97F4A7C15ULL;
        h ^= (s.b ^ (s.b >> 29)) * 0xC2B2AE3D27D4EB4FULL;
        h ^= h >> 32;
        h *= 0xD6E8FEB86659FD93ULL;
        h ^= h >> 29;
        return h;
    }

    FORCEINLINE u32 fatX(const State& s) { return (s.a >> FAT_X_SHIFT) & 7; }
    FORCEINLINE u32 fatY(const State& s) { return (s.a >> FAT_Y_SHIFT) & 7; }
    FORCEINLINE void setFat(State& s, const u32 x, const u32 y) {
        s.a = (s.a & CELLS_HALF) | (static_cast<u64>(x) << FAT_X_SHIFT) | (static_cast<u64>(y) << FAT_Y_SHIFT);
    }

    FORCEINLINE u32 getCell(const State& s, const u32 r, const u32 c) {
        return ((r < 3 ? s.a : s.b) >> (18 * (r % 3) + 3 * c)) & 7;
    }
    FORCEINLINE void setCell(State& s, const u32 r, const u32 c, const u32 v) {
        const u32 sh = 18 * (r % 3) + 3 * c;
        u64& half = r < 3 ? s.a : s.b;
        half = (half & ~(7ULL << sh)) | (static_cast<u64>(v) << sh);
    }

    // Cell i of a lane sits at bit 3i, so moving every cell n places forward
    // (right for rows, down for columns) is a left rotation by 3n.
    FORCEINLINE u64 rotateLane(const u64 lane, const u32 n) {
        const u32 k = 3 * n;
        return ((lane << k) | (lane >> (18 - k))) & LANE;
    }

    FORCEINLINE u64 readRow(const State& s, const u32 r) {
        return ((r < 3 ? s.a : s.b) >> (18 * (r % 3))) & LANE;
    }
    FORCEINLINE void writeRow(State& s, const u32 r, const u64 lane) {
        const u32 sh = 18 * (r % 3);
        u64& half = r < 3 ? s.a : s.b;
        half = (half & ~(LANE << sh)) | (lane << sh);
    }

    static constexpr u64 COL_SPREAD = 7ULL | (7ULL << 18) | (7ULL << 36);
    FORCEINLINE u64 gather3(const u64 h) { return (h & 07) | ((h >> 15) & 070) | ((h >> 30) & 0700); }
    FORCEINLINE u64 spread3(const u64 v) { return (v & 07) | ((v & 070) << 15) | ((v & 0700) << 30); }

    FORCEINLINE u64 readCol(const State& s, const u32 c) {
        const u32 sh = 3 * c;
        return gather3(s.a >> sh) | (gather3(s.b >> sh) << 9);
    }
    FORCEINLINE void writeCol(State& s, const u32 c, const u64 lane) {
        const u32 sh = 3 * c;
        const u64 m = COL_SPREAD << sh;
        s.a = (s.a & ~m) | (spread3(lane & 0777) << sh);
        s.b = (s.b & ~m) | (spread3(lane >> 9) << sh);
    }


    // Moves: 0-29 row r by n (5r + n - 1), 30-59 column c by n, 60-64 the 2x2
    // piece's two rows by n, 65-69 its two columns by n. A 2x2 board has no
    // single-lane moves on the piece's rows or columns, and the piece may not
    // end across the board edge (the one amount that would put it at x or y 5).
    static constexpr u8 ROW_BASE = 0;
    static constexpr u8 COL_BASE = 30;
    static constexpr u8 FAT_ROW_BASE = 60;
    static constexpr u8 FAT_COL_BASE = 65;

    FORCEINLINE u32 moveAmount(const u8 id) { return id % 5 + 1; }
    FORCEINLINE u8 inverseMove(const u8 id) { return static_cast<u8>(id - id % 5 + (4 - id % 5)); }
    FORCEINLINE bool isColMove(const u8 id) { return (id >= COL_BASE && id < FAT_ROW_BASE) || id >= FAT_COL_BASE; }
    FORCEINLINE bool isFatMove(const u8 id) { return id >= FAT_ROW_BASE; }

    // Order of commuting moves of one orientation: the lane index, the top
    // (left) one for the 2x2 pair. Row moves never change the piece's row, nor
    // column moves its column, so the key holds across such a pair.
    FORCEINLINE u32 laneKey(const State& before, const u8 id) {
        if (id < COL_BASE) return id / 5;
        if (id < FAT_ROW_BASE) return (id - COL_BASE) / 5;
        if (id < FAT_COL_BASE) return fatY(before);
        return fatX(before);
    }

    inline State applyMove(const State& s, const u8 id) {
        State t = s;
        const u32 n = moveAmount(id);
        if (id < COL_BASE) {
            const u32 r = id / 5;
            writeRow(t, r, rotateLane(readRow(s, r), n));
        } else if (id < FAT_ROW_BASE) {
            const u32 c = (id - COL_BASE) / 5;
            writeCol(t, c, rotateLane(readCol(s, c), n));
        } else if (id < FAT_COL_BASE) {
            const u32 x = fatX(s), y = fatY(s);
            writeRow(t, y, rotateLane(readRow(s, y), n));
            writeRow(t, y + 1, rotateLane(readRow(s, y + 1), n));
            setFat(t, (x + n) % 6, y);
        } else {
            const u32 x = fatX(s), y = fatY(s);
            writeCol(t, x, rotateLane(readCol(s, x), n));
            writeCol(t, x + 1, rotateLane(readCol(s, x + 1), n));
            setFat(t, x, (y + n) % 6);
        }
        return t;
    }

    inline bool moveValid(const State& s, const bool fat, const u8 id) {
        if (!fat) return id < FAT_ROW_BASE;
        const u32 x = fatX(s), y = fatY(s);
        if (id < COL_BASE) { const u32 r = id / 5; return r != y && r != y + 1; }
        if (id < FAT_ROW_BASE) { const u32 c = (id - COL_BASE) / 5; return c != x && c != x + 1; }
        if (id < FAT_COL_BASE) return moveAmount(id) != 5 - x;
        return id < FAT_COL_BASE + 5 && moveAmount(id) != 5 - y;
    }

    // f(id, child) for every legal move, each lane read once for its 5 shifts.
    template<typename F>
    FORCEINLINE void forEachChild(const State& s, const bool fat, F&& f) {
        if (!fat) {
            for (u32 r = 0; r < 6; ++r) {
                const u64 lane = readRow(s, r);
                for (u32 n = 1; n <= 5; ++n) {
                    State t = s;
                    writeRow(t, r, rotateLane(lane, n));
                    f(static_cast<u8>(ROW_BASE + 5 * r + n - 1), t);
                }
            }
            for (u32 c = 0; c < 6; ++c) {
                const u64 lane = readCol(s, c);
                for (u32 n = 1; n <= 5; ++n) {
                    State t = s;
                    writeCol(t, c, rotateLane(lane, n));
                    f(static_cast<u8>(COL_BASE + 5 * c + n - 1), t);
                }
            }
            return;
        }
        const u32 x = fatX(s), y = fatY(s);
        for (u32 r = 0; r < 6; ++r) {
            if (r == y || r == y + 1) continue;
            const u64 lane = readRow(s, r);
            for (u32 n = 1; n <= 5; ++n) {
                State t = s;
                writeRow(t, r, rotateLane(lane, n));
                f(static_cast<u8>(ROW_BASE + 5 * r + n - 1), t);
            }
        }
        {
            const u64 laneA = readRow(s, y), laneB = readRow(s, y + 1);
            for (u32 n = 1; n <= 5; ++n) {
                if (n == 5 - x) continue;
                State t = s;
                writeRow(t, y, rotateLane(laneA, n));
                writeRow(t, y + 1, rotateLane(laneB, n));
                setFat(t, (x + n) % 6, y);
                f(static_cast<u8>(FAT_ROW_BASE + n - 1), t);
            }
        }
        for (u32 c = 0; c < 6; ++c) {
            if (c == x || c == x + 1) continue;
            const u64 lane = readCol(s, c);
            for (u32 n = 1; n <= 5; ++n) {
                State t = s;
                writeCol(t, c, rotateLane(lane, n));
                f(static_cast<u8>(COL_BASE + 5 * c + n - 1), t);
            }
        }
        {
            const u64 laneA = readCol(s, x), laneB = readCol(s, x + 1);
            for (u32 n = 1; n <= 5; ++n) {
                if (n == 5 - y) continue;
                State t = s;
                writeCol(t, x, rotateLane(laneA, n));
                writeCol(t, x + 1, rotateLane(laneB, n));
                setFat(t, x, (y + n) % 6);
                f(static_cast<u8>(FAT_COL_BASE + n - 1), t);
            }
        }
    }

    // "R14" = row 1 shifted right 4, "C453" = the 2x2 piece's columns 4+5
    // down 3. zeroBased gives the build/levels_final spelling (R03, C342).
    inline std::string moveName(const State& before, const u8 id, const bool zeroBased) {
        const u32 o = zeroBased ? 0 : 1;
        const u32 n = moveAmount(id);
        std::string s;
        if (id < COL_BASE) {
            s = "R" + std::to_string(id / 5 + o);
        } else if (id < FAT_ROW_BASE) {
            s = "C" + std::to_string((id - COL_BASE) / 5 + o);
        } else if (id < FAT_COL_BASE) {
            const u32 y = fatY(before);
            s = "R" + std::to_string(y + o) + std::to_string(y + 1 + o);
        } else {
            const u32 x = fatX(before);
            s = "C" + std::to_string(x + o) + std::to_string(x + 1 + o);
        }
        return s + std::to_string(n);
    }


    // An exact set of states for membership tests: sorted by hash, with
    // offsets for the hash's top bits and, on big sets, a bitset on its low
    // bits that turns most misses away before the state array is read.
    class StateSet {
    public:
        std::vector<State> states;
        std::vector<u32> offsets;
        std::vector<u64> filter;
        u32 bits = 1;
        u64 filterMask = 0;

        MUND std::size_t size() const { return states.size(); }

        MUND FORCEINLINE bool containsHashed(const State& s, const u64 h) const {
            if (!filter.empty() && !((filter[(h & filterMask) >> 6] >> (h & 63)) & 1)) {
                return false;
            }
            const u64 bucket = h >> (64 - bits);
            for (u32 i = offsets[bucket], e = offsets[bucket + 1]; i < e; ++i) {
                if (states[i] == s) return true;
            }
            return false;
        }

        MUND bool contains(const State& s) const { return containsHashed(s, hashState(s)); }

        // states must already be sorted by hash.
        void finish(const bool withFilter) {
            bits = 1;
            while (bits < 30 && (1ULL << bits) < states.size() / 2) ++bits;
            offsets.assign((1ULL << bits) + 1, 0);
            for (const State& s : states) ++offsets[(hashState(s) >> (64 - bits)) + 1];
            for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
            filter.clear();
            filterMask = 0;
            if (withFilter && states.size() > 4096) {
                u32 fbits = 6;
                while (fbits < 30 && (1ULL << fbits) < states.size() * 8) ++fbits;
                filterMask = (1ULL << fbits) - 1;
                filter.assign((1ULL << fbits) / 64, 0);
                for (const State& s : states) {
                    const u64 h = hashState(s) & filterMask;
                    filter[h >> 6] |= 1ULL << (h & 63);
                }
            }
        }

        MUND u64 bytes() const {
            return states.capacity() * sizeof(State) + offsets.capacity() * sizeof(u32)
                   + filter.capacity() * sizeof(u64);
        }
    };

    // Partition of S_4 a hash belongs to (bits apart from the bucket and filter bits).
    FORCEINLINE u32 partOf(const u64 h, const u32 parts) {
        return static_cast<u32>((h >> 30) & 0xFFFFFF) % parts;
    }

    // The states exactly one move beyond `prev`: its neighbours that are in
    // neither `prev` nor `prev2`, kept if in partition `part` of `parts`.
    // Built in passes over the hash range so the candidate buffer stays small.
    inline StateSet buildLayer(const StateSet& prev, const StateSet* prev2, const bool fat,
                               const u32 part, const u32 parts, const bool withFilter) {
        struct Cand {
            u64 h;
            State s;
        };
        const u64 expected = prev.size() * (fat ? 48 : 60) / parts;
        u32 passBits = 0;
        while ((expected >> passBits) > (1u << 20)) ++passBits;

        StateSet out;
        std::vector<Cand> buf;
        for (u64 pass = 0; pass < (1ULL << passBits); ++pass) {
            buf.clear();
            for (const State& p : prev.states) {
                forEachChild(p, fat, [&](u8, const State& t) {
                    const u64 h = hashState(t);
                    if (passBits != 0 && (h >> (64 - passBits)) != pass) return;
                    if (parts > 1 && partOf(h, parts) != part) return;
                    buf.push_back({h, t});
                });
            }
            std::sort(buf.begin(), buf.end(), [](const Cand& x, const Cand& y) {
                return x.h < y.h || (x.h == y.h && x.s < y.s);
            });
            for (std::size_t i = 0; i < buf.size(); ++i) {
                if (i != 0 && buf[i].h == buf[i - 1].h && buf[i].s == buf[i - 1].s) continue;
                if (prev.containsHashed(buf[i].s, buf[i].h)) continue;
                if (prev2 != nullptr && prev2->containsHashed(buf[i].s, buf[i].h)) continue;
                out.states.push_back(buf[i].s);
            }
        }
        std::vector<Cand>().swap(buf);
        out.states.shrink_to_fit();
        out.finish(withFilter);
        return out;
    }

    inline StateSet singleton(const State& s) {
        StateSet out;
        out.states.push_back(s);
        out.finish(false);
        return out;
    }


    class Solver {
    public:
        static constexpr int MAX_LAYER = 4;
        static constexpr int MAX_DEPTH = 10;
        static constexpr int SMALL_MAX_DEPTH = 6;     // the most layers 0-3 reach
        static constexpr std::size_t MAX_LISTED = 100000;

        // Result of the last depth that was searched.
        u64 solutionCount = 0;
        std::vector<std::vector<u8>> solutions;       // capped at MAX_LISTED
        u64 invalidSolutions = 0;                     // must stay 0 (each listed path is replayed)
        u64 streamedStates = 0;

        // cells: 36 colours 0-7, row-major. fat: the 2x2 piece's top-left
        // (column x, row y) on each board. Returns "" or the reason it can't be solved.
        std::string setup(const u8 startCells[36], const u8 goalCells[36], const bool fatIn,
                          const u32 sx, const u32 sy, const u32 gx, const u32 gy) {
            fat = fatIn;
            start = makeState(startCells);
            goal = makeState(goalCells);
            if (fat) {
                if (sx > 4 || sy > 4 || gx > 4 || gy > 4) return "the 2x2 piece must lie inside the board";
                setFat(start, sx, sy);
                setFat(goal, gx, gy);
            }
            u32 counts[8] = {};
            for (u32 i = 0; i < 36; ++i) {
                if (startCells[i] > 7 || goalCells[i] > 7) return "colours must be 0-7";
                ++counts[startCells[i]];
                --counts[goalCells[i]];
            }
            for (const u32 c : counts) {
                if (c != 0) return "the start and goal boards have different colour counts";
            }
            S.clear();
            G.clear();
            S.push_back(singleton(start));
            G.push_back(singleton(goal));
            laneTablesBuilt = false;
            return "";
        }

        MUND bool isFat() const { return fat; }
        MUND std::size_t layerSize(const bool goalSide, const u32 k) const {
            const auto& L = goalSide ? G : S;
            return k < L.size() ? L[k].size() : 0;
        }

        MUND u64 memoryBytes() const {
            u64 total = rowTab.capacity() + colTab.capacity();
            for (const auto& l : S) total += l.bytes();
            for (const auto& l : G) total += l.bytes();
            return total;
        }

        // Smallest D <= min(maxDepth, 6) with a solution (results filled), or -1.
        int solveSmall(const int maxDepth) {
            for (int d = 0; d <= std::min(maxDepth, SMALL_MAX_DEPTH); ++d) {
                const int b = d / 2, a = d - b;
                ensureStart(a);
                ensureGoal(b);
                std::vector<Hit> hits;
                intersect(S[a], G[b], hits);
                if (!hits.empty()) {
                    listSolutions(hits, a, b);
                    if (solutionCount != 0) return d;
                }
            }
            return -1;
        }

        // Phase 2 (D = 7-10): G_4 in full, then this worker's part of S_4.
        void buildGoal4() {
            ensureGoal(3);
            if (G.size() == 4) G.push_back(buildLayer(G[3], &G[2], fat, 0, 1, true));
        }

        void buildStart4(const u32 part, const u32 parts) {
            ensureStart(3);
            S.resize(4);
            S.push_back(buildLayer(S[3], &S[2], fat, part, parts, false));
        }

        // Begin depth D (7-10) over this worker's part of S_4; returns the
        // number of seeds to step through.
        u64 beginDepth(const int d) {
            depth = d;
            middle = d - 8 < 0 ? 0 : d - 8;
            goalLayer = d == 7 ? 3 : 4;
            cursor = 0;
            hits.clear();
            streamedStates = 0;
            solutionCount = 0;
            solutions.clear();
            invalidSolutions = 0;
            prune = !fat && middle > 0;
            if (prune && !laneTablesBuilt) buildLaneTables();
            return S[4].size();
        }

        // Processes up to maxSeeds more seeds; true when the part is done.
        bool step(const u64 maxSeeds) {
            const auto& seeds = S[4].states;
            const u64 end = std::min<u64>(seeds.size(), cursor + maxSeeds);
            const StateSet& target = G[goalLayer];
            for (; cursor < end; ++cursor) {
                const State& s = seeds[cursor];
                if (middle == 0) {
                    if (target.contains(s)) hits.push_back({s, s, {0, 0}, 0});
                    continue;
                }
                if (prune && bound(s) > middle + 4) continue;
                if (middle == 1) {
                    forEachChild(s, fat, [&](const u8 id, const State& t) {
                        ++streamedStates;
                        if (target.contains(t)) hits.push_back({s, t, {id, 0}, 1});
                    });
                    continue;
                }
                forEachChild(s, fat, [&](const u8 id1, const State& t) {
                    if (prune && bound(t) > 5) return;
                    const bool col1 = isColMove(id1);
                    const u32 key1 = laneKey(s, id1);
                    forEachChild(t, fat, [&](const u8 id2, const State& u) {
                        bool swapToo = false;
                        if (isColMove(id2) == col1) {
                            const u32 key2 = laneKey(t, id2);
                            if (key2 < key1) return;                       // listed from the other order
                            if (key2 == key1 && !isFatMove(id1)) return;   // same lane twice: never shortest
                            swapToo = key2 != key1;
                        }
                        ++streamedStates;
                        if (!target.contains(u)) return;
                        hits.push_back({s, u, {id1, id2}, 2});
                        if (swapToo) hits.push_back({s, u, {id2, id1}, 2});
                    });
                });
            }
            return cursor >= seeds.size();
        }

        // Lists the solutions found by the stream.
        void finishDepth() { listSolutions(hits, 4, goalLayer); }

        // One solution per line, moves separated by spaces.
        MUND std::string solutionsText(const bool zeroBased) const {
            std::string out;
            for (const auto& line : solutions) {
                State s = start;
                for (std::size_t i = 0; i < line.size(); ++i) {
                    if (i != 0) out += ' ';
                    out += moveName(s, line[i], zeroBased);
                    s = applyMove(s, line[i]);
                }
                out += '\n';
            }
            return out;
        }

    private:
        struct Hit {
            State s;      // in S_a
            State y;      // in G_b
            u8 mid[2];
            u8 midLen;
        };

        bool fat = false;
        State start{}, goal{};
        std::vector<StateSet> S, G;
        std::vector<Hit> hits;
        int depth = 0, middle = 0, goalLayer = 4;
        u64 cursor = 0;
        bool prune = false;
        bool laneTablesBuilt = false;
        std::vector<u8> rowTab, colTab;   // [lane][18-bit value]: fewest wrong cells over the 6 rotations

        static State makeState(const u8 cells[36]) {
            State s;
            for (u32 r = 0; r < 6; ++r) {
                for (u32 c = 0; c < 6; ++c) setCell(s, r, c, cells[6 * r + c]);
            }
            return s;
        }

        void ensureStart(const int k) {
            while (static_cast<int>(S.size()) <= k) {
                const std::size_t n = S.size();
                S.push_back(buildLayer(S[n - 1], n >= 2 ? &S[n - 2] : nullptr, fat, 0, 1, false));
            }
        }

        void ensureGoal(const int k) {
            while (static_cast<int>(G.size()) <= k) {
                const std::size_t n = G.size();
                G.push_back(buildLayer(G[n - 1], n >= 2 ? &G[n - 2] : nullptr, fat, 0, 1, false));
            }
        }

        static void intersect(const StateSet& a, const StateSet& b, std::vector<Hit>& out) {
            const bool aSmaller = a.size() <= b.size();
            const StateSet& small = aSmaller ? a : b;
            const StateSet& big = aSmaller ? b : a;
            for (const State& s : small.states) {
                if (big.contains(s)) out.push_back({s, s, {0, 0}, 0});
            }
        }

        // Rotation lane bound: worst row + worst column, each the fewest cells
        // wrong over the lane's 6 rotations. A move changes it by at most 1.
        void buildLaneTables() {
            rowTab.assign(6 * (LANE + 1), 0);
            colTab.assign(6 * (LANE + 1), 0);
            for (u32 lane = 0; lane < 12; ++lane) {
                const u64 g = lane < 6 ? readRow(goal, lane) : readCol(goal, lane - 6);
                u64 rots[6];
                for (u32 k = 0; k < 6; ++k) rots[k] = k == 0 ? g : rotateLane(g, k);
                u8* tab = lane < 6 ? &rowTab[lane * (LANE + 1)] : &colTab[(lane - 6) * (LANE + 1)];
                for (u64 v = 0; v <= LANE; ++v) {
                    int best = 6;
                    for (const u64 r : rots) {
                        const u64 d = v ^ r;
                        best = std::min(best, my_popcount((d | (d >> 1) | (d >> 2)) & 0x9249));
                    }
                    tab[v] = static_cast<u8>(best);
                }
            }
            laneTablesBuilt = true;
        }

        MUND FORCEINLINE int bound(const State& s) const {
            int rows = 0, cols = 0;
            for (u32 i = 0; i < 6; ++i) {
                rows = std::max<int>(rows, rowTab[i * (LANE + 1) + readRow(s, i)]);
                cols = std::max<int>(cols, colTab[i * (LANE + 1) + readCol(s, i)]);
            }
            return rows + cols;
        }

        // Every path start -> x of exactly k moves (x in S_k).
        void pathsFromStart(const State& x, const int k, std::vector<std::vector<u8>>& out) const {
            if (k == 0) {
                if (x == start) out.emplace_back();
                return;
            }
            forEachChild(x, fat, [&](const u8 id, const State& p) {
                if (!S[k - 1].contains(p)) return;
                const std::size_t first = out.size();
                pathsFromStart(p, k - 1, out);
                for (std::size_t i = first; i < out.size(); ++i) out[i].push_back(inverseMove(id));
            });
        }

        // Every path y -> goal of exactly k moves (y in G_k).
        void pathsToGoal(const State& y, const int k, std::vector<std::vector<u8>>& out) const {
            if (k == 0) {
                if (y == goal) out.emplace_back();
                return;
            }
            forEachChild(y, fat, [&](const u8 id, const State& z) {
                if (!G[k - 1].contains(z)) return;
                const std::size_t first = out.size();
                pathsToGoal(z, k - 1, out);
                for (std::size_t i = first; i < out.size(); ++i) out[i].insert(out[i].begin(), id);
            });
        }

        void listSolutions(const std::vector<Hit>& found, const int a, const int b) {
            solutionCount = 0;
            solutions.clear();
            invalidSolutions = 0;
            std::vector<std::vector<u8>> pre, suf;
            for (const Hit& h : found) {
                pre.clear();
                suf.clear();
                pathsFromStart(h.s, a, pre);
                pathsToGoal(h.y, b, suf);
                solutionCount += static_cast<u64>(pre.size()) * suf.size();
                for (const auto& p : pre) {
                    for (const auto& q : suf) {
                        if (solutions.size() >= MAX_LISTED) break;
                        std::vector<u8> line = p;
                        line.insert(line.end(), h.mid, h.mid + h.midLen);
                        line.insert(line.end(), q.begin(), q.end());
                        if (!replays(line)) {
                            ++invalidSolutions;
                            continue;
                        }
                        solutions.push_back(std::move(line));
                    }
                }
            }
        }

        MUND bool replays(const std::vector<u8>& line) const {
            State s = start;
            for (const u8 id : line) {
                if (!moveValid(s, fat, id)) return false;
                s = applyMove(s, id);
            }
            return s == goal;
        }
    };

} // namespace web_solver
