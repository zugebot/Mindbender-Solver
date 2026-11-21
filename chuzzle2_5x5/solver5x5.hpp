// solver5x5.hpp - the optimal 5x5 Mind Bender solver (Chuzzle 2), shared by the command-line
// tool (solve5x5.cpp) and the web page's WebAssembly build (wasm5x5.cpp). Standalone: nothing
// from src/.
//
// A move turns one row right or one column down by 1-4 cells (cyclic); shown 1-indexed as in
// the main project: R14 = row 1 right 4, C23 = column 2 down 3. A board is 25 cells of 4 bits
// (colours 0-9, 9 being the rainbow chuzzle) in one 128-bit number, cell y*5+x at bits 4(y*5+x).
//
// The search meets in the middle: layers grow alternately from the start and from the goal
// (the smaller side next), each state keeping its distance and the number of shortest move
// sequences reaching it. The first layer that touches the other side gives the optimal length
// D; every shortest solution passes exactly one state of that layer, so the number of
// solutions is the sum over those states of (sequences from the start) x (sequences to the
// goal), and the solutions themselves are walked out from them.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace mb5 {

using u8 = uint8_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

static constexpr int N = 5, CELLS = 25, MOVES = 40;   // 5 rows + 5 columns, 4 amounts each
static constexpr u128 ONE = 1;
static constexpr u128 ROW_MASK = (ONE << 20) - 1;

inline int cellOf(u128 b, int i) { return int((b >> (4 * i)) & 15); }

// move id: lane * 4 + (amount - 1); lanes 0-4 rows (right), 5-9 columns (down)
inline u128 applyMove(u128 b, int id) {
    const int lane = id / 4, k = id % 4 + 1;
    if (lane < N) {
        const int sh = 20 * lane;
        const u128 row = (b >> sh) & ROW_MASK;
        const u128 rot = ((row << (4 * k)) | (row >> (20 - 4 * k))) & ROW_MASK;
        return (b & ~(ROW_MASK << sh)) | (rot << sh);
    }
    const int x = lane - N;
    int col[N];
    for (int y = 0; y < N; ++y) col[y] = cellOf(b, y * N + x);
    for (int y = 0; y < N; ++y) {
        const int at = 4 * (((y + k) % N) * N + x);
        b = (b & ~(u128(15) << at)) | (u128(col[y]) << at);
    }
    return b;
}
inline int inverseMove(int id) { return (id / 4) * 4 + (N - (id % 4 + 1)) - 1; }
inline std::string moveName(int id) {
    const int lane = id / 4, k = id % 4 + 1;
    char s[8];
    std::snprintf(s, sizeof s, "%c%d%d", lane < N ? 'R' : 'C', (lane < N ? lane : lane - N) + 1, k);
    return s;
}

inline u128 fromCells(const u8* cells) {
    u128 b = 0;
    for (int i = 0; i < CELLS; ++i) b |= u128(cells[i] & 15) << (4 * i);
    return b;
}
inline bool sameColourCounts(u128 a, u128 b) {
    int n[16] = {};
    for (int i = 0; i < CELLS; ++i) ++n[cellOf(a, i)], --n[cellOf(b, i)];
    for (int v : n) if (v) return false;
    return true;
}

// Open addressing, 24 bytes a state: the board's low 64 bits; its high 36 bits with the
// distance in bits 36-43 and bit 63 marking a used slot; the count of shortest sequences.
class StateMap {
public:
    struct Slot {
        u64 lo, hi, count;
        u8 dist() const { return u8(hi >> 36); }
    };
    StateMap() { slots_.assign(1 << 12, Slot{0, 0, 0}); }
    Slot* find(u128 b) {
        const u64 lo = u64(b), hi = u64(b >> 64);
        for (u64 i = hash(lo, hi) & mask();; i = (i + 1) & mask()) {
            Slot& s = slots_[i];
            if (!s.hi) return nullptr;
            if (s.lo == lo && (s.hi & HI_BITS) == hi) return &s;
        }
    }
    void insert(u128 b, u8 dist, u64 count) {
        if (10 * (used_ + 1) > 7 * slots_.size()) grow();
        place(u64(b), u64(b >> 64), dist, count);
        ++used_;
    }
    size_t size() const { return used_; }
    size_t bytes() const { return slots_.size() * sizeof(Slot); }
private:
    static constexpr u64 HI_BITS = (u64(1) << 36) - 1, USED = u64(1) << 63;
    u64 mask() const { return slots_.size() - 1; }
    static u64 hash(u64 lo, u64 hi) {
        u64 h = lo * 0x9E3779B97F4A7C15ULL ^ hi * 0xC2B2AE3D27D4EB4FULL;
        return h ^ (h >> 29);
    }
    void place(u64 lo, u64 hi, u8 dist, u64 count) {
        u64 i = hash(lo, hi) & mask();
        while (slots_[i].hi) i = (i + 1) & mask();
        slots_[i] = Slot{lo, hi | (u64(dist) << 36) | USED, count};
    }
    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        slots_.assign(old.size() * 2, Slot{0, 0, 0});
        for (const Slot& s : old)
            if (s.hi) place(s.lo, s.hi & HI_BITS, s.dist(), s.count);
    }
    std::vector<Slot> slots_;
    size_t used_ = 0;
};

struct Side {
    StateMap map;
    std::vector<std::vector<u128>> layers;
    explicit Side(u128 root) { map.insert(root, 0, 1); layers.push_back({root}); }
    int depth() const { return int(layers.size()) - 1; }
    // the next layer: children not seen before, each with the summed counts of its parents
    void grow() {
        const u8 d = u8(layers.size());
        std::vector<u128> next;
        for (u128 p : layers.back()) {
            const u64 c = map.find(p)->count;
            for (int m = 0; m < MOVES; ++m) {
                const u128 ch = applyMove(p, m);
                if (StateMap::Slot* s = map.find(ch)) {
                    if (s->dist() == d) s->count += c;
                } else {
                    map.insert(ch, d, c);
                    next.push_back(ch);
                }
            }
        }
        layers.push_back(std::move(next));
    }
};

struct Result {
    int length = -1;          // -1: none within maxDepth
    u64 solutions = 0;
    std::vector<std::vector<int>> listed;
    size_t states = 0, bytes = 0;
};

// Walks out up to `limit` shortest solutions through the meeting states.
inline void listSolutions(Side& from, Side& to, const std::vector<u128>& meet, size_t limit,
                          std::vector<std::vector<int>>& out) {
    std::vector<std::vector<int>> prefixes, suffixes;
    // prefixes: from the meeting state back to the start (the moves reversed at the end)
    auto back = [&](auto&& self, u128 b, std::vector<int>& acc) -> void {
        if (prefixes.size() >= limit) return;
        const int d = from.map.find(b)->dist();
        if (d == 0) { prefixes.emplace_back(acc.rbegin(), acc.rend()); return; }
        for (int m = 0; m < MOVES; ++m) {
            const u128 n = applyMove(b, m);
            const StateMap::Slot* s = from.map.find(n);
            if (s && s->dist() == d - 1) {
                acc.push_back(inverseMove(m));
                self(self, n, acc);
                acc.pop_back();
            }
        }
    };
    auto fwd = [&](auto&& self, u128 b, std::vector<int>& acc) -> void {
        if (suffixes.size() >= limit) return;
        const int d = to.map.find(b)->dist();
        if (d == 0) { suffixes.push_back(acc); return; }
        for (int m = 0; m < MOVES; ++m) {
            const u128 n = applyMove(b, m);
            const StateMap::Slot* s = to.map.find(n);
            if (s && s->dist() == d - 1) {
                acc.push_back(m);
                self(self, n, acc);
                acc.pop_back();
            }
        }
    };
    for (u128 m : meet) {
        if (out.size() >= limit) return;
        prefixes.clear();
        suffixes.clear();
        std::vector<int> acc;
        back(back, m, acc);
        acc.clear();
        fwd(fwd, m, acc);
        for (auto& p : prefixes)
            for (auto& s : suffixes) {
                if (out.size() >= limit) return;
                std::vector<int> all = p;
                all.insert(all.end(), s.begin(), s.end());
                out.push_back(std::move(all));
            }
    }
}

inline Result solve(u128 start, u128 goal, int maxDepth, size_t listLimit) {
    Result r;
    if (start == goal) { r.length = 0; r.solutions = 1; r.listed.push_back({}); return r; }
    Side S(start), G(goal);
    while (S.depth() + G.depth() < maxDepth) {
        const bool growStart = S.layers.back().size() <= G.layers.back().size();
        Side& grown = growStart ? S : G;
        Side& other = growStart ? G : S;
        grown.grow();
        int best = 1 << 30;
        for (u128 b : grown.layers.back())
            if (const StateMap::Slot* s = other.map.find(b)) best = std::min(best, int(s->dist()));
        if (best == 1 << 30) continue;
        std::vector<u128> meet;
        for (u128 b : grown.layers.back()) {
            const StateMap::Slot* s = other.map.find(b);
            if (s && s->dist() == best) {
                meet.push_back(b);
                r.solutions += grown.map.find(b)->count * s->count;
            }
        }
        r.length = grown.depth() + best;
        r.states = S.map.size() + G.map.size();
        r.bytes = S.map.bytes() + G.map.bytes();
        listSolutions(S, G, meet, listLimit, r.listed);
        return r;
    }
    r.states = S.map.size() + G.map.size();
    r.bytes = S.map.bytes() + G.map.bytes();
    return r;
}

// The check: one side only, from the start until the goal's layer.
inline Result solveOneSided(u128 start, u128 goal, int maxDepth) {
    Result r;
    Side S(start);
    for (;;) {
        if (const StateMap::Slot* s = S.map.find(goal)) {
            r.length = s->dist();
            r.solutions = s->count;
            break;
        }
        if (S.depth() >= maxDepth) break;
        S.grow();
    }
    r.states = S.map.size();
    return r;
}

inline bool replay(u128 start, u128 goal, const std::vector<int>& moves) {
    for (int m : moves) start = applyMove(start, m);
    return start == goal;
}

}  // namespace mb5
