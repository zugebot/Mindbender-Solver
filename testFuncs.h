#pragma once
// solver_finish.cpp
#include "code/include.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// ========================= new/delete counters =========================
static long long new_calls = 0;
static long double sort_time = 0.0;
static long double total_time = 0.0;

void* operator new(const std::size_t size) {
    ++new_calls;
    if (void* p = std::malloc(size)) return p;
    throw std::bad_alloc();
}
void* operator new[](const std::size_t size) {
    ++new_calls;
    if (void* p = std::malloc(size)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }

// ========================= Basic helpers =========================

static inline bool isRowIdx(const int idx) { return idx >= 0 && idx < 30; }
static inline bool isColIdx(const int idx) { return idx >= 30 && idx < 60; }

static inline int rowIdx(const int row, int shift /*1..5*/) { return row * 5 + (shift - 1); }
static inline int colIdx(const int col, int shift /*1..5*/) { return 30 + col * 5 + (shift - 1); }

static inline int shiftOf(int idx) {
    int off = isRowIdx(idx) ? idx : (idx - 30);
    return (off % 5) + 1; // 1..5
}
static inline int lineOf(int idx) {
    return isRowIdx(idx) ? (idx / 5)                 // 0..5 rows
                         : (6 + ((idx - 30) / 5));   // 6..11 cols
}
static inline int inverseShift(const int s) { return (s == 3) ? 3 : (6 - s); }
static inline int inverseIdx(const int idx) {
    if (idx < 0) return -1;
    if (isRowIdx(idx)) {
        const int r = idx / 5;
        return rowIdx(r, inverseShift(shiftOf(idx)));
    } else {
        const int c = (idx - 30) / 5;
        return colIdx(c, inverseShift(shiftOf(idx)));
    }
}

static inline void applyMove(Board& b, int idx) {
    applyMove(b, idx);
    // allActStructList[idx]. action(reinterpret_cast<B1B2&>(b));
}
template <class Seq>
static inline void applyMoves(Board& b, const Seq& seq, int n) {
    for (int i = 0; i < n; ++i) applyMove(b, seq[i]);
}
static inline int matchCount(const Board& a, const Board& goal) { return a.getScore1(goal); }
static inline bool isSolved(const Board& a, const Board& goal) { return a == goal; }

static inline std::string moveName(const int idx) {
    const auto& n = allActStructList[idx].name; // up to 4 chars
    std::string s; s.reserve(4);
    for (int i = 0; i < 4 && n[i] != '\0'; ++i) s.push_back(n[i]);
    return s;
}
static void print_path(const std::vector<int>& path) {
    tcout << "Moves (" << path.size() << "):";
    for (size_t i = 0; i < path.size(); ++i) {
        tcout << (i ? " " : " ") << moveName(path[i]);
    }
    tcout << "\n";
}

// ========================= Inline path (no heap) =========================

template<typename T, int CAP>
struct SmallPath {
    static_assert(std::is_unsigned_v<T>, "SmallPath T must be unsigned");
    static_assert(CAP > 0, "SmallPath CAP must be > 0");

    uint8_t len = 0;
    T data[CAP];

    inline void push(T m) {
        data[len++] = m;
    }

    inline T operator[](uint8_t i) const {
        return data[i];
    }

    inline void copy_from(const SmallPath& other) {
        len = other.len;
        if (len) std::memcpy(data, other.data, len * sizeof(T));
    }

    template<typename U>
    inline void append_to(std::vector<U>& out) const {
        out.reserve(out.size() + len);
        for (uint8_t i = 0; i < len; ++i) out.push_back(static_cast<U>(data[i]));
    }
};

constexpr int BEAM_MAX_DEPTH  = 18;
constexpr int MICRO_MAX_DEPTH = 12;
constexpr int PATH_CAP        = 24;

using Path32 = SmallPath<uint8_t, PATH_CAP>;

// ========================= Beam search (Phase B) =========================

struct BeamCfg {
    int WIDTH = 64;
    int MAX_DEPTH = 12;
    bool forbidSameLineTwice = true;
};

struct BeamNode {
    Board  b;
    Path32 path;          // no heap
    int8_t  last = -1;
    uint8_t g    = 0;
    uint8_t matches = 0;
};

struct BeamResult {
    Board bestBoard;
    bool solved = false;
    std::vector<uint8_t> bestPath; // output as std::vector (printed at end)
};

static BeamResult run_beam(const Board& start, const Board& goal, const BeamCfg& cfg) {
    tcout.setf(std::ios::unitbuf);

    BeamNode root;
    root.b = start;
    root.matches = matchCount(start, goal);

    BeamResult R{root.b, false, {}};
    if (root.matches == 36) { R.solved = true; return R; }

    std::vector<BeamNode> beam; beam.reserve(cfg.WIDTH);
    beam.push_back(root);

    int globalBest = root.matches;
    const Timer t0;

    for (int d = 1; d <= cfg.MAX_DEPTH; ++d) {
        std::vector<BeamNode> next;

        // Reserve generously: we’ll generate a lot of children then trim
        next.reserve(static_cast<size_t>(cfg.WIDTH) * 20);

        for (const auto& nd : beam) {
            for (int idx = 0; idx < 60; ++idx) {
                if (idx == inverseIdx(nd.last)) continue;
                if (cfg.forbidSameLineTwice && nd.last >= 0 && lineOf(idx) == lineOf(nd.last)) continue;

                BeamNode ch;
                ch.b = nd.b;                 // one Board copy per child (same as before)
                applyMove(ch.b, idx);
                ch.last = idx;
                ch.g    = nd.g + 1;
                ch.matches = matchCount(ch.b, goal);
                ch.path.copy_from(nd.path);
                ch.path.push(idx);

                if (ch.matches > globalBest) {
                    globalBest = ch.matches;
                    R.bestBoard = ch.b;
                    R.bestPath.clear();
                    R.bestPath.insert(R.bestPath.end(), ch.path.data, ch.path.data + ch.path.len);
                }
                if EXPECT_FALSE(ch.matches == 36) {
                    tcout << "[beam] depth " << d
                              << " | beam=? | topMatches=36 | bestEver=36 | elapsed="
                              << t0.getSeconds() << "s\n";
                    R.solved = true;
                    R.bestBoard = ch.b;
                    R.bestPath.clear();
                    R.bestPath.insert(R.bestPath.end(), ch.path.data, ch.path.data + ch.path.len);
                    return R;
                }
                next.push_back(std::move(ch));
            }
        }

        Timer sort_timer;
        std::ranges::partial_sort(next,
            next.size() > (size_t) cfg.WIDTH ? next.begin() + cfg.WIDTH : next.end(),
            [](const BeamNode& A, const BeamNode& B) {
                if (A.matches != B.matches) return A.matches > B.matches;
                return A.g < B.g;
            }
        );
        sort_time += sort_timer.getSeconds();

        if ((int)next.size() > cfg.WIDTH) next.resize(cfg.WIDTH);

        const int top = next.empty() ? -1 : next.front().matches;
        tcout << "[beam] depth " << d
                  << " | beam=" << (int)next.size()
                  << " | topMatches=" << top
                  << " | bestEver=" << globalBest
                  << " | elapsed=" << t0.getSeconds() << "s\n";

        if (next.empty()) break;
        beam.swap(next);
    }
    return R;
}

// ========================= Active-line helpers =========================

static inline void active_lines(const Board& b, const Board& goal,
                                bool activeRow[6], bool activeCol[6]) {
    for (int i = 0; i < 6; ++i) activeRow[i] = activeCol[i] = false;
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
            if (b.getColor(x, y) != goal.getColor(x, y)) {
                activeRow[y] = true;
                activeCol[x] = true;
            }
}

static inline int build_allowed_indices(const bool activeRow[6], const bool activeCol[6],
                                        int out[60]) {
    int n = 0;
    for (int r = 0; r < 6; ++r)
        if (activeRow[r])
            for (int s = 1; s <= 5; ++s) out[n++] = rowIdx(r, s);
    for (int c = 0; c < 6; ++c)
        if (activeCol[c])
            for (int s = 1; s <= 5; ++s) out[n++] = colIdx(c, s);
    return n;
}

// ========================= Mismatch utilities =========================

struct Cell { int x, y; };

static inline void collect_mismatches(const Board& b, const Board& goal, std::vector<Cell>& cells) {
    cells.clear(); cells.reserve(36);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
            if (b.getColor(x, y) != goal.getColor(x, y))
                cells.push_back({x, y});
}
static inline void rows_cols_from_cells(const std::vector<Cell>& v,
                                        std::vector<int>& rows, std::vector<int>& cols) {
    std::array<bool,6> r{}; std::array<bool,6> c{};
    for (const auto &p : v) { r[p.y] = true; c[p.x] = true; }
    rows.clear(); cols.clear();
    for (int i = 0; i < 6; ++i) { if (r[i]) rows.push_back(i); if (c[i]) cols.push_back(i); }
}

// ========================= Commutator utilities (fixed arrays) =========================

struct Cand {
    int    seq[8];
    uint8_t len = 0;
    Board  b;
    int    matches = 0;
};

static inline void push4(int* dst, int a, int b, int c, int d) {
    dst[0] = a; dst[1] = b; dst[2] = c; dst[3] = d;
}
static inline void test_comm_pair(const Board& start, const Board& goal,
                                  const int r, const int sr, const int c, const int sc,
                                  std::vector<Cand>& out) {
    const int A = rowIdx(r, sr),  AInv = inverseIdx(A);
    const int B = colIdx(c, sc),  BInv = inverseIdx(B);

    // [A,B]
    {
        Cand k; k.len = 4; push4(k.seq, A, B, AInv, BInv); k.b = start;
        applyMoves(k.b, k.seq, k.len);
        k.matches = matchCount(k.b, goal);
        out.push_back(std::move(k)); // move
    }
    // [B,A]
    {
        Cand k; k.len = 4; push4(k.seq, B, A, BInv, AInv); k.b = start;
        applyMoves(k.b, k.seq, k.len);
        k.matches = matchCount(k.b, goal);
        out.push_back(std::move(k)); // move
    }
}

static inline void enumerate_commutators(const Board& cur, const Board& goal,
                                         const std::vector<int>& rows,
                                         const std::vector<int>& cols,
                                         std::vector<Cand>& out) {
    out.clear();
    out.reserve(1800); // 6*6*5*5*2 worst case

    const bool useAllRows = rows.empty();
    const bool useAllCols = cols.empty();

    if (useAllRows) {
        for (int r = 0; r < 6; ++r) {
            if (useAllCols) {
                for (int c = 0; c < 6; ++c) {
                    for (int sr = 1; sr <= 5; ++sr)
                        for (int sc = 1; sc <= 5; ++sc)
                            test_comm_pair(cur, goal, r, sr, c, sc, out);
                }
            } else {
                for (const int c : cols) {
                    for (int sr = 1; sr <= 5; ++sr)
                        for (int sc = 1; sc <= 5; ++sc)
                            test_comm_pair(cur, goal, r, sr, c, sc, out);
                }
            }
        }
    } else {
        for (const int r : rows) {
            if (useAllCols) {
                for (int c = 0; c < 6; ++c) {
                    for (int sr = 1; sr <= 5; ++sr)
                        for (int sc = 1; sc <= 5; ++sc)
                            test_comm_pair(cur, goal, r, sr, c, sc, out);
                }
            } else {
                for (const int c : cols) {
                    for (int sr = 1; sr <= 5; ++sr)
                        for (int sc = 1; sc <= 5; ++sc)
                            test_comm_pair(cur, goal, r, sr, c, sc, out);
                }
            }
        }
    }
}

// ========================= MICRO-BEAM (active-line only) =========================

struct MicroBeamCfg {
    int WIDTH = 8192;
    int MAX_DEPTH = 8;
    bool forbidSameLineTwice = true;
};
struct MicroRes {
    bool solved = false;
    std::vector<int> path;
    Board board;
};

static MicroRes run_micro_beam_active(const Board& start, const Board& goal, const MicroBeamCfg& cfg) {
    tcout.setf(std::ios::unitbuf);

    struct N {
        Board  b;
        int    last = -1;
        int    g = 0;
        int    matches = 0;
        Path32 path;
    };

    N root; root.b = start; root.matches = matchCount(start, goal);

    MicroRes R{false, {}, start};
    if (root.matches == 36) { R.solved = true; return R; }

    std::vector<N> beam; beam.reserve(cfg.WIDTH);
    beam.push_back(root);

    int globalBest = root.matches;
    Timer t0;

    for (int depth = 1; depth <= cfg.MAX_DEPTH; ++depth) {
        std::vector<N> next; next.reserve(cfg.WIDTH * 8);

        for (const N& nd : beam) {
            bool aRow[6], aCol[6]; active_lines(nd.b, goal, aRow, aCol);
            int allowed[60]; const int an = build_allowed_indices(aRow, aCol, allowed);

            if (an == 0) {
                R.solved = true; R.board = nd.b;
                R.path.assign(nd.path.data, nd.path.data + nd.path.len);
                tcout << "[micro] depth " << (depth-1) << " solved\n";
                return R;
            }

            for (int i = 0; i < an; ++i) {
                const int idx = allowed[i];
                if (idx == inverseIdx(nd.last)) continue;
                if (cfg.forbidSameLineTwice && nd.last >= 0 && lineOf(idx) == lineOf(nd.last)) continue;

                N ch;
                ch.b = nd.b;
                applyMove(ch.b, idx);
                ch.last = idx;
                ch.g    = nd.g + 1;
                ch.matches = matchCount(ch.b, goal);
                ch.path.copy_from(nd.path);
                ch.path.push(idx);

                if (ch.matches > globalBest) {
                    globalBest = ch.matches;
                    R.board = ch.b;
                    R.path.assign(ch.path.data, ch.path.data + ch.path.len);
                }
                if (ch.matches == 36) {
                    R.solved = true; R.board = ch.b;
                    R.path.assign(ch.path.data, ch.path.data + ch.path.len);
                    tcout << "[micro] depth " << depth
                              << " | beam=? | topMatches=36 | bestEver=36 | elapsed="
                              << t0.getSeconds() << "s\n";
                    return R;
                }
                next.push_back(std::move(ch)); // move
            }
        }

        Timer sort_timer;
        std::partial_sort(
            next.begin(),
            next.size() > (size_t)cfg.WIDTH ? next.begin() + cfg.WIDTH : next.end(),
            next.end(),
            [](const N& A, const N& B) {
                if (A.matches != B.matches) return A.matches > B.matches;
                return A.g < B.g;
            }
        );
        sort_time += sort_timer.getSeconds();
        if ((int)next.size() > cfg.WIDTH) next.resize(cfg.WIDTH);

        int top = next.empty() ? -1 : next.front().matches;
        tcout << "[micro] depth " << depth
                  << " | beam=" << (int)next.size()
                  << " | topMatches=" << top
                  << " | bestEver=" << globalBest
                  << " | elapsed=" << t0.getSeconds() << "s\n";

        if (next.empty()) break;
        beam.swap(next);
    }
    return R;
}

// ========================= Exact finishers (32/36, 33/36) =========================

static bool finish_32_exact(Board& cur, const Board& goal, std::vector<int>& outMoves) {
    std::vector<Cell> mm; collect_mismatches(cur, goal, mm);
    if ((int)mm.size() != 4) return false;

    std::vector<int> rows, cols; rows_cols_from_cells(mm, rows, cols);
    if (rows.size() > 2 || cols.size() > 2) return false;

    std::vector<Cand> L1; enumerate_commutators(cur, goal, rows, cols, L1);
    for (auto& c : L1) {
        if (c.matches == 36) {
            applyMoves(cur, c.seq, c.len);
            outMoves.insert(outMoves.end(), c.seq, c.seq + c.len);
            tcout << "[t32] 1 commutator\n";
            return true;
        }
    }
    for (auto& c1 : L1) {
        std::vector<Cand> L2; enumerate_commutators(c1.b, goal, rows, cols, L2);
        for (auto& c2 : L2) {
            if (c2.matches == 36) {
                int seq[8]; int n = 0;
                for (int i = 0; i < c1.len; ++i) seq[n++] = c1.seq[i];
                for (int i = 0; i < c2.len; ++i) seq[n++] = c2.seq[i];
                applyMoves(cur, seq, n);
                outMoves.insert(outMoves.end(), seq, seq + n);
                tcout << "[t32] 2 commutators\n";
                return true;
            }
        }
    }
    return false;
}
static bool finish_33_exact(Board& cur, const Board& goal, std::vector<int>& outMoves) {
    std::vector<Cell> mm; collect_mismatches(cur, goal, mm);
    if ((int)mm.size() != 3) return false;

    std::vector<int> rows, cols; rows_cols_from_cells(mm, rows, cols);

    std::vector<Cand> L1; enumerate_commutators(cur, goal, rows, cols, L1);
    for (auto& c : L1) {
        if (c.matches == 36) {
            applyMoves(cur, c.seq, c.len);
            outMoves.insert(outMoves.end(), c.seq, c.seq + c.len);
            tcout << "[t33] 1 commutator\n";
            return true;
        }
    }
    for (auto& c1 : L1) {
        std::vector<Cand> L2; enumerate_commutators(c1.b, goal, rows, cols, L2);
        for (auto& c2 : L2) {
            if (c2.matches == 36) {
                int seq[8]; int n = 0;
                for (int i = 0; i < c1.len; ++i) seq[n++] = c1.seq[i];
                for (int i = 0; i < c2.len; ++i) seq[n++] = c2.seq[i];
                applyMoves(cur, seq, n);
                outMoves.insert(outMoves.end(), seq, seq + n);
                tcout << "[t33] 2 commutators\n";
                return true;
            }
        }
    }
    return false;
}

// ========================= Hill-climb + local beam (finisher) =========================

struct FinishCfg {
    int WIDTH = 48;
    int MAX_DEPTH = 12;
    bool avoidSameLineTwice = false;
};

static bool improve_by_one_comm(Board& cur, const Board& goal, std::vector<int>& outMoves) {
    bool aRow[6], aCol[6]; active_lines(cur, goal, aRow, aCol);

    std::vector<int> rows, cols;
    for (int i = 0; i < 6; ++i) if (aRow[i]) rows.push_back(i);
    for (int i = 0; i < 6; ++i) if (aCol[i]) cols.push_back(i);

    const int base = matchCount(cur, goal);

    std::vector<Cand> cand; enumerate_commutators(cur, goal, rows, cols, cand);
    auto better = std::ranges::max_element(cand,
        [](const Cand& A, const Cand& B){ return A.matches < B.matches; });

    if (better != cand.end() && better->matches > base) {
        applyMoves(cur, better->seq, better->len);
        outMoves.insert(outMoves.end(), better->seq, better->seq + better->len);
        return true;
    }

    enumerate_commutators(cur, goal, {}, {}, cand);
    better = std::ranges::max_element(cand,
        [](const Cand& A, const Cand& B){ return A.matches < B.matches; });
    if (better != cand.end() && better->matches > base) {
        applyMoves(cur, better->seq, better->len);
        outMoves.insert(outMoves.end(), better->seq, better->seq + better->len);
        return true;
    }
    return false;
}

struct FNode {
    Board  b;
    int8_t  last = -1;
    uint8_t g = 0;
    uint8_t matches = 0;
    Path32 path;
};

static bool finish_local_beam(const Board& start, const Board& goal,
                              const FinishCfg& cfg, std::vector<int>& outMoves) {
    tcout.setf(std::ios::unitbuf);
    if (isSolved(start, goal)) return true;

    std::vector<FNode> beam; beam.reserve(cfg.WIDTH);
    FNode root; root.b = start; root.matches = matchCount(start, goal);
    beam.push_back(root);

    int globalBest = root.matches;
    const Timer t0;

    for (int depth = 1; depth <= cfg.MAX_DEPTH; ++depth) {
        std::vector<FNode> next; next.reserve(cfg.WIDTH * 8);

        for (const FNode& nd : beam) {
            bool aRow[6], aCol[6]; active_lines(nd.b, goal, aRow, aCol);
            bool any = false; for (int i = 0; i < 6; ++i) any |= aRow[i] | aCol[i];
            if (!any) {
                nd.path.append_to(outMoves);
                tcout << "[finish] depth=" << depth-1 << " solved\n";
                return true;
            }

            int allowed[60];
            const int an = build_allowed_indices(aRow, aCol, allowed);
            for (int k = 0; k < an; ++k) {
                const int idx = allowed[k];
                if (idx == inverseIdx(nd.last)) continue;
                if (cfg.avoidSameLineTwice && nd.last >= 0 && lineOf(idx) == lineOf(nd.last)) continue;

                FNode ch;
                ch.b = nd.b;
                applyMove(ch.b, idx);
                ch.last = idx;
                ch.g    = nd.g + 1;
                ch.matches = matchCount(ch.b, goal);
                ch.path.copy_from(nd.path);
                ch.path.push(idx);

                if (ch.matches > globalBest) globalBest = ch.matches;
                if (ch.matches == 36) {
                    ch.path.append_to(outMoves);
                    tcout << "[finish] depth=" << depth << " solved\n";
                    return true;
                }
                next.push_back(std::move(ch)); // move
            }
        }

        Timer sort_timer;
        std::partial_sort(
            next.begin(),
            next.size() > (size_t)cfg.WIDTH ? next.begin() + cfg.WIDTH : next.end(),
            next.end(),
            [](const FNode& A, const FNode& B) {
                if (A.matches != B.matches) return A.matches > B.matches;
                return A.g < B.g;
            }
        );
        sort_time += sort_timer.getSeconds();
        if ((int)next.size() > cfg.WIDTH) next.resize(cfg.WIDTH);

        int top = next.empty() ? -1 : next.front().matches;
        tcout << "[finish] depth " << depth
                  << " | beam=" << (int)next.size()
                  << " | topMatches=" << top
                  << " | bestEver=" << globalBest
                  << " | elapsed=" << t0.getSeconds() << "s\n";

        if (next.empty()) {
            tcout << "[finish] frontier empty\n";
            return false;
        }
        beam.swap(next);
    }
    tcout << "[finish] depth cap hit\n";
    return false;
}

// ========================= Orchestrated finisher =========================

static bool finish_with_comm_then_beam(Board& cur, const Board& goal,
                                       const FinishCfg& cfg, std::vector<int>& outMoves) {
    auto try_micro = [&](const int mNow)->bool {
        if (mNow == 33 || mNow == 32) {
            MicroBeamCfg mcfg;
            if (mNow == 33) { mcfg.WIDTH = 8192;  mcfg.MAX_DEPTH = 8; }
            else            { mcfg.WIDTH = 16384; mcfg.MAX_DEPTH = 10; }
            mcfg.forbidSameLineTwice = true;
            const auto mr = run_micro_beam_active(cur, goal, mcfg);
            if (mr.solved) {
                applyMoves(cur, mr.path, (int)mr.path.size());
                outMoves.insert(outMoves.end(), mr.path.begin(), mr.path.end());
                tcout << "[polish] micro-beam solved from " << mNow << "/36\n";
                return true;
            }
            int before = matchCount(cur, goal);
            if (matchCount(mr.board, goal) > before) {
                applyMoves(cur, mr.path, (int)mr.path.size());
                outMoves.insert(outMoves.end(), mr.path.begin(), mr.path.end());
                tcout << "[polish] micro-beam improved " << before << " -> "
                          << matchCount(cur, goal) << "\n";
                if (matchCount(cur, goal) == 36) return true;
            }
        }
        return false;
    };

    { int m = matchCount(cur, goal); if (try_micro(m)) return true; }

    { int m = matchCount(cur, goal);
      if (m == 32 && finish_32_exact(cur, goal, outMoves)) return true;
      if (m == 33 && finish_33_exact(cur, goal, outMoves)) return true; }

    const int MAX_COMM_ITERS = 6;
    for (int it = 0; it < MAX_COMM_ITERS; ++it) {
        if (isSolved(cur, goal)) return true;
        int before = matchCount(cur, goal);
        if (!improve_by_one_comm(cur, goal, outMoves)) break;
        int after = matchCount(cur, goal);
        tcout << "[comm] iter " << it+1 << " " << before << " -> " << after << "\n";
        if (after == 36) return true;
        if (after <= before) break;
        if (try_micro(after)) return true;
    }

    { int m = matchCount(cur, goal); if (try_micro(m)) return true; }

    std::vector<int> tail;
    if (finish_local_beam(cur, goal, cfg, tail)) {
        outMoves.insert(outMoves.end(), tail.begin(), tail.end());
        return true;
    }
    return false;
}

// ========================= Main =========================

static inline void end_report() {
    tcout << "New Calls : " << new_calls << "\n";
    tcout << "Time Sort : " << sort_time << "\n";
    tcout << "Time Total: " << total_time << "\n";
}

int mainTestFuncs(int argc, char** argv) {
    tcout.setf(std::ios::unitbuf);
    const char* id = (argc >= 2 ? argv[1] : "10-3");
    const Timer total_timer;

    const auto pair = BoardLookup::getBoardPair(id);
    tcout << pair->toString() << std::endl;
    Board start = pair->getStartState();
    const Board goal  = pair->getEndState();

    std::vector<int> totalPath;
    const int best = matchCount(start, goal);
    tcout << "[init] puzzle=" << id << " matches=" << best << "/36\n";
    if (best == 36) { tcout << "Already solved.\n"; end_report(); return 0; }

    // Phase B: main beam
    BeamResult br;
    {
        BeamCfg bcfg;
        bcfg.WIDTH = (1 << 16) * 3;  // 196,608
        bcfg.MAX_DEPTH = 12;
        bcfg.forbidSameLineTwice = true;

        br = run_beam(start, goal, bcfg);
        if (br.solved) {
            for (int m : br.bestPath) { applyMove(start, m); totalPath.push_back(m); }
            total_time = total_timer.getSeconds();
            tcout << "Solved in beam.\n";
            print_path(totalPath);
            return 0;
        }
        for (int m : br.bestPath) { applyMove(start, m); totalPath.push_back(m); }
        tcout << "[handoff] matches=" << matchCount(start, goal) << "/36\n";
    }

    // Phase C: finisher
    {
        FinishCfg fcfg; fcfg.WIDTH = 48; fcfg.MAX_DEPTH = 12; fcfg.avoidSameLineTwice = false;
        std::vector<int> tidy;
        if (finish_with_comm_then_beam(start, goal, fcfg, tidy)) {
            for (int m : tidy) { applyMove(start, m); totalPath.push_back(m); }
            total_time = total_timer.getSeconds();
            tcout << "Solved in finisher.\n";
            print_path(totalPath);
            end_report();
            return 0;
        } else {
            total_time = total_timer.getSeconds();
            tcout << "No exact solution within tidy-up budget.\n";
            print_path(totalPath);
            end_report();
            return 1;
        }
    }

    total_time = total_timer.getSeconds();
    tcout << "Reached end of main.\n";
    print_path(totalPath);
    end_report();
    return 0;
}
