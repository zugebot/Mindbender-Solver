#pragma once

#include "code/include.hpp"
#include "code/solver/frontier_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace heur {

    inline constexpr std::size_t kHeuristicCount = 12;

    inline constexpr std::array<const char*, kHeuristicCount> kHeuristicNames = {
            "transport_chebyshev",
            "transport_manhattan",
            "rowcol_imbalance",
            "edge_disagreement",
            "misplaced_cells",
            "color_parity_imbalance",
            "quadrant_color_imbalance",
            "rowcol_signature_matching",
            "sinkhorn_entropic_ot",
            "mincostflow_move_aware",
            "cycle_consistency_penalty",
            "mi_entropy_signature_mismatch",
    };

    inline constexpr std::array<double, kHeuristicCount> kBlendWeights = {
            1.2,  // transport_chebyshev
            1.5,  // transport_manhattan
            1.7,  // rowcol_imbalance
            0.7,  // edge_disagreement
            1.0,  // misplaced_cells
            0.6,  // color_parity_imbalance
            0.8,  // quadrant_color_imbalance
            0.7,  // rowcol_signature_matching
            1.0,  // sinkhorn_entropic_ot
            1.3,  // mincostflow_move_aware
            0.6,  // cycle_consistency_penalty
            0.7   // mi_entropy_signature_mismatch
    };

    struct Pos {
        int x = 0;
        int y = 0;
    };

    struct HungarianResult {
        int total_cost = 0;
        std::vector<int> match_row_to_col;
    };

    inline int torusDelta(const int a, const int b) {
        const int d = std::abs(a - b);
        return std::min(d, 6 - d);
    }

    inline int cellTransportCost(const Pos& from, const Pos& to, const bool manhattan) {
        const int dx = torusDelta(from.x, to.x);
        const int dy = torusDelta(from.y, to.y);
        return manhattan ? (dx + dy) : std::max(dx, dy);
    }

    inline HungarianResult hungarianSolve(const std::vector<std::vector<int>>& cost) {
        const int n = static_cast<int>(cost.size());
        if (n == 0) {
            return {};
        }

        const int inf = 1000000000;
        std::vector<int> u(n + 1, 0), v(n + 1, 0), p(n + 1, 0), way(n + 1, 0);

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<int> minv(n + 1, inf);
            std::vector<char> used(n + 1, false);
            do {
                used[j0] = true;
                const int i0 = p[j0];
                int delta = inf;
                int j1 = 0;
                for (int j = 1; j <= n; ++j) {
                    if (used[j]) {
                        continue;
                    }
                    const int cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
                for (int j = 0; j <= n; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do {
                const int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        std::vector<int> matchRowToCol(n + 1, 0);
        for (int j = 1; j <= n; ++j) {
            matchRowToCol[p[j]] = j;
        }

        int total = 0;
        std::vector<int> match_row_to_col_0(static_cast<std::size_t>(n), 0);
        for (size_t i = 1; i <= static_cast<size_t>(n); ++i) {
            const int col0 = matchRowToCol[i] - 1;
            match_row_to_col_0[i - 1] = col0;
            total += cost[i - 1][col0];
        }

        HungarianResult out;
        out.total_cost = total;
        out.match_row_to_col = std::move(match_row_to_col_0);
        return out;
    }

    inline int hungarianMinCost(const std::vector<std::vector<int>>& cost) {
        return hungarianSolve(cost).total_cost;
    }

    inline int heuristicColorTransportMatching(const Board& state, const Board& goal, const bool manhattan) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int total = 0;

        for (int color = 1; color <= maxColor; ++color) {
            std::vector<Pos> s;
            std::vector<Pos> g;

            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    if (state.getColor(x, y) == color) {
                        s.push_back({int(x), int(y)});
                    }
                    if (goal.getColor(x, y) == color) {
                        g.push_back({int(x), int(y)});
                    }
                }
            }

            if (s.size() != g.size()) {
                return 1000000000;
            }
            if (s.empty()) {
                continue;
            }

            const size_t n = s.size();
            std::vector<std::vector<int>> cost(n, std::vector<int>(n, 0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    cost[i][j] = cellTransportCost(s[i], g[j], manhattan);
                }
            }

            total += hungarianMinCost(cost);
        }

        return total;
    }

    inline int heuristicRowColImbalance(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int total = 0;

        for (int color = 1; color <= maxColor; ++color) {
            std::array<int, 6> sRow{};
            std::array<int, 6> gRow{};
            std::array<int, 6> sCol{};
            std::array<int, 6> gCol{};

            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    const int sColor = state.getColor(x, y);
                    const int gColor = goal.getColor(x, y);
                    if (sColor == color) {
                        ++sRow[y];
                        ++sCol[x];
                    }
                    if (gColor == color) {
                        ++gRow[y];
                        ++gCol[x];
                    }
                }
            }

            for (size_t i = 0; i < 6; ++i) {
                total += std::abs(sRow[i] - gRow[i]);
                total += std::abs(sCol[i] - gCol[i]);
            }
        }

        return total / 2;
    }

    inline int heuristicEdgeDisagreement(const Board& state, const Board& goal) {
        int bad = 0;

        for (size_t x = 0; x < 6; ++x) {
            for (size_t y = 0; y < 6; ++y) {
                const size_t xr = (x + 1) % 6;
                const size_t yd = (y + 1) % 6;

                const bool sEqR = state.getColor(x, y) == state.getColor(xr, y);
                const bool gEqR = goal.getColor(x, y) == goal.getColor(xr, y);
                if (sEqR != gEqR) {
                    ++bad;
                }

                const bool sEqD = state.getColor(x, y) == state.getColor(x, yd);
                const bool gEqD = goal.getColor(x, y) == goal.getColor(x, yd);
                if (sEqD != gEqD) {
                    ++bad;
                }
            }
        }

        return bad;
    }

    inline int heuristicMisplacedCells(const Board& state, const Board& goal) {
        int bad = 0;
        for (size_t x = 0; x < 6; ++x) {
            for (size_t y = 0; y < 6; ++y) {
                if (state.getColor(x, y) != goal.getColor(x, y)) {
                    ++bad;
                }
            }
        }
        return bad;
    }

    inline int heuristicColorParityImbalance(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int total = 0;

        for (int color = 1; color <= maxColor; ++color) {
            int sEven = 0;
            int gEven = 0;
            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    const bool even = ((x + y) & 1ULL) == 0;
                    if (even && state.getColor(x, y) == color) {
                        ++sEven;
                    }
                    if (even && goal.getColor(x, y) == color) {
                        ++gEven;
                    }
                }
            }
            total += std::abs(sEven - gEven);
        }

        return total;
    }

    inline int heuristicQuadrantColorImbalance(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int total = 0;

        auto quadrantIndex = [](const size_t x, const size_t y) -> size_t {
            const size_t qx = (x >= 3) ? 1 : 0;
            const size_t qy = (y >= 3) ? 1 : 0;
            return qy * 2 + qx;
        };

        for (int color = 1; color <= maxColor; ++color) {
            std::array<int, 4> s{};
            std::array<int, 4> g{};

            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    const size_t q = quadrantIndex(x, y);
                    if (state.getColor(x, y) == color) {
                        ++s[q];
                    }
                    if (goal.getColor(x, y) == color) {
                        ++g[q];
                    }
                }
            }

            for (size_t q = 0; q < 4; ++q) {
                total += std::abs(s[q] - g[q]);
            }
        }

        return total / 2;
    }

    inline int rowHistogramDiff(const Board& a, const Board& b, const size_t rowA, const size_t rowB, const int maxColor) {
        const size_t colorCount = static_cast<size_t>(maxColor) + 1;
        std::vector<int> ha(colorCount, 0);
        std::vector<int> hb(colorCount, 0);

        for (size_t x = 0; x < 6; ++x) {
            ++ha[a.getColor(x, rowA)];
            ++hb[b.getColor(x, rowB)];
        }

        int diff = 0;
        for (int c = 1; c <= maxColor; ++c) {
            diff += std::abs(ha[c] - hb[c]);
        }
        return diff / 2;
    }

    inline int colHistogramDiff(const Board& a, const Board& b, const size_t colA, const size_t colB, const int maxColor) {
        const size_t colorCount = static_cast<size_t>(maxColor) + 1;
        std::vector<int> ha(colorCount, 0);
        std::vector<int> hb(colorCount, 0);

        for (size_t y = 0; y < 6; ++y) {
            ++ha[a.getColor(colA, y)];
            ++hb[b.getColor(colB, y)];
        }

        int diff = 0;
        for (int c = 1; c <= maxColor; ++c) {
            diff += std::abs(ha[c] - hb[c]);
        }
        return diff / 2;
    }

    inline int heuristicRowColSignatureMatching(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));

        std::vector<std::vector<int>> rowCost(6, std::vector<int>(6, 0));
        std::vector<std::vector<int>> colCost(6, std::vector<int>(6, 0));

        for (size_t r1 = 0; r1 < 6; ++r1) {
            for (size_t r2 = 0; r2 < 6; ++r2) {
                rowCost[r1][r2] = rowHistogramDiff(state, goal, r1, r2, maxColor);
                colCost[r1][r2] = colHistogramDiff(state, goal, r1, r2, maxColor);
            }
        }

        return hungarianMinCost(rowCost) + hungarianMinCost(colCost);
    }

    inline int heuristicSinkhornEntropicOT(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        constexpr double eps = 1.5;
        constexpr int iters = 25;
        double total = 0.0;

        for (int color = 1; color <= maxColor; ++color) {
            std::vector<Pos> s;
            std::vector<Pos> g;
            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    if (state.getColor(x, y) == color) {
                        s.push_back({int(x), int(y)});
                    }
                    if (goal.getColor(x, y) == color) {
                        g.push_back({int(x), int(y)});
                    }
                }
            }

            if (s.size() != g.size()) {
                return 1000000000;
            }
            if (s.empty()) {
                continue;
            }

            const size_t n = s.size();
            std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
            std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));

            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    const double cij = cellTransportCost(s[i], g[j], true);
                    C[i][j] = cij;
                    K[i][j] = std::exp(-cij / eps);
                }
            }

            const double marg = 1.0 / static_cast<double>(n);
            std::vector<double> u(n, marg);
            std::vector<double> v(n, marg);

            for (int t = 0; t < iters; ++t) {
                for (size_t i = 0; i < n; ++i) {
                    double denom = 0.0;
                    for (size_t j = 0; j < n; ++j) {
                        denom += K[i][j] * v[j];
                    }
                    u[i] = (denom > 1e-12) ? (marg / denom) : marg;
                }
                for (size_t j = 0; j < n; ++j) {
                    double denom = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        denom += K[i][j] * u[i];
                    }
                    v[j] = (denom > 1e-12) ? (marg / denom) : marg;
                }
            }

            double expected_cost = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    const double pij = u[i] * K[i][j] * v[j];
                    expected_cost += pij * C[i][j];
                }
            }

            total += expected_cost;
        }

        return static_cast<int>(std::lround(total * 100.0));
    }

    inline int cellMoveAwareFlowCost(const Pos& from, const Pos& to) {
        const int dx = torusDelta(from.x, to.x);
        const int dy = torusDelta(from.y, to.y);
        const int same_ring_bonus = (from.x == to.x || from.y == to.y) ? 0 : 1;
        const int half_turn_penalty = (dx == 3 || dy == 3) ? 1 : 0;
        return (dx + dy) * 2 + same_ring_bonus + half_turn_penalty;
    }

    inline int heuristicMinCostFlowMoveAware(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int total = 0;

        for (int color = 1; color <= maxColor; ++color) {
            std::vector<Pos> s;
            std::vector<Pos> g;

            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    if (state.getColor(x, y) == color) {
                        s.push_back({int(x), int(y)});
                    }
                    if (goal.getColor(x, y) == color) {
                        g.push_back({int(x), int(y)});
                    }
                }
            }

            if (s.size() != g.size()) {
                return 1000000000;
            }
            if (s.empty()) {
                continue;
            }

            const size_t n = s.size();
            std::vector<std::vector<int>> cost(n, std::vector<int>(n, 0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    cost[i][j] = cellMoveAwareFlowCost(s[i], g[j]);
                }
            }
            total += hungarianMinCost(cost);
        }

        return total;
    }

    inline int heuristicCycleConsistencyPenalty(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        int penalty = 0;

        for (int color = 1; color <= maxColor; ++color) {
            std::vector<Pos> s;
            std::vector<Pos> g;
            for (size_t x = 0; x < 6; ++x) {
                for (size_t y = 0; y < 6; ++y) {
                    if (state.getColor(x, y) == color) {
                        s.push_back({int(x), int(y)});
                    }
                    if (goal.getColor(x, y) == color) {
                        g.push_back({int(x), int(y)});
                    }
                }
            }

            if (s.size() != g.size()) {
                return 1000000000;
            }
            if (s.empty()) {
                continue;
            }

            std::sort(s.begin(), s.end(), [](const Pos& a, const Pos& b) {
                if (a.y != b.y) return a.y < b.y;
                return a.x < b.x;
            });
            std::sort(g.begin(), g.end(), [](const Pos& a, const Pos& b) {
                if (a.y != b.y) return a.y < b.y;
                return a.x < b.x;
            });

            const size_t n = s.size();
            std::vector<std::vector<int>> cost(n, std::vector<int>(n, 0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    cost[i][j] = cellTransportCost(s[i], g[j], false);
                }
            }

            const HungarianResult res = hungarianSolve(cost);
            const std::vector<int>& p = res.match_row_to_col;
            std::vector<char> seen(n, 0);
            int cycles = 0;
            for (size_t i = 0; i < n; ++i) {
                if (seen[i] != 0) {
                    continue;
                }
                ++cycles;
                size_t cur = i;
                while (seen[cur] == 0) {
                    seen[cur] = 1;
                    cur = static_cast<size_t>(p[cur]);
                }
            }

            penalty += (static_cast<int>(n) - cycles);
        }

        return penalty;
    }

    inline double entropyFromCounts(const std::vector<int>& counts, const int total) {
        if (total <= 0) {
            return 0.0;
        }
        double h = 0.0;
        for (const int c : counts) {
            if (c <= 0) {
                continue;
            }
            const double p = static_cast<double>(c) / static_cast<double>(total);
            h -= p * std::log(p);
        }
        return h;
    }

    inline double mutualInfoAxisColor(const Board& b, const int maxColor, const bool rows) {
        constexpr size_t axis = 6;
        constexpr double total = 36.0;
        std::vector<std::vector<int>> joint(axis, std::vector<int>(static_cast<size_t>(maxColor + 1), 0));
        std::vector<int> axisCounts(axis, 0);
        std::vector<int> colorCounts(static_cast<size_t>(maxColor + 1), 0);

        for (size_t x = 0; x < 6; ++x) {
            for (size_t y = 0; y < 6; ++y) {
                const size_t idx = rows ? y : x;
                const int color = b.getColor(x, y);
                ++joint[idx][color];
                ++axisCounts[idx];
                ++colorCounts[color];
            }
        }

        double mi = 0.0;
        for (size_t i = 0; i < axis; ++i) {
            const double pa = static_cast<double>(axisCounts[i]) / total;
            if (pa <= 0.0) {
                continue;
            }
            for (int c = 1; c <= maxColor; ++c) {
                const int j = joint[i][c];
                if (j <= 0) {
                    continue;
                }
                const double pjc = static_cast<double>(j) / total;
                const double pc = static_cast<double>(colorCounts[c]) / total;
                if (pc > 0.0) {
                    mi += pjc * std::log(pjc / (pa * pc));
                }
            }
        }

        return mi;
    }

    inline int heuristicMutualInfoEntropySignatureMismatch(const Board& state, const Board& goal) {
        const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
        double score = 0.0;

        const double miRowsS = mutualInfoAxisColor(state, maxColor, true);
        const double miRowsG = mutualInfoAxisColor(goal, maxColor, true);
        const double miColsS = mutualInfoAxisColor(state, maxColor, false);
        const double miColsG = mutualInfoAxisColor(goal, maxColor, false);
        score += std::abs(miRowsS - miRowsG);
        score += std::abs(miColsS - miColsG);

        for (size_t i = 0; i < 6; ++i) {
            std::vector<int> sRow(static_cast<size_t>(maxColor + 1), 0);
            std::vector<int> gRow(static_cast<size_t>(maxColor + 1), 0);
            std::vector<int> sCol(static_cast<size_t>(maxColor + 1), 0);
            std::vector<int> gCol(static_cast<size_t>(maxColor + 1), 0);

            for (size_t x = 0; x < 6; ++x) {
                ++sRow[state.getColor(x, i)];
                ++gRow[goal.getColor(x, i)];
            }
            for (size_t y = 0; y < 6; ++y) {
                ++sCol[state.getColor(i, y)];
                ++gCol[goal.getColor(i, y)];
            }

            score += std::abs(entropyFromCounts(sRow, 6) - entropyFromCounts(gRow, 6));
            score += std::abs(entropyFromCounts(sCol, 6) - entropyFromCounts(gCol, 6));
        }

        return static_cast<int>(std::lround(score * 1000.0));
    }

    inline std::array<int, kHeuristicCount> evaluateHeuristics(const Board& state, const Board& goal) {
        return {
                heuristicColorTransportMatching(state, goal, false),
                heuristicColorTransportMatching(state, goal, true),
                heuristicRowColImbalance(state, goal),
                heuristicEdgeDisagreement(state, goal),
                heuristicMisplacedCells(state, goal),
                heuristicColorParityImbalance(state, goal),
                heuristicQuadrantColorImbalance(state, goal),
                heuristicRowColSignatureMatching(state, goal),
                heuristicSinkhornEntropicOT(state, goal),
                heuristicMinCostFlowMoveAware(state, goal),
                heuristicCycleConsistencyPenalty(state, goal),
                heuristicMutualInfoEntropySignatureMismatch(state, goal)
        };
    }

    template<int LOOKAHEAD_DEPTH>
    inline std::array<int, kHeuristicCount> evaluateLookaheadMinScores(const Board& seedBoard, const Board& goal) {
        if constexpr (LOOKAHEAD_DEPTH == 0) {
            return evaluateHeuristics(seedBoard, goal);
        } else {
            std::array<int, kHeuristicCount> best{};
            best.fill(1000000000);

            JVec<B1B2> states;
            JVec<u64> hashes;
            buildUniqueNoneDepthFrontierB1B2<LOOKAHEAD_DEPTH>(seedBoard, states, hashes, false);

            if (states.empty()) {
                return evaluateHeuristics(seedBoard, goal);
            }

            for (std::size_t i = 0; i < states.size(); ++i) {
                const Board b = makeBoardFromState(states[i]);
                const auto scores = evaluateHeuristics(b, goal);
                for (size_t h = 0; h < kHeuristicCount; ++h) {
                    if (scores[h] < best[h]) {
                        best[h] = scores[h];
                    }
                }
            }
            return best;
        }
    }

    inline std::array<int, kHeuristicCount> evaluateLookaheadMinScoresAtDepth(const Board& seedBoard,
                                                                              const Board& goal,
                                                                              const int lookaheadDepth) {
        switch (lookaheadDepth) {
            case 0:
                return evaluateLookaheadMinScores<0>(seedBoard, goal);
            case 1:
                return evaluateLookaheadMinScores<1>(seedBoard, goal);
            case 2:
                return evaluateLookaheadMinScores<2>(seedBoard, goal);
            case 3:
                return evaluateLookaheadMinScores<3>(seedBoard, goal);
            default:
                return evaluateLookaheadMinScores<0>(seedBoard, goal);
        }
    }

} // namespace heur