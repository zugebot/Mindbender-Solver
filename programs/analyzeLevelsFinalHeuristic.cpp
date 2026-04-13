#include "code/include.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace {

const fs::path kLevelsFinalDir = "levels_final";
constexpr int kMinDepth = 10;
constexpr int kHeuristicCount = 12;

const std::array<const char*, kHeuristicCount> kHeuristicNames = {
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

struct FileMeta {
    fs::path path;
    int x = 0;
    int y = 0;
    int c = 0;
    int m = 0;
};

struct StepStats {
    std::size_t inc = 0;
    std::size_t dec = 0;
    std::size_t eq = 0;
};

struct SolutionStats {
    bool solved = false;
    std::array<bool, kHeuristicCount> non_increasing{};
    std::array<bool, kHeuristicCount> strictly_decreasing{};
    std::array<std::vector<int>, kHeuristicCount> trajectory;
    std::array<StepStats, kHeuristicCount> step_stats{};

    SolutionStats() {
        non_increasing.fill(true);
        strictly_decreasing.fill(true);
    }
};

struct Pos {
    int x = 0;
    int y = 0;
};

struct HungarianResult {
    int total_cost = 0;
    std::vector<int> match_row_to_col;
};

struct AggregateStats {
    std::size_t files = 0;
    std::size_t files_skipped_missing_level = 0;
    std::size_t files_skipped_fat = 0;
    std::size_t files_processed = 0;

    std::size_t lines_total = 0;
    std::size_t lines_parse_failed = 0;
    std::size_t lines_bad_length = 0;
    std::size_t lines_unsolved = 0;
    std::size_t solutions_valid = 0;

    std::array<std::size_t, kHeuristicCount> solutions_non_increasing{};
    std::array<std::size_t, kHeuristicCount> solutions_strictly_decreasing{};
    std::array<StepStats, kHeuristicCount> steps{};
};

static std::string trimCopy(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static bool parseFileName(const fs::path& filepath, FileMeta& out) {
    static const std::regex pattern(R"((\d+)\-(\d+)_c(\d+)_(\d+)\.txt)");

    std::smatch match;
    const std::string filename = filepath.filename().string();
    if (!std::regex_match(filename, match, pattern)) {
        return false;
    }

    out.path = filepath;
    out.x = std::stoi(match[1].str());
    out.y = std::stoi(match[2].str());
    out.c = std::stoi(match[3].str());
    out.m = std::stoi(match[4].str());
    return true;
}

static bool compareMeta(const FileMeta& a, const FileMeta& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.c != b.c) return a.c < b.c;
    return a.m < b.m;
}

static std::vector<std::string> readLines(const fs::path& filePath) {
    std::vector<std::string> lines;
    std::ifstream in(filePath);
    if (!in.is_open()) {
        return lines;
    }

    std::string line;
    while (std::getline(in, line)) {
        line = trimCopy(line);
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    return lines;
}

static int torusDelta(const int a, const int b) {
    const int d = std::abs(a - b);
    return std::min(d, 6 - d);
}

static int cellTransportCost(const Pos& from, const Pos& to, const bool manhattan) {
    const int dx = torusDelta(from.x, to.x);
    const int dy = torusDelta(from.y, to.y);
    return manhattan ? (dx + dy) : std::max(dx, dy);
}

static HungarianResult hungarianSolve(const std::vector<std::vector<int>>& cost) {
    const int n = static_cast<int>(cost.size());
    if (n == 0) {
        return {};
    }

    const int inf = 1e9;
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
    for (int i = 1; i <= n; ++i) {
        const int col0 = matchRowToCol[i] - 1;
        match_row_to_col_0[static_cast<std::size_t>(i - 1)] = col0;
        total += cost[i - 1][col0];
    }

    HungarianResult out;
    out.total_cost = total;
    out.match_row_to_col = std::move(match_row_to_col_0);
    return out;
}

static int hungarianMinCost(const std::vector<std::vector<int>>& cost) {
    return hungarianSolve(cost).total_cost;
}

static int heuristicColorTransportMatching(const Board& state, const Board& goal, const bool manhattan) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int total = 0;

    for (int color = 1; color <= maxColor; ++color) {
        std::vector<Pos> s;
        std::vector<Pos> g;

        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    s.push_back({x, y});
                }
                if (goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    g.push_back({x, y});
                }
            }
        }

        if (s.size() != g.size()) {
            return 1e9;
        }
        if (s.empty()) {
            continue;
        }

        const int n = static_cast<int>(s.size());
        std::vector<std::vector<int>> cost(n, std::vector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cost[i][j] = cellTransportCost(s[static_cast<std::size_t>(i)], g[static_cast<std::size_t>(j)], manhattan);
            }
        }

        total += hungarianMinCost(cost);
    }

    return total;
}

static int heuristicRowColImbalance(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int total = 0;

    for (int color = 1; color <= maxColor; ++color) {
        std::array<int, 6> sRow{};
        std::array<int, 6> gRow{};
        std::array<int, 6> sCol{};
        std::array<int, 6> gCol{};

        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                const int sColor = state.getColor(static_cast<u8>(x), static_cast<u8>(y));
                const int gColor = goal.getColor(static_cast<u8>(x), static_cast<u8>(y));
                if (sColor == color) {
                    ++sRow[static_cast<std::size_t>(y)];
                    ++sCol[static_cast<std::size_t>(x)];
                }
                if (gColor == color) {
                    ++gRow[static_cast<std::size_t>(y)];
                    ++gCol[static_cast<std::size_t>(x)];
                }
            }
        }

        for (int i = 0; i < 6; ++i) {
            total += std::abs(sRow[static_cast<std::size_t>(i)] - gRow[static_cast<std::size_t>(i)]);
            total += std::abs(sCol[static_cast<std::size_t>(i)] - gCol[static_cast<std::size_t>(i)]);
        }
    }

    return total / 2;
}

static int heuristicEdgeDisagreement(const Board& state, const Board& goal) {
    int bad = 0;

    for (int x = 0; x < 6; ++x) {
        for (int y = 0; y < 6; ++y) {
            const int xr = (x + 1) % 6;
            const int yd = (y + 1) % 6;

            const bool sEqR = state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == state.getColor(static_cast<u8>(xr), static_cast<u8>(y));
            const bool gEqR = goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == goal.getColor(static_cast<u8>(xr), static_cast<u8>(y));
            if (sEqR != gEqR) {
                ++bad;
            }

            const bool sEqD = state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == state.getColor(static_cast<u8>(x), static_cast<u8>(yd));
            const bool gEqD = goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == goal.getColor(static_cast<u8>(x), static_cast<u8>(yd));
            if (sEqD != gEqD) {
                ++bad;
            }
        }
    }

    return bad;
}

static int cellMoveAwareFlowCost(const Pos& from, const Pos& to) {
    const int dx = torusDelta(from.x, to.x);
    const int dy = torusDelta(from.y, to.y);
    const int same_ring_bonus = (from.x == to.x || from.y == to.y) ? 0 : 1;
    const int half_turn_penalty = (dx == 3 || dy == 3) ? 1 : 0;
    return (dx + dy) * 2 + same_ring_bonus + half_turn_penalty;
}

static int heuristicMisplacedCells(const Board& state, const Board& goal) {
    int bad = 0;
    for (int x = 0; x < 6; ++x) {
        for (int y = 0; y < 6; ++y) {
            if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) != goal.getColor(static_cast<u8>(x), static_cast<u8>(y))) {
                ++bad;
            }
        }
    }
    return bad;
}

static int heuristicColorParityImbalance(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int total = 0;

    for (int color = 1; color <= maxColor; ++color) {
        int sEven = 0;
        int gEven = 0;
        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                const bool even = ((x + y) & 1) == 0;
                if (even && state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    ++sEven;
                }
                if (even && goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    ++gEven;
                }
            }
        }
        total += std::abs(sEven - gEven);
    }

    return total;
}

static int heuristicQuadrantColorImbalance(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int total = 0;

    auto quadrantIndex = [](const int x, const int y) -> int {
        const int qx = (x >= 3) ? 1 : 0;
        const int qy = (y >= 3) ? 1 : 0;
        return qy * 2 + qx;
    };

    for (int color = 1; color <= maxColor; ++color) {
        std::array<int, 4> s{};
        std::array<int, 4> g{};

        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                const int q = quadrantIndex(x, y);
                if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    ++s[static_cast<std::size_t>(q)];
                }
                if (goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    ++g[static_cast<std::size_t>(q)];
                }
            }
        }

        for (int q = 0; q < 4; ++q) {
            total += std::abs(s[static_cast<std::size_t>(q)] - g[static_cast<std::size_t>(q)]);
        }
    }

    return total / 2;
}

static int rowHistogramDiff(const Board& a, const Board& b, const int rowA, const int rowB, const int maxColor) {
    std::vector<int> ha(static_cast<std::size_t>(maxColor + 1), 0);
    std::vector<int> hb(static_cast<std::size_t>(maxColor + 1), 0);

    for (int x = 0; x < 6; ++x) {
        ++ha[static_cast<std::size_t>(a.getColor(static_cast<u8>(x), static_cast<u8>(rowA)))];
        ++hb[static_cast<std::size_t>(b.getColor(static_cast<u8>(x), static_cast<u8>(rowB)))];
    }

    int diff = 0;
    for (int c = 1; c <= maxColor; ++c) {
        diff += std::abs(ha[static_cast<std::size_t>(c)] - hb[static_cast<std::size_t>(c)]);
    }
    return diff / 2;
}

static int colHistogramDiff(const Board& a, const Board& b, const int colA, const int colB, const int maxColor) {
    std::vector<int> ha(static_cast<std::size_t>(maxColor + 1), 0);
    std::vector<int> hb(static_cast<std::size_t>(maxColor + 1), 0);

    for (int y = 0; y < 6; ++y) {
        ++ha[static_cast<std::size_t>(a.getColor(static_cast<u8>(colA), static_cast<u8>(y)))];
        ++hb[static_cast<std::size_t>(b.getColor(static_cast<u8>(colB), static_cast<u8>(y)))];
    }

    int diff = 0;
    for (int c = 1; c <= maxColor; ++c) {
        diff += std::abs(ha[static_cast<std::size_t>(c)] - hb[static_cast<std::size_t>(c)]);
    }
    return diff / 2;
}

static int heuristicRowColSignatureMatching(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));

    std::vector<std::vector<int>> rowCost(6, std::vector<int>(6, 0));
    std::vector<std::vector<int>> colCost(6, std::vector<int>(6, 0));

    for (int r1 = 0; r1 < 6; ++r1) {
        for (int r2 = 0; r2 < 6; ++r2) {
            rowCost[static_cast<std::size_t>(r1)][static_cast<std::size_t>(r2)] = rowHistogramDiff(state, goal, r1, r2, maxColor);
            colCost[static_cast<std::size_t>(r1)][static_cast<std::size_t>(r2)] = colHistogramDiff(state, goal, r1, r2, maxColor);
        }
    }

    return hungarianMinCost(rowCost) + hungarianMinCost(colCost);
}

static int heuristicSinkhornEntropicOT(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    constexpr double eps = 1.5;
    constexpr int iters = 25;
    double total = 0.0;

    for (int color = 1; color <= maxColor; ++color) {
        std::vector<Pos> s;
        std::vector<Pos> g;
        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    s.push_back({x, y});
                }
                if (goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    g.push_back({x, y});
                }
            }
        }

        if (s.size() != g.size()) {
            return 1000000000;
        }
        if (s.empty()) {
            continue;
        }

        const int n = static_cast<int>(s.size());
        std::vector<std::vector<double>> C(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
        std::vector<std::vector<double>> K(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double cij = static_cast<double>(cellTransportCost(s[static_cast<std::size_t>(i)], g[static_cast<std::size_t>(j)], true));
                C[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cij;
                K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = std::exp(-cij / eps);
            }
        }

        const double marg = 1.0 / static_cast<double>(n);
        std::vector<double> u(static_cast<std::size_t>(n), marg);
        std::vector<double> v(static_cast<std::size_t>(n), marg);

        for (int t = 0; t < iters; ++t) {
            for (int i = 0; i < n; ++i) {
                double denom = 0.0;
                for (int j = 0; j < n; ++j) {
                    denom += K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j)];
                }
                u[static_cast<std::size_t>(i)] = (denom > 1e-12) ? (marg / denom) : marg;
            }
            for (int j = 0; j < n; ++j) {
                double denom = 0.0;
                for (int i = 0; i < n; ++i) {
                    denom += K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * u[static_cast<std::size_t>(i)];
                }
                v[static_cast<std::size_t>(j)] = (denom > 1e-12) ? (marg / denom) : marg;
            }
        }

        double expected_cost = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double pij = u[static_cast<std::size_t>(i)] * K[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j)];
                expected_cost += pij * C[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
        }

        total += expected_cost;
    }

    return static_cast<int>(std::lround(total * 100.0));
}

static int heuristicMinCostFlowMoveAware(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int total = 0;

    for (int color = 1; color <= maxColor; ++color) {
        std::vector<Pos> s;
        std::vector<Pos> g;

        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    s.push_back({x, y});
                }
                if (goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    g.push_back({x, y});
                }
            }
        }

        if (s.size() != g.size()) {
            return 1000000000;
        }
        if (s.empty()) {
            continue;
        }

        const int n = static_cast<int>(s.size());
        std::vector<std::vector<int>> cost(static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n), 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cellMoveAwareFlowCost(s[static_cast<std::size_t>(i)], g[static_cast<std::size_t>(j)]);
            }
        }
        total += hungarianMinCost(cost);
    }

    return total;
}

static int heuristicCycleConsistencyPenalty(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    int penalty = 0;

    for (int color = 1; color <= maxColor; ++color) {
        std::vector<Pos> s;
        std::vector<Pos> g;
        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                if (state.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    s.push_back({x, y});
                }
                if (goal.getColor(static_cast<u8>(x), static_cast<u8>(y)) == color) {
                    g.push_back({x, y});
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

        const int n = static_cast<int>(s.size());
        std::vector<std::vector<int>> cost(static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n), 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cellTransportCost(s[static_cast<std::size_t>(i)], g[static_cast<std::size_t>(j)], false);
            }
        }

        const HungarianResult res = hungarianSolve(cost);
        const std::vector<int>& p = res.match_row_to_col;
        std::vector<char> seen(static_cast<std::size_t>(n), 0);
        int cycles = 0;
        for (int i = 0; i < n; ++i) {
            if (seen[static_cast<std::size_t>(i)] != 0) {
                continue;
            }
            ++cycles;
            int cur = i;
            while (seen[static_cast<std::size_t>(cur)] == 0) {
                seen[static_cast<std::size_t>(cur)] = 1;
                cur = p[static_cast<std::size_t>(cur)];
            }
        }

        penalty += (n - cycles);
    }

    return penalty;
}

static double entropyFromCounts(const std::vector<int>& counts, const int total) {
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

static double mutualInfoAxisColor(const Board& b, const int maxColor, const bool rows) {
    const int axis = 6;
    const int total = 36;
    std::vector<std::vector<int>> joint(static_cast<std::size_t>(axis), std::vector<int>(static_cast<std::size_t>(maxColor + 1), 0));
    std::vector<int> axisCounts(static_cast<std::size_t>(axis), 0);
    std::vector<int> colorCounts(static_cast<std::size_t>(maxColor + 1), 0);

    for (int x = 0; x < 6; ++x) {
        for (int y = 0; y < 6; ++y) {
            const int idx = rows ? y : x;
            const int color = b.getColor(static_cast<u8>(x), static_cast<u8>(y));
            ++joint[static_cast<std::size_t>(idx)][static_cast<std::size_t>(color)];
            ++axisCounts[static_cast<std::size_t>(idx)];
            ++colorCounts[static_cast<std::size_t>(color)];
        }
    }

    double mi = 0.0;
    for (int i = 0; i < axis; ++i) {
        const double pa = static_cast<double>(axisCounts[static_cast<std::size_t>(i)]) / static_cast<double>(total);
        if (pa <= 0.0) {
            continue;
        }
        for (int c = 1; c <= maxColor; ++c) {
            const int j = joint[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            if (j <= 0) {
                continue;
            }
            const double pjc = static_cast<double>(j) / static_cast<double>(total);
            const double pc = static_cast<double>(colorCounts[static_cast<std::size_t>(c)]) / static_cast<double>(total);
            if (pc > 0.0) {
                mi += pjc * std::log(pjc / (pa * pc));
            }
        }
    }

    return mi;
}

static int heuristicMutualInfoEntropySignatureMismatch(const Board& state, const Board& goal) {
    const int maxColor = static_cast<int>(std::max(state.getColorCount(), goal.getColorCount()));
    double score = 0.0;

    const double miRowsS = mutualInfoAxisColor(state, maxColor, true);
    const double miRowsG = mutualInfoAxisColor(goal, maxColor, true);
    const double miColsS = mutualInfoAxisColor(state, maxColor, false);
    const double miColsG = mutualInfoAxisColor(goal, maxColor, false);
    score += std::abs(miRowsS - miRowsG);
    score += std::abs(miColsS - miColsG);

    for (int i = 0; i < 6; ++i) {
        std::vector<int> sRow(static_cast<std::size_t>(maxColor + 1), 0);
        std::vector<int> gRow(static_cast<std::size_t>(maxColor + 1), 0);
        std::vector<int> sCol(static_cast<std::size_t>(maxColor + 1), 0);
        std::vector<int> gCol(static_cast<std::size_t>(maxColor + 1), 0);

        for (int x = 0; x < 6; ++x) {
            ++sRow[static_cast<std::size_t>(state.getColor(static_cast<u8>(x), static_cast<u8>(i)))];
            ++gRow[static_cast<std::size_t>(goal.getColor(static_cast<u8>(x), static_cast<u8>(i)))];
        }
        for (int y = 0; y < 6; ++y) {
            ++sCol[static_cast<std::size_t>(state.getColor(static_cast<u8>(i), static_cast<u8>(y)))];
            ++gCol[static_cast<std::size_t>(goal.getColor(static_cast<u8>(i), static_cast<u8>(y)))];
        }

        score += std::abs(entropyFromCounts(sRow, 6) - entropyFromCounts(gRow, 6));
        score += std::abs(entropyFromCounts(sCol, 6) - entropyFromCounts(gCol, 6));
    }

    return static_cast<int>(std::lround(score * 1000.0));
}

static std::array<int, kHeuristicCount> evaluateHeuristics(const Board& state, const Board& goal) {
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

static SolutionStats analyzeSolution(const Board& start,
                                     const Board& goal,
                                     const std::vector<u8>& moves) {
    SolutionStats out;

    Board cur = start;
    auto prev = evaluateHeuristics(cur, goal);
    for (int h = 0; h < kHeuristicCount; ++h) {
        out.trajectory[static_cast<std::size_t>(h)].push_back(prev[static_cast<std::size_t>(h)]);
    }

    for (const u8 mv : moves) {
        allActStructList[mv].action(cur);
        const auto curH = evaluateHeuristics(cur, goal);

        for (int h = 0; h < kHeuristicCount; ++h) {
            const int hv = curH[static_cast<std::size_t>(h)];
            out.trajectory[static_cast<std::size_t>(h)].push_back(hv);

            const int delta = hv - prev[static_cast<std::size_t>(h)];
            if (delta > 0) {
                ++out.step_stats[static_cast<std::size_t>(h)].inc;
                out.non_increasing[static_cast<std::size_t>(h)] = false;
                out.strictly_decreasing[static_cast<std::size_t>(h)] = false;
            } else if (delta < 0) {
                ++out.step_stats[static_cast<std::size_t>(h)].dec;
            } else {
                ++out.step_stats[static_cast<std::size_t>(h)].eq;
                out.strictly_decreasing[static_cast<std::size_t>(h)] = false;
            }
        }

        prev = curH;
    }

    out.solved = (cur == goal);
    return out;
}

static void printTrajectory(const std::vector<int>& traj) {
    for (std::size_t i = 0; i < traj.size(); ++i) {
        if (i != 0) {
            std::cout << " -> ";
        }
        std::cout << traj[i];
    }
}

} // namespace

int main() {
    if (!fs::exists(kLevelsFinalDir) || !fs::is_directory(kLevelsFinalDir)) {
        std::cerr << "[ERROR] Missing directory: " << kLevelsFinalDir << "\n";
        return 1;
    }

    std::vector<FileMeta> files;
    for (const auto& entry : fs::directory_iterator(kLevelsFinalDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".txt") {
            continue;
        }

        FileMeta meta;
        if (!parseFileName(entry.path(), meta)) {
            continue;
        }

        if (meta.c < kMinDepth) {
            continue;
        }

        files.push_back(meta);
    }

    std::sort(files.begin(), files.end(), compareMeta);

    if (files.empty()) {
        std::cout << "No files found with c >= " << kMinDepth << " in " << kLevelsFinalDir << "\n";
        return 0;
    }

    AggregateStats global;
    global.files = files.size();

    for (const FileMeta& meta : files) {
        const std::string level_name = std::to_string(meta.x) + "-" + std::to_string(meta.y);
        const BoardPair* pair = BoardLookup::getBoardPair(level_name);
        if (pair == nullptr) {
            ++global.files_skipped_missing_level;
            continue;
        }

        const Board start = pair->getStartState();
        const Board goal = pair->getEndState();
        if (start.getFatBool() || goal.getFatBool()) {
            ++global.files_skipped_fat;
            continue;
        }

        ++global.files_processed;

        const auto lines = readLines(meta.path);
        global.lines_total += lines.size();

        std::size_t file_valid = 0;
        std::array<std::size_t, kHeuristicCount> file_non_inc{};
        std::array<std::size_t, kHeuristicCount> file_strict_dec{};
        std::array<StepStats, kHeuristicCount> file_steps{};

        bool printed_example = false;

        for (const std::string& line : lines) {
            std::vector<u8> moves;
            try {
                moves = Memory::parseNormMoveString(line);
            } catch (...) {
                ++global.lines_parse_failed;
                continue;
            }

            if (static_cast<int>(moves.size()) != meta.c) {
                ++global.lines_bad_length;
                continue;
            }

            const SolutionStats s = analyzeSolution(start, goal, moves);
            if (!s.solved) {
                ++global.lines_unsolved;
                continue;
            }

            ++file_valid;
            ++global.solutions_valid;

            for (int h = 0; h < kHeuristicCount; ++h) {
                const std::size_t hi = static_cast<std::size_t>(h);
                if (s.non_increasing[hi]) {
                    ++file_non_inc[hi];
                    ++global.solutions_non_increasing[hi];
                }
                if (s.strictly_decreasing[hi]) {
                    ++file_strict_dec[hi];
                    ++global.solutions_strictly_decreasing[hi];
                }

                file_steps[hi].inc += s.step_stats[hi].inc;
                file_steps[hi].dec += s.step_stats[hi].dec;
                file_steps[hi].eq += s.step_stats[hi].eq;

                global.steps[hi].inc += s.step_stats[hi].inc;
                global.steps[hi].dec += s.step_stats[hi].dec;
                global.steps[hi].eq += s.step_stats[hi].eq;
            }

            if (!printed_example) {
                for (int h = 0; h < kHeuristicCount; ++h) {
                    std::cout << "  example " << kHeuristicNames[static_cast<std::size_t>(h)] << ": ";
                    printTrajectory(s.trajectory[static_cast<std::size_t>(h)]);
                    std::cout << "\n";
                }
                printed_example = true;
            }
        }

        std::cout << "[" << level_name << ", c" << meta.c << "]"
                  << " lines=" << lines.size()
                  << " valid=" << file_valid
                  << " file=" << meta.path.filename().string()
                  << "\n";

        for (int h = 0; h < kHeuristicCount; ++h) {
            const std::size_t hi = static_cast<std::size_t>(h);
            std::cout << "    " << kHeuristicNames[hi]
                      << " nonInc=" << file_non_inc[hi]
                      << " strictDec=" << file_strict_dec[hi]
                      << " step(dec/eq/inc)="
                      << file_steps[hi].dec << "/" << file_steps[hi].eq << "/" << file_steps[hi].inc
                      << "\n";
        }
    }

    std::cout << "\n=== Global Summary ===\n"
              << "files total           : " << global.files << "\n"
              << "files processed       : " << global.files_processed << "\n"
              << "files skipped (level) : " << global.files_skipped_missing_level << "\n"
              << "files skipped (fat)   : " << global.files_skipped_fat << "\n"
              << "lines total           : " << global.lines_total << "\n"
              << "lines parse failed    : " << global.lines_parse_failed << "\n"
              << "lines bad length      : " << global.lines_bad_length << "\n"
              << "lines unsolved replay : " << global.lines_unsolved << "\n"
              << "solutions valid       : " << global.solutions_valid << "\n";

    for (int h = 0; h < kHeuristicCount; ++h) {
        const std::size_t hi = static_cast<std::size_t>(h);
        std::cout << "  " << kHeuristicNames[hi] << "\n"
                  << "    solutions nonInc   : " << global.solutions_non_increasing[hi] << "\n"
                  << "    solutions strictDec: " << global.solutions_strictly_decreasing[hi] << "\n"
                  << "    steps dec/eq/inc   : "
                  << global.steps[hi].dec << "/" << global.steps[hi].eq << "/" << global.steps[hi].inc << "\n";
    }

    return 0;
}

