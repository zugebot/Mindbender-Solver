// BetterHeuristic.hpp
#pragma once
#include "MindbenderSolver/include.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <queue>

// Admissible lower bound: minimum number of lines (rows/cols) you must act on
// to "touch" every mismatched cell. This is exactly the size of a minimum
// vertex cover in the bipartite graph (rows ↔ cols with edges at mismatches).
// By Kőnig's theorem, min vertex cover == max matching. We compute a 6x6
// Hopcroft–Karp specialized to fixed-size arrays (no heap).
//
// Final h = max( maxMatching, ceil(mismatches / 6) ).
namespace better_heur {

struct Heuristic {
    // Store goal colors flattened 6x6, no heap
    std::array<uint8_t, 36> goal{};

    static Heuristic fromGoalBoard(const Board& g) {
        Heuristic H;
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 6; ++x)
                H.goal[y * 6 + x] = static_cast<uint8_t>(g.getColor(x, y));
        return H;
    }

    inline uint8_t goalColor(int x, int y) const { return goal[y * 6 + x]; }

    // Small fixed-capacity vector (0..6 entries) to avoid allocations
    struct SmallVec {
        uint8_t n = 0;
        uint8_t v[6];
        inline void clear() { n = 0; }
        inline void push(uint8_t x) { v[n++] = x; }
        inline uint8_t size() const { return n; }
        inline uint8_t operator[](uint8_t i) const { return v[i]; }
    };

    // Evaluate admissible lower bound
    inline uint8_t eval(const Board& b) const {
        // Build edges: row y → col x when cell (x,y) mismatches goal
        std::array<SmallVec, 6> adj{};
        int mismatches = 0;

        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                const uint8_t cur  = static_cast<uint8_t>(b.getColor(x, y));
                const uint8_t want = goalColor(x, y);
                if (cur != want) {
                    adj[y].push(static_cast<uint8_t>(x));
                    ++mismatches;
                }
            }
        }

        if (mismatches == 0) return 0;

        const uint8_t lb_cover = maxMatching6x6(adj);           // == min vertex cover
        const uint8_t lb_batch = static_cast<uint8_t>((mismatches + 5) / 6); // ceil(m/6)
        const uint8_t h = (lb_cover > lb_batch) ? lb_cover : lb_batch;
        return h; // 0..6
    }

private:
    // Hopcroft–Karp specialized for 6x6, no heap
    static uint8_t maxMatching6x6(const std::array<SmallVec, 6>& adj) {
        int8_t matchU[6]; // row -> col or -1
        int8_t matchV[6]; // col -> row or -1
        int8_t dist[6];

        for (int i = 0; i < 6; ++i) { matchU[i] = -1; matchV[i] = -1; }

        auto bfs = [&]() -> bool {
            std::queue<uint8_t> q;
            bool reachableFreeRight = false;

            for (uint8_t u = 0; u < 6; ++u) {
                if (matchU[u] == -1) { dist[u] = 0; q.push(u); }
                else                  { dist[u] = -1; }
            }
            while (!q.empty()) {
                uint8_t u = q.front(); q.pop();
                for (uint8_t i = 0; i < adj[u].size(); ++i) {
                    uint8_t v = adj[u][i];
                    int8_t u2 = matchV[v];
                    if (u2 >= 0) {
                        if (dist[u2] == -1) {
                            dist[u2] = static_cast<int8_t>(dist[u] + 1);
                            q.push(static_cast<uint8_t>(u2));
                        }
                    } else {
                        // There exists a free right vertex reachable from some free left
                        reachableFreeRight = true;
                    }
                }
            }
            return reachableFreeRight;
        };

        // DFS layered augment
        std::function<bool(uint8_t)> dfs = [&](uint8_t u) -> bool {
            for (uint8_t i = 0; i < adj[u].size(); ++i) {
                uint8_t v  = adj[u][i];
                int8_t  u2 = matchV[v];
                if (u2 == -1 || (dist[u2] == dist[u] + 1 && dfs(static_cast<uint8_t>(u2)))) {
                    matchU[u] = static_cast<int8_t>(v);
                    matchV[v] = static_cast<int8_t>(u);
                    return true;
                }
            }
            dist[u] = -1; // dead end in this layer
            return false;
        };

        uint8_t matching = 0;
        while (bfs()) {
            for (uint8_t u = 0; u < 6; ++u)
                if (matchU[u] == -1 && dfs(u))
                    ++matching;
        }
        return matching; // 0..6
    }
};

} // namespace better_heur
