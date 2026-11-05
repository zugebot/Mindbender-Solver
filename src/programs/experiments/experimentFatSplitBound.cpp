// src/programs/experiments/experimentFatSplitBound.cpp
//
// Research (2026-11-04): a lower bound for 2x2 boards that splits the moves by type.
//
// Every solution uses some column moves and some row moves, so
//     length >= (fewest column moves needed when row moves are free)
//             + (fewest row moves needed when column moves are free).
// The rotation lane bound is this with each half estimated one lane at a time; on 2x2
// boards (a move on the piece's lanes shifts two lanes) each half is at most 3, so it never
// exceeds 6 and cannot prune a 2x2 search, which needs more than the goal side's depth (7).
// Here each half is computed exactly on an abstract board: the cells of a colour class
// (1 = in the class) and the piece position, the same abstraction as pattern_db.hpp, but
// searched with a 0-1 BFS where the free move type costs nothing. Real moves map to
// abstract moves, so each table value is a lower bound on its half; the max over classes
// (and the rotation bound's halves) is still one, and the two halves add.
//
//   experimentFatSplitBound <puzzle> [--max-cells K] [--threads N] [--moves N] [--goal-depth T]
//                           [--samples N]
//     Builds the tables for every class of 1-3 colours with at most K cells (or at least
//     36 - K), replays the puzzle's known solutions in build/levels_final (run from the repo
//     root) and reports violations (must be 0), then samples left states of an N-move search
//     <5, N-T-5, T> (exact depth-5 seeds plus random moves) and prints how often
//     h > T + r at each of the last levels (r = moves still below the node), i.e. how much
//     of the stream the bound would skip.

#include "src/chuzzles/include.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/solver/bounds/lane_bound.hpp"
#include "src/solver/bounds/pattern_db.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

    using pattern_db::Table;

    // A move-type-split table: dist[i] = fewest costly moves from abstract state i to the
    // goal's abstract board, the other move type free.
    struct SplitTable {
        Table t;                 // indexing (class, ranks, piece position)
        bool countCols = true;   // true: column moves cost 1, row moves are free
        std::vector<u8> dist;
        u32 maxDist = 0;
        double seconds = 0;
    };

    // fatActionsIndexes rows: slots [0, 24) are row moves, [24, 48) column moves
    constexpr u32 FAT_ROW_SLOTS = 24;

    template<typename Fn>
    void forChunks(const u32 threads, const u64 count, const u64 chunk, Fn&& fn) {
        std::atomic<u64> next{0};
        std::vector<std::thread> pool;
        for (u32 th = 0; th < threads; ++th) {
            pool.emplace_back([&, th] {
                for (u64 c; (c = next.fetch_add(1)) * chunk < count;) {
                    fn(th, c * chunk, std::min(count, (c + 1) * chunk));
                }
            });
        }
        for (auto& p : pool) {
            p.join();
        }
    }

    SplitTable buildSplit(const B1B2& target, const u8 classMask, const bool countCols, const u32 threads) {
        Timer timer;
        SplitTable s;
        s.countCols = countCols;
        Table& t = s.t;
        t.classMask = classMask;
        t.fat = target.getFatBool();
        u64 f1 = 0, f2 = 0;
        for (u32 c = 0; c < 8; ++c) {
            if (classMask & (1U << c)) {
                f1 |= pattern_db::colourFlags(target.b1, c);
                f2 |= pattern_db::colourFlags(target.b2, c);
            }
        }
        const B1B2 goal((target.b1 & ~((1ULL << 54) - 1)) | f1, f2);
        t.cells = static_cast<u32>(my_popcount(f1) + my_popcount(f2));
        t.complement = t.cells > 18;
        t.rankK = t.complement ? 36 - t.cells : t.cells;
        t.entries = pattern_db::BINOM.v[36][t.rankK] * (t.fat ? 25 : 1);
        s.dist.assign(t.entries, 255);
        u8* dist = s.dist.data();

        // children of abstract state i of one move type (free or costly)
        auto children = [&](const u64 i, const bool costly, u64* out) -> u32 {
            const B1B2 x = t.abstractOfIndex(i, goal);
            const u8* idx = fatActionsIndexes[x.getFatXY()];
            u32 n = 0;
            for (u32 a = 0; a < frontier_recovery_detail::FAT_NONE_MOVE_COUNT; ++a) {
                if ((a >= FAT_ROW_SLOTS) != (costly == countCols)) {
                    continue;
                }
                B1B2 y = x;
                allActStructList[idx[a]].action(y);
                if (!(y == x)) {
                    out[n++] = t.indexOfAbstract(y);
                }
            }
            return n;
        };
        auto claim = [&](const u64 j, const u8 d) {
            u8 expected = 255;
            return dist[j] == 255 && __atomic_compare_exchange_n(&dist[j], &expected, d, false, __ATOMIC_RELAXED,
                                                                 __ATOMIC_RELAXED);
        };

        std::vector<u64> queue{t.indexOfAbstract(goal)};
        dist[queue[0]] = 0;
        for (u32 d = 0;; ++d) {
            // close level d under the free moves (BFS by queue)
            while (!queue.empty()) {
                std::vector<std::vector<u64>> next(threads);
                forChunks(threads, queue.size(), 1ULL << 12, [&](const u32 th, const u64 lo, const u64 hi) {
                    u64 kids[64];
                    for (u64 q = lo; q < hi; ++q) {
                        const u32 n = children(queue[q], false, kids);
                        for (u32 k = 0; k < n; ++k) {
                            if (claim(kids[k], static_cast<u8>(d))) {
                                next[th].push_back(kids[k]);
                            }
                        }
                    }
                });
                queue.clear();
                for (auto& v : next) {
                    queue.insert(queue.end(), v.begin(), v.end());
                }
            }
            // one costly move from every state at level d (scan)
            std::vector<std::vector<u64>> next(threads);
            forChunks(threads, t.entries, 1ULL << 16, [&](const u32 th, const u64 lo, const u64 hi) {
                u64 kids[64];
                for (u64 i = lo; i < hi; ++i) {
                    if (dist[i] != d) {
                        continue;
                    }
                    const u32 n = children(i, true, kids);
                    for (u32 k = 0; k < n; ++k) {
                        if (claim(kids[k], static_cast<u8>(d + 1))) {
                            next[th].push_back(kids[k]);
                        }
                    }
                }
            });
            for (auto& v : next) {
                queue.insert(queue.end(), v.begin(), v.end());
            }
            if (queue.empty()) {
                break;
            }
            s.maxDist = d + 1;
        }
        s.seconds = timer.getSeconds();
        return s;
    }

    struct Bound {
        std::vector<SplitTable> tables;
        B1B2 goal;

        // h = colMoves + rowMoves, each the max of its tables and the rotation bound's half
        void eval(const B1B2& s, int& rot, int& split) const {
            u8 a[12][6], g[12][6];
            lane_bound::laneCells(s, a);
            lane_bound::laneCells(goal, g);
            int r = 0, c = 0;
            for (u32 L = 0; L < 12; ++L) {
                const int d = lane_bound::rotationDistanceRef(a[L], g[L]);
                (L < 6 ? r : c) = std::max(L < 6 ? r : c, d);
            }
            int colMoves = (r + 1) / 2, rowMoves = (c + 1) / 2;   // rows fixed by column moves and vice versa
            rot = colMoves + rowMoves;
            for (const SplitTable& st : tables) {
                const int v = st.dist[st.t.indexOf(s)];
                if (st.countCols) {
                    colMoves = std::max(colMoves, v);
                } else {
                    rowMoves = std::max(rowMoves, v);
                }
            }
            split = colMoves + rowMoves;
        }
    };

    std::vector<u8> chooseClasses(const B1B2& target, const u32 maxCells) {
        u32 count[8] = {};
        for (u32 c = 0; c < 8; ++c) {
            count[c] = static_cast<u32>(my_popcount(pattern_db::colourFlags(target.b1, c))
                                        + my_popcount(pattern_db::colourFlags(target.b2, c)));
        }
        std::vector<u8> out;
        for (u32 m = 1; m < 256; ++m) {
            u32 cells = 0, colours = 0;
            bool used = true;
            for (u32 c = 0; c < 8; ++c) {
                if (m & (1U << c)) {
                    used &= count[c] != 0;
                    cells += count[c];
                    ++colours;
                }
            }
            if (used && colours <= 3 && cells > 0 && cells < 36 && std::min(cells, 36 - cells) <= maxCells) {
                out.push_back(static_cast<u8>(m));
            }
        }
        return out;
    }

    int checkSolutions(const std::string& puzzle, const Board& start, const Bound& bound) {
        const std::regex name("(" + std::regex_replace(puzzle, std::regex("-"), "\\-") + R"()_c(\d+)_(\d+)\.txt)");
        u64 states = 0, violations = 0, lines = 0;
        int startRot = 0, startSplit = 0, optimal = 0;
        for (const auto& e : std::filesystem::directory_iterator("build/levels_final")) {
            std::smatch m;
            const std::string fn = e.path().filename().string();
            if (!std::regex_match(fn, m, name)) {
                continue;
            }
            optimal = std::stoi(m[2].str());
            std::ifstream in(e.path());
            std::string line;
            while (std::getline(in, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                    line.pop_back();
                }
                if (line.empty()) {
                    continue;
                }
                const std::vector<u8> moves = Memory::parseFatMoveString(line);
                const int n = static_cast<int>(moves.size());
                Board b = start;
                for (int i = 0; i <= n; ++i) {
                    int rot = 0, split = 0;
                    bound.eval(b.asB1B2(), rot, split);
                    if (i == 0) {
                        startRot = rot;
                        startSplit = split;
                    }
                    ++states;
                    if (split > n - i) {
                        if (violations < 10) {
                            tcout << "VIOLATION step " << i << ": h=" << split << " > " << n - i << " in " << line << "\n";
                        }
                        ++violations;
                    }
                    if (i < n) {
                        allActStructList[moves[i]].action(b);
                    }
                }
                ++lines;
            }
        }
        if (lines == 0) {
            tcout << "known solutions: none in build/levels_final (unsolved puzzle?)\n";
            return 0;
        }
        tcout << "known solutions: " << lines << " (optimal " << optimal << "), " << states << " path states, "
              << violations << " violations -> " << (violations == 0 ? "ADMISSIBLE" : "NOT ADMISSIBLE")
              << "; at the start: rotation " << startRot << ", split " << startSplit << "\n";
        return violations == 0 ? 0 : 1;
    }

} // namespace

int main(const int argc, char** argv) {
    if (argc < 2) {
        tcout << "usage: experimentFatSplitBound <puzzle> [--max-cells K] [--threads N] [--moves N] [--goal-depth T]"
                 " [--samples N]\n";
        return 2;
    }
    const std::string puzzle = argv[1];
    u32 maxCells = 7, threads = 6, moves = 17, goalDepth = 7, samples = 2'000'000;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const u32 v = static_cast<u32>(std::stoul(argv[i + 1]));
        if (a == "--max-cells") maxCells = v;
        else if (a == "--threads") threads = v;
        else if (a == "--moves") moves = v;
        else if (a == "--goal-depth") goalDepth = v;
        else if (a == "--samples") samples = v;
        else {
            tcout << "unknown option " << a << "\n";
            return 2;
        }
    }
    const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
    if (pair == nullptr) {
        tcout << "unknown puzzle " << puzzle << "\n";
        return 2;
    }
    const Board start = pair->getStartState();
    const Board goalBoard = pair->getEndState();
    if (!start.getFatBool()) {
        tcout << puzzle << " is not a 2x2 board\n";
        return 2;
    }
    Bound bound;
    bound.goal = goalBoard.asB1B2();

    const std::vector<u8> classes = chooseClasses(bound.goal, maxCells);
    tcout << puzzle << ": " << classes.size() << " colour classes with at most " << maxCells << " cells (or their"
          << " complement), 2 tables each\n";
    for (const u8 m : classes) {
        for (const bool cols : {true, false}) {
            bound.tables.push_back(buildSplit(bound.goal, m, cols, threads));
            const SplitTable& s = bound.tables.back();
            std::vector<u64> hist(16, 0);
            for (const u8 d : s.dist) {
                hist[std::min<u32>(d, 15)]++;
            }
            tcout << "  " << s.t.name() << " " << s.t.cells << " cells, " << (cols ? "column" : "row")
                  << " moves (other type free): " << s.t.entries << " entries, max " << s.maxDist << ", "
                  << s.seconds << "s; share by value:";
            for (u32 d = 0; d <= s.maxDist; ++d) {
                tcout << " " << d << "=" << 100.0 * static_cast<double>(hist[d]) / static_cast<double>(s.t.entries)
                      << "%";
            }
            tcout << "\n";
        }
    }

    if (checkSolutions(puzzle, start, bound) != 0) {
        return 1;
    }

    // Left states of <5, moves - goalDepth - 5, goalDepth>: exact depth-5 seeds, then random moves.
    const u32 left = moves - goalDepth;
    JVec<B1B2> seeds;
    {
        JVec<u64> hashes;
        buildUniqueNoneDepthFrontierB1B2<5>(start, seeds, hashes, false);
    }
    tcout << "sampling " << samples << " random walks from " << seeds.size() << " depth-5 seeds to depth " << left
          << " (an " << moves << "-move search, goal side " << goalDepth << ")\n";
    const u32 levels = left - 5;
    std::vector<std::array<std::atomic<u64>, 16>> rotHist(levels + 1), splitHist(levels + 1);
    for (auto& h : rotHist) for (auto& b : h) b = 0;
    for (auto& h : splitHist) for (auto& b : h) b = 0;
    forChunks(threads, samples, 4096, [&](const u32 th, const u64 lo, const u64 hi) {
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL * (lo + 1) + th);
        for (u64 k = lo; k < hi; ++k) {
            B1B2 s = seeds[rng() % seeds.size()];
            for (u32 step = 0; step <= levels; ++step) {
                int rot = 0, split = 0;
                bound.eval(s, rot, split);
                rotHist[step][std::min(rot, 15)]++;
                splitHist[step][std::min(split, 15)]++;
                if (step == levels) {
                    break;
                }
                const u8* idx = fatActionsIndexes[s.getFatXY()];
                for (;;) {
                    B1B2 y = s;
                    allActStructList[idx[rng() % frontier_recovery_detail::FAT_NONE_MOVE_COUNT]].action(y);
                    if (!(y == s)) {
                        s = y;
                        break;
                    }
                }
            }
        }
    });
    for (u32 step = 0; step <= levels; ++step) {
        const u32 depth = 5 + step;
        const int below = static_cast<int>(left - depth);   // moves still to stream below this node
        const int limit = static_cast<int>(goalDepth) + below;
        auto share = [&](const std::array<std::atomic<u64>, 16>& h) {
            u64 n = 0, above = 0;
            for (int v = 0; v < 16; ++v) {
                n += h[v];
                if (v > limit) above += h[v];
            }
            return 100.0 * static_cast<double>(above) / static_cast<double>(std::max<u64>(1, n));
        };
        tcout << "  depth " << depth << " (" << below << " moves below, prune when h > " << limit << "): rotation "
              << share(rotHist[step]) << "%, split " << share(splitHist[step]) << "%; split h:";
        u64 n = 0;
        for (int v = 0; v < 16; ++v) n += splitHist[step][v];
        for (int v = 0; v < 16; ++v) {
            if (splitHist[step][v] != 0) {
                tcout << " " << v << "=" << 100.0 * static_cast<double>(splitHist[step][v]) / static_cast<double>(n)
                      << "%";
            }
        }
        tcout << "\n";
    }
    return 0;
}
