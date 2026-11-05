// src/programs/experiments/experimentBlockJoin.cpp
//
// Research prototype (2026-11-04): meet in the middle across a free block of moves.
//
// A row move only rotates its row, so the row class of a state (every row turned to its
// smallest rotation; on 2x2 boards the piece's row pair shifted so the piece is at column 0,
// with the piece's row kept) is unchanged by row moves, and two states share a row class
// exactly when one block of row moves (distinct lanes) turns one into the other. Same for
// columns. In a shortest solution of length L the moves of one direction between two moves of
// the other are on distinct lanes, so take the block that contains move A + 1: the state
// before it is at most A moves from the start, the state after it at most B = L - 1 - A from
// the goal, and they share a class. So every solution of length L is found by joining the
// start side to depth A with the goal side to depth B on class keys: one level less than a
// plain meet in the middle.
//
// This prototype checks that claim exactly, in RAM, on small searches: exact distances from
// both ends (depth <= 5), goal states indexed by row and column class, every start-side state
// looked up, matches with (start distance + lanes that differ + goal distance) == L expanded
// into every full solution (all shortest prefixes, all block orderings and periodic-row
// amounts, all shortest suffixes), each replayed on the board, then compared with the known
// answer file in build/levels_final (run from the repo root).
//
//   experimentBlockJoin <puzzle> --start-depth A --goal-depth B [--threads N]
//
// It also counts class matches against useful ones (the false-match rate of the join).

#include "src/chuzzles/include.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/solver/bounds/lane_bound.hpp"
#include "src/solver/ssd/class_key.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

    struct Key {
        u64 a = 0, b = 0;
        bool operator<(const Key& o) const { return a != o.a ? a < o.a : b < o.b; }
        bool operator==(const Key& o) const { return a == o.a && b == o.b; }
    };

    // ---------------- moves ----------------

    // Every move available on `s` as a global action index.
    std::vector<u8> movesOf(const B1B2& s) {
        std::vector<u8> out;
        if (s.getFatBool()) {
            const u8* idx = fatActionsIndexes[s.getFatXY()];
            for (u32 a = 0; a < frontier_recovery_detail::FAT_NONE_MOVE_COUNT; ++a) {
                out.push_back(idx[a]);
            }
        } else {
            for (u32 a = 0; a < 30; ++a) out.push_back(static_cast<u8>(a));
            for (u32 a = 32; a < 62; ++a) out.push_back(static_cast<u8>(a));
        }
        return out;
    }

    // row moves: normal 0-29, the piece's row pair 64-88; column moves: 32-61, 89-113
    bool isRowMove(const u8 m) { return m < 32 || (m >= 64 && m <= 88); }

    B1B2 apply(const B1B2& s, const u8 move) {
        B1B2 t = s;
        allActStructList[move].action(t);
        return t;
    }

    // ---------------- class keys ----------------

    // lanes: 0-5 rows (cell k = column k), 6-11 columns (cell k = row k)
    u32 laneCode(const u8 (&cells)[6], const u32 rot) {
        u32 v = 0;
        for (u32 k = 0; k < 6; ++k) {
            v = v * 8 + cells[(k + rot) % 6];
        }
        return v;
    }

    u32 minRotation(const u8 (&cells)[6]) {
        u32 best = ~0U;
        for (u32 r = 0; r < 6; ++r) {
            best = std::min(best, laneCode(cells, r));
        }
        return best;
    }

    // Row class (cols = false) or column class (cols = true).
    Key classKey(const B1B2& s, const bool cols) {
        u8 lanes[12][6];
        lane_bound::laneCells(s, lanes);
        const bool fat = s.getFatBool();
        const u32 fx = fat ? s.getFatX() : 0, fy = fat ? s.getFatY() : 0;
        u64 code[6];
        for (u32 i = 0; i < 6; ++i) {
            const u32 L = cols ? 6 + i : i;
            // the piece's lane pair: rows fy, fy+1 (columns fx, fx+1); rotate so the piece is at 0
            const bool pair = fat && (cols ? (i == fx || i == fx + 1) : (i == fy || i == fy + 1));
            code[i] = pair ? laneCode(lanes[L], cols ? fy : fx) : minRotation(lanes[L]);
        }
        Key k;
        k.a = (code[0] << 36) | (code[1] << 18) | code[2];
        k.b = (code[3] << 36) | (code[4] << 18) | code[5];
        // the piece's row stays fixed under row moves (its column under column moves)
        k.b |= static_cast<u64>(fat ? 1 + (cols ? fx : fy) : 0) << 56;
        return k;
    }

    // Lanes (of one direction) where x and z differ.
    u32 differingLanes(const B1B2& x, const B1B2& z, const bool cols, std::vector<u32>* which = nullptr) {
        u8 a[12][6], b[12][6];
        lane_bound::laneCells(x, a);
        lane_bound::laneCells(z, b);
        u32 n = 0;
        for (u32 i = 0; i < 6; ++i) {
            const u32 L = cols ? 6 + i : i;
            if (!std::equal(a[L], a[L] + 6, b[L])) {
                ++n;
                if (which) which->push_back(i);
            }
        }
        if (x.getFatBool()) {
            // the piece's pair is one lane: count it once
            const u32 p = cols ? x.getFatX() : x.getFatY();
            const bool d0 = !std::equal(a[(cols ? 6 : 0) + p], a[(cols ? 6 : 0) + p] + 6, b[(cols ? 6 : 0) + p]);
            const bool d1 = !std::equal(a[(cols ? 6 : 0) + p + 1], a[(cols ? 6 : 0) + p + 1] + 6,
                                        b[(cols ? 6 : 0) + p + 1]);
            const bool posDiffers = cols ? x.getFatY() != z.getFatY() : x.getFatX() != z.getFatX();
            const bool pairDiffers = d0 || d1 || posDiffers;
            n -= (d0 ? 1 : 0) + (d1 ? 1 : 0);
            n += pairDiffers ? 1 : 0;
            if (which) {
                which->erase(std::remove_if(which->begin(), which->end(),
                                            [&](const u32 i) { return i == p || i == p + 1; }),
                             which->end());
                if (pairDiffers) which->push_back(100 + p);   // 100 + first lane of the pair
            }
        }
        return n;
    }

    // ---------------- exact distances (depth <= 5) ----------------

    struct DistSet {
        std::vector<std::pair<B1B2, u8>> items;   // sorted by state
        int dist(const B1B2& s) const {
            auto it = std::lower_bound(items.begin(), items.end(), std::make_pair(s, u8{0}),
                                       [](const auto& p, const auto& q) { return p.first < q.first; });
            return it != items.end() && it->first == s ? it->second : -1;
        }
    };

    DistSet exactDistances(const Board& root, const u32 depth) {
        std::vector<std::pair<B1B2, u8>> all{{root.asB1B2(), 0}};
        for (u32 d = 1; d <= depth; ++d) {
            JVec<B1B2> f;
            JVec<u64> h;
            switch (d) {
                case 1: buildUniqueNoneDepthFrontierB1B2<1>(root, f, h, false); break;
                case 2: buildUniqueNoneDepthFrontierB1B2<2>(root, f, h, false); break;
                case 3: buildUniqueNoneDepthFrontierB1B2<3>(root, f, h, false); break;
                case 4: buildUniqueNoneDepthFrontierB1B2<4>(root, f, h, false); break;
                case 5: buildUniqueNoneDepthFrontierB1B2<5>(root, f, h, false); break;
                default: throw std::runtime_error("depth <= 5");
            }
            for (std::size_t i = 0; i < f.size(); ++i) {
                all.emplace_back(f[i], static_cast<u8>(d));
            }
        }
        std::sort(all.begin(), all.end(), [](const auto& p, const auto& q) {
            return p.first < q.first || (p.first == q.first && p.second < q.second);
        });
        DistSet ds;
        for (const auto& p : all) {
            if (ds.items.empty() || !(ds.items.back().first == p.first)) {
                ds.items.push_back(p);   // the smallest depth of each state
            }
        }
        return ds;
    }

    // ---------------- solution expansion ----------------

    using Path = std::vector<u8>;

    // every shortest path from the root (dist 0) to x, given x's exact distance d
    void prefixes(const DistSet& ds, const B1B2& x, const int d, Path& tail, std::vector<Path>& out) {
        if (d == 0) {
            out.emplace_back(tail.rbegin(), tail.rend());
            return;
        }
        for (const u8 m : movesOf(x)) {
            const B1B2 y = apply(x, m);   // a neighbour one closer to the root?
            if (y == x || ds.dist(y) != d - 1) continue;
            for (const u8 back : movesOf(y)) {
                if (apply(y, back) == x) {
                    tail.push_back(back);
                    prefixes(ds, y, d - 1, tail, out);
                    tail.pop_back();
                }
            }
        }
    }

    // every shortest path from z to the root of `ds` (z at exact distance d)
    void suffixes(const DistSet& ds, const B1B2& z, const int d, Path& head, std::vector<Path>& out) {
        if (d == 0) {
            out.push_back(head);
            return;
        }
        for (const u8 m : movesOf(z)) {
            const B1B2 y = apply(z, m);
            if (y == z || ds.dist(y) != d - 1) continue;
            head.push_back(m);
            suffixes(ds, y, d - 1, head, out);
            head.pop_back();
        }
    }

    // every block (one move per differing lane of the direction, any order) from x to z
    void blocks(const B1B2& x, const B1B2& z, const bool cols, std::vector<Path>& out) {
        std::vector<std::vector<u8>> perLane;
        std::vector<u32> lanes;
        differingLanes(x, z, cols, &lanes);
        u8 zl[12][6];
        lane_bound::laneCells(z, zl);
        for (const u32 lane : lanes) {
            std::vector<u8> options;
            for (const u8 m : movesOf(x)) {
                const B1B2 y = apply(x, m);
                if (y == x) continue;
                std::vector<u32> changed;
                differingLanes(x, y, cols, &changed);
                if (isRowMove(m) == cols || changed.size() != 1 || changed[0] != lane) continue;
                // the moved lane now matches z (and the piece position, for the pair)
                std::vector<u32> still;
                differingLanes(y, z, cols, &still);
                if (std::find(still.begin(), still.end(), lane) == still.end()) {
                    options.push_back(m);
                }
            }
            if (options.empty()) return;   // cannot happen for a true class match
            perLane.push_back(options);
        }
        // choose one option per lane, then every order
        std::vector<Path> choices{{}};
        for (const auto& opts : perLane) {
            std::vector<Path> next;
            for (const auto& c : choices) {
                for (const u8 m : opts) {
                    Path p = c;
                    p.push_back(m);
                    next.push_back(p);
                }
            }
            choices.swap(next);
        }
        for (Path c : choices) {
            std::sort(c.begin(), c.end());
            do {
                out.push_back(c);
            } while (std::next_permutation(c.begin(), c.end()));
        }
    }

} // namespace

int main(const int argc, char** argv) {
    if (argc < 2) {
        tcout << "usage: experimentBlockJoin <puzzle> --start-depth A --goal-depth B\n";
        return 2;
    }
    const std::string puzzle = argv[1];
    u32 A = 4, B = 4;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--start-depth") A = static_cast<u32>(std::stoul(argv[i + 1]));
        else if (a == "--goal-depth") B = static_cast<u32>(std::stoul(argv[i + 1]));
        else if (a == "--threads") {}
        else { tcout << "unknown option " << a << "\n"; return 2; }
    }
    const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
    if (pair == nullptr) { tcout << "unknown puzzle\n"; return 2; }
    const Board start = pair->getStartState(), goal = pair->getEndState();
    const bool fat = start.getFatBool();
    const u32 L = A + B + 1;

    Timer timer;
    const DistSet fromStart = exactDistances(start, A);
    const DistSet fromGoal = exactDistances(goal, B);
    tcout << puzzle << (fat ? " (2x2)" : "") << ": start side " << fromStart.items.size() << " states (depth <= " << A
          << "), goal side " << fromGoal.items.size() << " (depth <= " << B << "), " << timer.getSeconds() << "s\n";

    // goal index by class: (key, goal-state index), one table per direction
    std::vector<std::pair<Key, u32>> table[2];
    u64 keyMismatch = 0, laneMismatch = 0;   // the shared header vs this file's definitions (only equality matters)
    for (int c = 0; c < 2; ++c) {
        table[c].reserve(fromGoal.items.size());
        for (u32 i = 0; i < fromGoal.items.size(); ++i) {
            Key k;
            class_key::classCode(fromGoal.items[i].first.b1, fromGoal.items[i].first.b2, c == 1, k.a, k.b);
            table[c].emplace_back(k, i);
            if (!(k == classKey(fromGoal.items[i].first, c == 1)) && keyMismatch++ < 0) {}
        }
        std::sort(table[c].begin(), table[c].end(), [](const auto& p, const auto& q) { return p.first < q.first; });
        u64 distinct = 0, biggest = 0, run = 0;
        for (std::size_t i = 0; i < table[c].size(); ++i) {
            run = (i > 0 && table[c][i].first == table[c][i - 1].first) ? run + 1 : 1;
            distinct += run == 1;
            biggest = std::max(biggest, run);
        }
        tcout << "  " << (c ? "column" : "row") << " classes of the goal side: " << distinct << " distinct, largest "
              << biggest << " states\n";
    }

    // join
    u64 matches = 0, useful = 0, shorter = 0;
    std::map<int, u64> byTotal;
    std::vector<std::tuple<u32, u32, int>> hits;   // (start index, goal index, cols)
    for (u32 si = 0; si < fromStart.items.size(); ++si) {
        const auto& [x, a] = fromStart.items[si];
        for (int c = 0; c < 2; ++c) {
            Key k;
            class_key::classCode(x.b1, x.b2, c == 1, k.a, k.b);   // the shared header's class
            auto it = std::lower_bound(table[c].begin(), table[c].end(), std::make_pair(k, 0U),
                                       [](const auto& p, const auto& q) { return p.first < q.first; });
            for (; it != table[c].end() && it->first == k; ++it) {
                const auto& [z, b] = fromGoal.items[it->second];
                ++matches;
                const u32 lanes = class_key::differingLanes(x.b1, x.b2, z.b1, z.b2, c == 1);
                if (lanes != differingLanes(x, z, c == 1)) ++laneMismatch;
                const int total = a + static_cast<int>(lanes) + b;
                byTotal[total]++;
                if (total < static_cast<int>(L)) ++shorter;
                if (total == static_cast<int>(L)) {
                    ++useful;
                    hits.emplace_back(si, it->second, c);
                }
            }
        }
    }
    tcout << "header checks: class-code mismatches (layout may differ, harmless) " << keyMismatch
          << ", lane-count mismatches " << laneMismatch << "\n";
    tcout << "join: " << matches << " class matches, " << useful << " of length " << L << ", " << shorter
          << " shorter; totals:";
    for (const auto& [t, n] : byTotal) tcout << " " << t << "=" << n;
    tcout << " (" << timer.getSeconds() << "s)\n";

    // expand into solutions
    std::set<Path> found;
    for (const auto& [si, gi, c] : hits) {
        const auto& [x, a] = fromStart.items[si];
        const auto& [z, b] = fromGoal.items[gi];
        std::vector<Path> pre, mid, post;
        Path tmp;
        prefixes(fromStart, x, a, tmp, pre);
        suffixes(fromGoal, z, b, tmp, post);
        blocks(x, z, c == 1, mid);
        for (const Path& p : pre) {
            for (const Path& m : mid) {
                for (const Path& q : post) {
                    Path s = p;
                    s.insert(s.end(), m.begin(), m.end());
                    s.insert(s.end(), q.begin(), q.end());
                    found.insert(s);
                }
            }
        }
    }
    // replay check
    u64 bad = 0;
    for (const Path& s : found) {
        B1B2 b = start.asB1B2();
        for (const u8 m : s) b = apply(b, m);
        if (!(b == goal.asB1B2()) || s.size() != L) ++bad;
    }

    // known answers
    std::set<Path> known;
    int knownLen = -1;
    const std::regex name("(" + std::regex_replace(puzzle, std::regex("-"), "\\-") + R"()_c(\d+)_(\d+)\.txt)");
    for (const auto& e : std::filesystem::directory_iterator("build/levels_final")) {
        std::smatch m;
        const std::string fn = e.path().filename().string();
        if (!std::regex_match(fn, m, name) || std::stoi(m[2].str()) != static_cast<int>(L)) continue;
        knownLen = std::stoi(m[2].str());   // the file of length L (some puzzles also keep longer ones)
        std::ifstream in(e.path());
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            const std::vector<u8> mv = fat ? Memory::parseFatMoveString(line) : Memory::parseNormMoveString(line);
            known.insert(Path(mv.begin(), mv.end()));
        }
    }
    u64 missing = 0, extra = 0;
    for (const Path& k : known) missing += found.count(k) == 0;
    for (const Path& f : found) extra += known.count(f) == 0;
    tcout << "solutions of length " << L << ": " << found.size() << " (replay failures " << bad << "); known: "
          << known.size() << " of length " << knownLen << " -> ";
    if (knownLen == static_cast<int>(L)) {
        tcout << "missing " << missing << ", extra " << extra << (missing == 0 && extra == 0 && bad == 0 ? "  IDENTICAL SET" : "  DIFFERENT")
              << "\n";
    } else {
        tcout << (knownLen < static_cast<int>(L) && shorter > 0 ? "shorter solutions seen, as expected" : "lengths differ")
              << "\n";
    }
    tcout << "TOTAL " << timer.getSeconds() << "s\n";
    return 0;
}
