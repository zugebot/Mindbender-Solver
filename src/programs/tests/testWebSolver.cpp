// src/programs/tests/testWebSolver.cpp
//
// Checks the browser solver (src/solver/web/web_solver.hpp) against
// build/levels_final: for every solved puzzle whose optimal length is at most
// --max (default 10), the same length and the identical solution set. With
// --parts N the depth 7-10 search runs as N partitions (as the page's workers
// do) and their results are merged. Run from the repo root.
//
//   testWebSolver [--max 10] [--parts 1] [--only 4-2,9-1] [--levels build/levels_final]

#include "src/chuzzles/levels.hpp"
#include "src/solver/web/web_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Known {
    int moves = 0;
    fs::path file;
};

static std::set<std::string> readLines(const fs::path& p) {
    std::set<std::string> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) out.insert(line);
    }
    return out;
}

static std::set<std::string> splitLines(const std::string& text) {
    std::set<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) out.insert(line);
    }
    return out;
}

int main(int argc, char** argv) {
    int maxMoves = 10;
    unsigned parts = 1;
    std::string only;
    fs::path levelsDir = "build/levels_final";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--max") maxMoves = std::stoi(v);
        else if (k == "--parts") parts = static_cast<unsigned>(std::stoul(v));
        else if (k == "--only") only = "," + v + ",";
        else if (k == "--levels") levelsDir = v;
    }

    // Shortest file per puzzle: <p>_c<moves>_<count>.txt
    std::map<std::string, Known> known;
    for (const auto& e : fs::directory_iterator(levelsDir)) {
        const std::string name = e.path().filename().string();
        const auto c = name.find("_c");
        if (c == std::string::npos) continue;
        const std::string puzzle = name.substr(0, c);
        const int moves = std::stoi(name.substr(c + 2));
        auto it = known.find(puzzle);
        if (it == known.end() || moves < it->second.moves) known[puzzle] = {moves, e.path()};
    }

    int checked = 0, failed = 0;
    double totalSeconds = 0;
    for (const auto& [puzzle, k] : known) {
        if (k.moves > maxMoves) continue;
        if (!only.empty() && only.find("," + puzzle + ",") == std::string::npos) continue;
        const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) continue;
        const Board start = pair->getStartState(), goal = pair->getEndState();
        u8 sc[36], gc[36];
        for (u8 y = 0; y < 6; ++y) {
            for (u8 x = 0; x < 6; ++x) {
                sc[6 * y + x] = start.getColor(x, y);
                gc[6 * y + x] = goal.getColor(x, y);
            }
        }
        const bool fat = start.getFatBool();

        const auto t0 = std::chrono::steady_clock::now();
        int found = -1;
        u64 count = 0, invalid = 0, streamed = 0, memory = 0;
        std::size_t s4 = 0, g4 = 0;
        std::set<std::string> got;
        {
            web_solver::Solver first;
            const std::string err = first.setup(sc, gc, fat, start.getFatX(), start.getFatY(),
                                                goal.getFatX(), goal.getFatY());
            if (!err.empty()) {
                std::printf("%-5s setup failed: %s\n", puzzle.c_str(), err.c_str());
                ++failed;
                continue;
            }
            found = first.solveSmall(maxMoves);
            if (found >= 0) {
                count = first.solutionCount;
                invalid = first.invalidSolutions;
                got = splitLines(first.solutionsText(true));
                memory = first.memoryBytes();
            } else if (maxMoves >= 7) {
                std::vector<web_solver::Solver> workers(parts);
                for (unsigned p = 0; p < parts; ++p) {
                    workers[p].setup(sc, gc, fat, start.getFatX(), start.getFatY(), goal.getFatX(), goal.getFatY());
                    workers[p].buildGoal4();
                    workers[p].buildStart4(p, parts);
                    s4 += workers[p].layerSize(false, 4);
                }
                g4 = workers[0].layerSize(true, 4);
                memory = workers[0].memoryBytes();
                for (int d = 7; d <= std::min(maxMoves, web_solver::Solver::MAX_DEPTH) && found < 0; ++d) {
                    u64 dCount = 0;
                    for (auto& w : workers) {
                        w.beginDepth(d);
                        while (!w.step(1 << 16)) {}
                        w.finishDepth();
                        dCount += w.solutionCount;
                        invalid += w.invalidSolutions;
                        streamed += w.streamedStates;
                    }
                    if (dCount == 0) continue;
                    found = d;
                    count = dCount;
                    for (auto& w : workers) {
                        const auto lines = splitLines(w.solutionsText(true));
                        got.insert(lines.begin(), lines.end());
                    }
                }
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        totalSeconds += secs;

        const std::set<std::string> want = readLines(k.file);
        const bool ok = found == k.moves && count == want.size() && got == want && invalid == 0;
        ++checked;
        if (!ok) ++failed;
        std::printf("%-5s %s%s  moves %d (want %d)  solutions %llu (want %zu)  %.2fs  S4 %zu G4 %zu  streamed %llu  mem %.0f MB%s\n",
                    puzzle.c_str(), ok ? "ok  " : "FAIL", fat ? " 2x2" : "    ", found, k.moves,
                    static_cast<unsigned long long>(count), want.size(), secs, s4, g4,
                    static_cast<unsigned long long>(streamed), memory / 1e6,
                    invalid != 0 ? "  INVALID LINES" : "");
        if (!ok && got != want) {
            int shown = 0;
            for (const auto& l : want) {
                if (!got.count(l) && shown++ < 3) std::printf("      missing: %s\n", l.c_str());
            }
            shown = 0;
            for (const auto& l : got) {
                if (!want.count(l) && shown++ < 3) std::printf("      extra:   %s\n", l.c_str());
            }
        }
        std::fflush(stdout);
    }
    std::printf("\n%d puzzles checked, %d failed, %.1fs total\n", checked, failed, totalSeconds);
    return failed == 0 && checked > 0 ? 0 : 1;
}
