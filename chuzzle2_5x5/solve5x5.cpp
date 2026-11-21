// solve5x5 - the command-line front end of solver5x5.hpp (optimal 5x5 Mind Bender solver).
//
// Usage:
//   solve5x5 [--file puzzles_5x5.txt] [--seed N] [--scramble N] [--check] [--list K]
//       every goal in the file, scrambled with as many random moves as its par (or --scramble
//       N); the game's start boards are not known yet
//   solve5x5 --goal "11011 11011 00000 11011 11011" --start "....." [--check] [--list K]
// --check redoes each puzzle with a plain one-sided search (up to --check-max moves) and
// compares the length and the number of solutions.

#include "solver5x5.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>

using namespace mb5;

static bool parseBoard(const std::string& text, u128& out) {
    std::string cells;
    for (char c : text)
        if (c >= '0' && c <= '9') cells += c;
        else if (c != ' ' && c != '\t') return false;
    if (cells.size() != CELLS) return false;
    u8 v[CELLS];
    for (int i = 0; i < CELLS; ++i) v[i] = u8(cells[i] - '0');
    out = fromCells(v);
    return true;
}
static std::string boardText(u128 b) {
    std::string s;
    for (int i = 0; i < CELLS; ++i) {
        if (i && i % N == 0) s += ' ';
        s += char('0' + cellOf(b, i));
    }
    return s;
}

static u128 scramble(u128 goal, int moves, std::mt19937& rng) {
    u128 b = goal;
    int lastLane = -1;
    for (int i = 0; i < moves; ++i) {
        int m;
        do m = int(rng() % MOVES); while (m / 4 == lastLane);
        lastLane = m / 4;
        b = applyMove(b, m);
    }
    return b;
}

static double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static int checkMax = 6;   // the one-sided check's deepest search (depth 7 runs out of memory)

static bool run(const std::string& label, u128 start, u128 goal, int maxDepth, bool check, size_t list) {
    std::printf("%s\n  start %s\n  goal  %s\n", label.c_str(), boardText(start).c_str(), boardText(goal).c_str());
    if (!sameColourCounts(start, goal)) { std::printf("  start and goal have different colour counts\n"); return false; }
    const auto t0 = std::chrono::steady_clock::now();
    const Result r = solve(start, goal, maxDepth, list);
    const double t = secondsSince(t0);
    if (r.length < 0) {
        std::printf("  no solution within %d moves (%zu states, %.0f MB, %.2f s)\n", maxDepth, r.states, r.bytes / 1e6, t);
        return true;
    }
    std::printf("  shortest: %d moves, %llu solutions (%zu states, %.0f MB, %.3f s)\n", r.length,
                (unsigned long long)r.solutions, r.states, r.bytes / 1e6, t);
    bool ok = true;
    for (const auto& s : r.listed) {
        std::string line;
        for (int m : s) line += moveName(m) + ' ';
        const bool good = replay(start, goal, s) && int(s.size()) == r.length;
        ok &= good;
        std::printf("    %s%s\n", line.c_str(), good ? "" : " <- DOES NOT SOLVE IT");
    }
    if (check && r.length > checkMax) {
        std::printf("  check skipped: a one-sided search to %d moves needs too much memory (--check-max %d)\n", r.length, checkMax);
    } else if (check) {
        const auto t1 = std::chrono::steady_clock::now();
        const Result c = solveOneSided(start, goal, maxDepth);
        const bool same = c.length == r.length && c.solutions == r.solutions;
        ok &= same;
        std::printf("  check (one-sided search): %d moves, %llu solutions, %.3f s: %s\n", c.length,
                    (unsigned long long)c.solutions, secondsSince(t1), same ? "SAME" : "DIFFERENT");
    }
    return ok;
}

int main(int argc, char** argv) {
    std::string file = "puzzles_5x5.txt", goalText, startText;
    unsigned seed = 1;
    bool check = false;
    size_t list = 3;
    int maxDepth = 14;
    int scrambleMoves = 0;   // 0: each puzzle's par
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--file") file = next();
        else if (a == "--seed") seed = unsigned(std::stoul(next()));
        else if (a == "--check") check = true;
        else if (a == "--list") list = size_t(std::stoul(next()));
        else if (a == "--max") maxDepth = std::stoi(next());
        else if (a == "--check-max") checkMax = std::stoi(next());
        else if (a == "--scramble") scrambleMoves = std::stoi(next());
        else if (a == "--goal") goalText = next();
        else if (a == "--start") startText = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (!goalText.empty() || !startText.empty()) {
        u128 goal, start;
        if (!parseBoard(goalText, goal) || !parseBoard(startText, start)) {
            std::fprintf(stderr, "--goal and --start each need 25 digits (spaces allowed)\n");
            return 2;
        }
        return run("custom puzzle", start, goal, maxDepth, check, list) ? 0 : 1;
    }
    std::ifstream in(file);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", file.c_str()); return 2; }
    std::string line;
    int failures = 0, count = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        int number, par;
        ls >> number >> par;
        std::string rest, r;
        while (ls >> r) rest += r;
        u128 goal;
        if (!parseBoard(rest, goal)) { std::fprintf(stderr, "bad line: %s\n", line.c_str()); return 2; }
        std::mt19937 rng(seed * 1000003u + unsigned(number));
        const int k = scrambleMoves ? scrambleMoves : par;
        const u128 start = scramble(goal, k, rng);
        char label[80];
        std::snprintf(label, sizeof label, "puzzle %d (par %d), scrambled with %d random moves", number, par, k);
        failures += run(label, start, goal, maxDepth, check, list) ? 0 : 1;
        ++count;
    }
    std::printf("\n%d puzzles, %d failed\n", count, failures);
    return failures ? 1 : 0;
}
