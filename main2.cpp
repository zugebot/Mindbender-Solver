// main2.cpp
#include "BHeuristic.hpp"
#include "LHeuristic.hpp"
#include "MindbenderSolver/include.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

/* ============================================================
   Global new/delete overrides (allocation counter)
   ============================================================ */
static long long new_calls = 0;

void* operator new(std::size_t size) {
    ++new_calls;
    if (void* p = std::malloc(size)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    ++new_calls;
    if (void* p = std::malloc(size)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
/* sized deletes (good for some libstdc++ builds) */
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

template <typename Int>
std::string with_commas(Int v) {
    static_assert(std::is_integral_v<Int>, "with_commas expects an integral type");
    std::string s = std::to_string(v);
    int start = (s[0] == '-') ? 1 : 0;
    for (int i = static_cast<int>(s.size()) - 3; i > start; i -= 3) {
        s.insert(static_cast<std::string::size_type>(i), 1, ',');
    }
    return s;
}

/* ============================================================
   ATTEMPTS as a variable template (default = int)
   Usage: ATTEMPTS<>, ATTEMPTS<float>, ATTEMPTS<double>, ...
   ============================================================ */
template<typename T = int>
inline constexpr T ATTEMPTS = static_cast<T>(1'000'000);

/* ============================================================
   Helpers
   ============================================================ */
using MoveFn = void(*)(B1B2&);

static inline void apply_move_fn(MoveFn fn, Board& b) {
    fn(reinterpret_cast<B1B2&>(b));
}

static void print_rule(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) std::cout << '-';
    std::cout << "\n";
}

struct BenchResult {
    std::string name;
    double seconds = 0.0;       // total measured time
    double per_sec = 0.0;       // ATTEMPTS / seconds
    long long sum = 0;          // accumulated eval value (to keep optimizer honest)
};

/* Single benchmark runner.
   - eval_func must be callable as int(const Board&)
   - Does ATTEMPTS loops, each: C_1_1(temp); sum += eval(temp);
*/
template<class EvalCallable>
static BenchResult bench(const std::string& name, const Board& goal, EvalCallable&& eval_func) {
    Board temp = goal;
    const Timer t0;
    long long sum = 0;
    for (int i = 0; i < ATTEMPTS<>; ++i) {
        C_1_1(temp);
        sum += static_cast<long long>(eval_func(temp));
    }
    const double seconds = t0.getSeconds();
    const double per_sec = ATTEMPTS<double> / seconds;
    return BenchResult{ name, seconds, per_sec, sum };
}

int main(const int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    const char* id = (argc >= 2 ? argv[1] : "7-1");
    const Timer total_timer;

    const auto pair = BoardLookup::getBoardPair(id);
    Board start = pair->getStartState();
    const Board goal = pair->getEndState();
    const B1B2 goal_b1b2 = goal;

    // Heuristics
    const auto H1 = lband_heuristic::LHeuristic::fromGoalBoard(goal);
    const auto H2 = better_heur::Heuristic::fromGoalBoard(goal);

    /* ========================================================
       Eval registry (for printing step-by-step rows)
       ======================================================== */
    struct EvalEntry {
        std::string label;                       // short column label
        bool printable;                          // whether to include in the step table
        std::string title;                       // long name for benchmarks
        std::function<int(const Board&)> func;   // eval function
    };

    const std::vector<EvalEntry> evals = {
        // Baseline placeholder (not printed in step table)
        {"H0", false, "Baseline (loop + call, no-op eval)", [&](const Board&){ return 0; }},
        {"H4", true,  "Original getScore1", [&](const Board& b){ return 36 - b.getScore1(goal); }},
        {"H3", true,  "Original getScore3", [&](const Board& b){ return b.getScore3(goal_b1b2); }},
        {"H2", true,  "Better Heuristic",   [&](const Board& b){ return H2.eval(b); }},
        {"H1", true,  "LBand Heuristic",    [&](const Board& b){ return H1.eval(b); }},
    };

    /* ========================================================
       Baseline-subtracted benchmarks
       ======================================================== */
    std::cout << "\n=== Heuristic Benchmarks (net of baseline) ===\n";

    // 1) Measure baseline ONCE (same loop, same C_1_1, same call overhead, no-op eval)
    const auto base = bench("Baseline (loop + call, no-op eval)", goal,
                            [](const Board&){ return 0; });

    // 2) Run each heuristic and print NET throughput (PerSec after subtracting baseline time)
    auto print_bench_row = [&](const BenchResult& r, double baseline_seconds) {
        double net_seconds = r.seconds - baseline_seconds;
        if (net_seconds < 1e-9) net_seconds = 1e-9; // guard
        const double net_per_sec = ATTEMPTS<double> / net_seconds;

        std::cout
            << std::left  << std::setw(20) << r.name
            << " | Net/s: " << std::right << std::setw(10) << with_commas(static_cast<int>(net_per_sec))
            << " | sum: "   << std::right << std::setw(10) << with_commas(r.sum)
            << "\n"
            << std::left; // reset for next row's name
    };



    for (const auto& e : evals) {
        if (e.title.rfind("Baseline", 0) == 0) continue; // skip printing baseline row
        const auto r = bench(e.title, goal, e.func);
        print_bench_row(r, base.seconds);
    }
    print_rule(74);

    /* ========================================================
       Step-by-step eval table (easy to extend)
       ======================================================== */

    // Define a static list of moves as {name, function-pointer}
    static constexpr std::array<std::pair<const char*, MoveFn>, 11> test_moves = {{
        {"C_0_3", C_0_3},
        {"C_1_4", C_1_4},
        {"C_5_1", C_5_1},
        {"R_4_5", R_4_5},
        {"C_2_3", C_2_3},
        {"R_4_3", R_4_3},
        {"C_0_2", C_0_2},
        {"R_3_3", R_3_3},
        {"R_5_5", R_5_5},
        {"C_3_5", C_3_5},
        {"C_4_3", C_4_3},
    }};

    const int move_w = 8;
    const int col_w  = 5;

    std::cout << "\n=== Step-by-Step Eval Table ===\n";
    std::cout << std::left << std::setw(move_w) << "Move";
    for (const auto& e : evals) {
        if (e.printable) std::cout << " | " << std::setw(col_w) << e.label;
    }
    std::cout << "\n";
    print_rule(move_w + static_cast<int>(evals.size()) * (3 + col_w));

    for (const auto& [name, fn] : test_moves) {
        apply_move_fn(fn, start);
        std::cout << std::left << std::setw(move_w) << name;
        for (const auto& e : evals) {
            if (e.printable) {
                std::cout << " | " << std::setw(col_w) << e.func(start);
            }
        }
        std::cout << "\n";
    }

    // Final board readout
    std::cout << "\nFinal State (vs goal):\n" << start.toString(goal) << "\n";

    // Tail metrics
    std::cout << "New Calls: " << new_calls << "\n";
    std::cout << "Total Time: " << total_timer.getSeconds() << "s\n";
    return 0;
}
