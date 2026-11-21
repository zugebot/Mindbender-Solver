// wasm5x5.cpp - the web page's 5x5 solver: solver5x5.hpp as WebAssembly (build_wasm.ps1 here
// writes web/mb5_solver.js). One call solves a puzzle; the results stay until the next call.
//
//   mb5_solve(start, goal, maxDepth, listLimit): 25 colour bytes each; the shortest length, or
//       -1 with none in maxDepth moves, -2 if the boards hold different chuzzles
//   mb5_count(): the number of shortest solutions (a double: exact below 2^53)
//   mb5_text(): up to listLimit of them, one per line, moves 1-indexed ("R14 C23\n")
//   mb5_states(), mb5_bytes(): the states searched and the tables' memory

#include "solver5x5.hpp"

#include <emscripten/emscripten.h>

namespace {
mb5::Result last;
std::string text;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE int mb5_solve(const unsigned char* start, const unsigned char* goal, int maxDepth, int listLimit) {
    const mb5::u128 s = mb5::fromCells(start), g = mb5::fromCells(goal);
    text.clear();
    last = mb5::Result{};
    if (!mb5::sameColourCounts(s, g)) return -2;
    last = mb5::solve(s, g, maxDepth, size_t(listLimit > 0 ? listLimit : 1));
    for (const auto& sol : last.listed) {
        if (!mb5::replay(s, g, sol)) continue;        // never happens; listed are walked out exactly
        for (size_t i = 0; i < sol.size(); ++i) {
            if (i) text += ' ';
            text += mb5::moveName(sol[i]);
        }
        text += '\n';
    }
    return last.length;
}

EMSCRIPTEN_KEEPALIVE double mb5_count() { return double(last.solutions); }
EMSCRIPTEN_KEEPALIVE const char* mb5_text() { return text.c_str(); }
EMSCRIPTEN_KEEPALIVE double mb5_states() { return double(last.states); }
EMSCRIPTEN_KEEPALIVE double mb5_bytes() { return double(last.bytes); }

}
