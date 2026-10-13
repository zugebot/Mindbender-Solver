// src/programs/web/wasmSolver.cpp
//
// WebAssembly entry points for the browser solver (web/solver-worker.js calls
// them; web/build_wasm.ps1 builds web/mb_solver.js). One solver per worker:
//   mb_setup -> mb_solve_small (0-6 moves) -> mb_build_goal4 -> mb_build_start4
//   -> per depth 7-10: mb_begin_depth, mb_step until done, mb_finish_depth.
// Solutions come back as text, one per line, 1-based moves (R14 C453).

#include "src/solver/web/web_solver.hpp"

#include <emscripten/emscripten.h>

#include <string>

static web_solver::Solver g_solver;
static std::string g_text;

extern "C" {

// Returns a pointer to "" on success or to the reason the boards can't be solved.
EMSCRIPTEN_KEEPALIVE const char* mb_setup(const u8* startCells, const u8* goalCells, const int fat,
                                          const int sx, const int sy, const int gx, const int gy) {
    g_text = g_solver.setup(startCells, goalCells, fat != 0, sx, sy, gx, gy);
    return g_text.c_str();
}

EMSCRIPTEN_KEEPALIVE int mb_solve_small(const int maxDepth) { return g_solver.solveSmall(maxDepth); }

EMSCRIPTEN_KEEPALIVE void mb_build_goal4() { g_solver.buildGoal4(); }

EMSCRIPTEN_KEEPALIVE void mb_build_start4(const int part, const int parts) { g_solver.buildStart4(part, parts); }

EMSCRIPTEN_KEEPALIVE double mb_begin_depth(const int depth) { return static_cast<double>(g_solver.beginDepth(depth)); }

EMSCRIPTEN_KEEPALIVE int mb_step(const int maxSeeds) { return g_solver.step(static_cast<u64>(maxSeeds)) ? 1 : 0; }

EMSCRIPTEN_KEEPALIVE void mb_finish_depth() { g_solver.finishDepth(); }

EMSCRIPTEN_KEEPALIVE double mb_solution_count() { return static_cast<double>(g_solver.solutionCount); }

EMSCRIPTEN_KEEPALIVE double mb_invalid_count() { return static_cast<double>(g_solver.invalidSolutions); }

EMSCRIPTEN_KEEPALIVE double mb_streamed() { return static_cast<double>(g_solver.streamedStates); }

EMSCRIPTEN_KEEPALIVE double mb_layer_size(const int goalSide, const int k) {
    return static_cast<double>(g_solver.layerSize(goalSide != 0, static_cast<u32>(k)));
}

EMSCRIPTEN_KEEPALIVE double mb_memory_bytes() { return static_cast<double>(g_solver.memoryBytes()); }

EMSCRIPTEN_KEEPALIVE const char* mb_solutions_text() {
    g_text = g_solver.solutionsText(false);
    return g_text.c_str();
}

} // extern "C"
