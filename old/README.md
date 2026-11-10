# old/

Code kept for history. Nothing here is built (CMake does not list it) and
nothing outside this folder includes it. Paths mirror where each file used to
live, so `git log --follow old/<path>` shows its full history.

Moved here on 2026-10-12, when `src/` was reorganized:

| File | Why |
|---|---|
| `src/programs/UnitTesting.cpp` | not built; uses `Board::getHash`, which no longer exists |
| `src/programs/test11MoveSolution.cpp` | not built; uses `Board::getHash` |
| `src/programs/test12MoveHeuristicAtHalf.cpp` | not built; calls a removed `applyMove` |
| `src/programs/testDFS.cpp` | not built; uses `B1B2::getScore3Till` and other removed scoring methods |
| `src/programs/testHeuristics.cpp` | not built; ambiguous `getColor` overloads since the board API changed |
| `src/programs/testPerms.cpp` | not built; needs a removed `moveVectorToDirectString` |
| `src/programs/testDepth4Direct.cpp` | was built, no longer compiles (frontier builder signature changed) |
| `src/programs/testDepth4DirectFat.cpp` | was built, no longer compiles (`StateHash::refreshHashFunc` removed) |
| `src/programs/CFCC.cpp` | was built, no longer compiles (`perms.hpp`: "T must be Board or B1B2") |
| `src/programs/benchmarkMatchingBound.cpp` | was built, no longer compiles (`StateHash::refreshB1B2` removed) |
| `src/programs/debug_intersection_4_4.cpp` | was built, no longer compiles (`StateHash::refreshHashFunc` removed) |
| `src/programs/testHeuristicsOnLeftSeed.cpp` | was built, no longer compiles (`StateHash::refreshHashFunc` removed) |
| `src/heuristics/heuristics.hpp` | only used by `testHeuristicsOnLeftSeed.cpp` |
| `src/idk/` | scratch mains and heuristics, not referenced anywhere |
| `src/solver/frontier_right_index.hpp` | not included anywhere; replaced by `frontier_right_index2.hpp` |
| `src/programs/experiments/experimentBoundSample_research.cpp` | a copy of the bound sampling tool before its research modes were removed: the SWAR colour-mix bound, single-colour pattern databases (`--pdb`, `--pdb-check`) and the prototype of the grouped pattern databases (`--pdb2`; now `src/solver/bounds/pattern_db.hpp`). The live tool keeps `--check`, sampling, `--bench` (both lane-table kinds), `--deep` and `--colors` |

"No longer compiles" was checked with the Release build on 2026-10-12 before
the move (MinGW g++, C++23).

Added on 2026-11-10: `early_versions/`, the solvers from before this
repository (Java 2023 to January 2024, the 2x2 move-table prototype, the 2023
C++ port), copied from backups. Its README lists where each came from, how
they worked, and how their answers compare with the proven ones.

Moved here on 2026-11-02:

| File | Why |
|---|---|
| `src/gpu/vulkan/shaders/emit.glsl` | the Vulkan node kernels' first child handling (5 child boards per lane in registers, 20-byte staged copies); replaced when `expand.comp` / `expand_fat.comp` were laid out for AMD GCN (masks rebuilt from the parent, 4-byte staging): 128 -> 64 / 40 VGPRs, 16-2 bench 8.04 s -> 4.43 s on a Radeon Pro WX 7100 |
