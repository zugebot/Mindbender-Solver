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

"No longer compiles" was checked with the Release build on 2026-10-12 before
the move (MinGW g++, C++23).
