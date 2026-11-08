# Mindbender-Solver

Finds every shortest solution to the 100 Mind Bender puzzles from PopCap's *Chuzzle* (6x6 boards
whose rows and columns shift cyclically, some with a 2x2 piece), and proves that nothing shorter
exists.

**Status:** 100/100 solved optimally (2026-11-07), up to 18 moves. The answers, with every
shortest solution, are in `build/levels_final/`. The 22 placements of the 2x2 piece that the
game also accepts (the piece on another 2x2 block of its colour) are solved too
(`build/levels_final/game_accepts/`).

## Project history

- **January 2024 to July 2025** (64 commits; tag `pre-ai`, commit `cc10f0c`): written by me: the
  board stored as two 64-bit integers, row and column shifts with BMI2 `pext`/`pdep`, the
  meet-in-the-middle solver, support for the 2x2 piece, and parallel sorting for the frontiers.
  A few side files from September 2024 may have had ChatGPT help: the thread-sort helpers
  (`th_parallel_sort.hpp`, `th_radix_sort.hpp`), `testHeuristics.cpp`, and the scripts
  `append_size.py`, `remove_end.py` and `mask_maker.py`.
- **March to July 2026** (20 commits): my work, with ChatGPT help in some files: an experimental
  heuristic (`src/idk/`), the rotation tests (`testRotations.cpp`), the ML experiment
  (`src/ml_torch/`) and a mouse-path script (`archive/chuzzle_mouse_dp.py`); also a CUDA
  prototype. 64/100 puzzles solved by July 2026, up to 13 moves.
- **October 2026 on, AI-assisted** (with Claude; every such commit is marked `Co-Authored-By`):
  an exact-verification pass over every result, lower-bound pruning, the 14-16-move search
  design, a job scheduler, and a browser (WebAssembly) version. 87/100 solved by late October,
  with proven lower bounds for the other 13.
- **November 2026** (with Claude): a CUDA version of the large solver's inner loop for the
  RTX 5060 Ti (`src/gpu/`): a 15-move search on puzzle 16-2 went from 42 min to 54 s, and on a
  board with the 2x2 piece (18-4) from 10 h to 25 min (`docs/changes-2026-11-01.md`, `-11-02.md`).
  91/100 solved; the other 9 need at least 16 moves (the 2x2 ones) or 17.
- **November 2026, later** (with Claude): every 2x2 board searched through 16 moves on the GPU
  (6-4, 15-4, 16-1 and 19-4 solved, 7-11 h each), 17-move searches for the normal boards, a
  Vulkan version of the GPU code (any vendor, Windows and Linux) and several GPUs per run. On a
  friend's PC with three AMD Radeon Pro GPUs, 18-3 and 14-3 were solved at 17 moves. 97/100
  solved (`docs/changes-2026-11-03.md`, `-11-04.md`).
- **November 5-7, 2026** (with Claude): the last three. 19-2 was solved at 17 moves on the
  RTX 5060 Ti (9 h 50 min, with the free-block join `--block`, which saves one move on the
  start side). 15-3 and 16-3 needed 18 moves: each search was split into work units and run
  on up to nine rented vast.ai RTX 5090s (`mbq fleet`; about 18 hours of machines in all,
  $131.54), 15-3 in 13 h 37 min of wall time (76 units) and 16-3 in 31.5 machine-hours
  (30 units), finishing on 2026-11-07. The 22 game-accepted placements of the 2x2 piece were
  solved the same week. 100/100 solved (`docs/changes-2026-11-05-07.md`).

## Results

Shortest solution length over the 100 puzzles:

| Moves | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Puzzles | 1 | 2 | 5 | 5 | 1 | 2 | 4 | 10 | 6 | 9 | 10 | 9 | 5 | 8 | 8 | 10 | 3 | 2 |

- 18 moves: 15-3 (4384 solutions; 2x2 piece) and 16-3 (3524; a normal board, searched in
  reverse, from the goal).
- 17 moves: 14-3 (1790 solutions), 18-3 (896) and 19-2 (3960).
- 16 moves: 6-4, 12-1, 15-1, 15-4, 16-1, 17-3, 19-3, 19-4, 20-2, 20-5.
- Most solutions: 4-5, 57 663 shortest solutions of 9 moves. Only 1-1 and 1-3 have a single one.

Up to 13 moves the CPU was enough; 14-16 moves needed the lower-bound pruning (20-5 at 16 moves
took 36.5 h on the CPU), the 2x2 boards at 16 and every 17-move search needed the GPU (hours
per puzzle instead of days to weeks), and the two 18-move searches needed several GPUs at once
(a 17-move search on a 2x2 board streams ~1e16 states; one more move costs ~30x).

## How the large searches work

A puzzle of N moves is split into a start side and a goal side that meet in the middle. The goal
side's states (up to 7 moves from the goal, tens of billions) are indexed by fingerprint; the
start side is generated as exact depth-5 seeds, each streamed 4-6 more moves on the GPU, and every
state is tested against the goal side (a Bloom filter on the GPU, the fingerprint index on the
CPU). Hits are confirmed, and every shortest solution is rebuilt from the middle states they
pass through. On boards without the 2x2 piece a lower bound (per row and column, the fewest cells
wrong over the lane's rotations) prunes almost everything; on 2x2 boards it prunes nothing, so
those searches are much larger. When the goal side does not fit in memory, the search runs in
passes over parts of it. `CLAUDE.md` and `docs/changes-*.md` have the details and measurements.

The code as of July 2025, before the clearly AI-assisted files above: `git checkout pre-ai`, or
the `pre-ai` tag on GitHub.

## More

- An outline of the codebase:
  https://docs.google.com/document/d/17agxdvm6SnTsG4v5XrAdCslnxOafnWyb4Zu2hGxkrWg
- `build/README.md`: the per-puzzle status, kept up to date by the scheduler (now: none unsolved),
  and the game-accepted placements.
- `docs/changes-*.md`: what changed and why, from October 2026 on.
- `src/gpu/README.md`: the GPU solver (design, checks, measurements).
