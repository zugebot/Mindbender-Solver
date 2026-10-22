# Mindbender-Solver

Finds every shortest solution to the 100 Mind Bender puzzles from PopCap's *Chuzzle* (6x6 boards
whose rows and columns shift cyclically, some with a 2x2 piece), and proves that nothing shorter
exists.

**Status:** 87/100 solved optimally, up to 16 moves. Every other puzzle has a proven lower bound
(`build/README.md`). The answers, with every shortest solution, are in `build/levels_final/`.

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
  design, a job scheduler, and a browser (WebAssembly) version. 87/100 solved, with proven lower
  bounds for the other 13.

The code as of July 2025, before the clearly AI-assisted files above: `git checkout pre-ai`, or
the `pre-ai` tag on GitHub.

## More

- An outline of the codebase:
  https://docs.google.com/document/d/17agxdvm6SnTsG4v5XrAdCslnxOafnWyb4Zu2hGxkrWg
- `build/README.md`: the unsolved puzzles and their bounds.
- `docs/changes-*.md`: what changed and why, from October 2026 on.
