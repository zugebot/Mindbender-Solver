# Mindbender-Solver

Finds every shortest solution to the 100 Mind Bender puzzles from PopCap's *Chuzzle* (6x6 boards
whose rows and columns shift cyclically, some with a 2x2 piece), and proves that nothing shorter
exists.

**Status:** 87/100 solved optimally, up to 16 moves. Every other puzzle has a proven lower bound
(`build/README.md`). The answers, with every shortest solution, are in `build/levels_final/`.

## Project history

- **January 2024 to July 2026, without AI assistance** (84 commits; tag `pre-ai`, commit
  `2ff1fb8`): the board stored as two 64-bit integers, row and column shifts with BMI2
  `pext`/`pdep`, the meet-in-the-middle solver, support for the 2x2 piece, parallel and radix
  sorts and memory work for the frontiers, and a CUDA experiment (branch `gpu-impl`).
  64/100 puzzles solved, up to 13 moves.
- **October 2026, AI-assisted** (with Claude; every such commit is marked `Co-Authored-By`): an
  exact-verification pass over every result, lower-bound pruning, the 14-16-move search design,
  a job scheduler, and a browser (WebAssembly) version. 87/100 solved, with proven lower bounds
  for the other 13.

The code as it was before any AI involvement: `git checkout pre-ai`, or the `pre-ai` tag on
GitHub.

## More

- An outline of the codebase:
  https://docs.google.com/document/d/17agxdvm6SnTsG4v5XrAdCslnxOafnWyb4Zu2hGxkrWg
- `build/README.md`: the unsolved puzzles and their bounds.
- `docs/changes-*.md`: what changed and why, from October 2026 on.
