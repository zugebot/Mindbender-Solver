# chuzzle2_5x5: an optimal solver for 5x5 Mind Bender boards

Chuzzle 2 has 14 Mind Bender puzzles on 5x5 boards (`puzzles_5x5.txt`, taken from the puzzle
list in `Chuzzle2.exe`; kept local, not committed). This folder solves them on its own: the
solver is `solver5x5.hpp`, used by the command-line tool (`solve5x5.cpp`, `build.cmd`) and by
the web page (`wasm5x5.cpp`, `build_wasm.ps1` -> `web/mb5_solver.js`). Nothing from `src/`, not
part of the CMake build, so the 6x6 solvers are untouched.

## Build and run

```bat
build.cmd
solve5x5 --check
solve5x5 --goal "11011 11011 00000 11011 11011" --start "10011 11110 11010 11011 10001"
```

| Option | What it does |
|---|---|
| `--file F` | the puzzle list (default `puzzles_5x5.txt`: number, par, five goal rows) |
| `--scramble N` | start boards made with N random moves (default: each puzzle's par) |
| `--seed N` | the random scramble's seed (default 1) |
| `--goal` / `--start` | one puzzle of your own, 25 digits each (spaces allowed) |
| `--list K` | show K shortest solutions (default 3) |
| `--check` | redo each puzzle with a plain one-sided search and compare length and count |
| `--check-max D` | skip the check past D moves (default 6; 7 needs too much memory) |
| `--max D` | give up past D moves (default 14) |

Moves are 1-indexed as in the main project: `R14` = row 1 right 4, `C23` = column 2 down 3. A
digit is a colour; 9 is the rainbow chuzzle, just a colour no other cell has.

## On the web page

`build_wasm.ps1` (Emscripten, `emcc` on PATH) writes `web/mb5_solver.js` (46 KB). The page's
solver workers load it beside the 6x6 solver; the size tool in the custom editor switches the
boards to 5x5, and Solve finds and lists every shortest solution in one call (puzzle 23
scrambled with 30 moves: 8 moves, 28 solutions, 0.3 s, the same as `solve5x5`).

## How it works

A board is 25 cells of 4 bits in one 128-bit number; a move turns a row right or a column down
by 1-4 (40 moves). The search grows layers from the start and from the goal in turn (the
smaller side next). Each state keeps its distance and how many shortest move sequences reach
it. The first new layer that touches the other side gives the optimal length; every shortest
solution passes exactly one state of that layer, so the solution count is the sum over those
states of (sequences from the start) x (sequences to the goal). Each listed solution is
replayed on the start board before it is printed.

## Results (2026-11-21, i7-10700F)

The game's start boards are not known yet, so the starts here are random scrambles.

- Each goal scrambled with its par in moves: all 14 solved in under 0.3 s each; the
  one-sided check agrees on length and count for the 13 that are 6 moves or fewer.
- Each goal scrambled with 20 moves: the hardest needs 9 moves (puzzles 7, 15, 16, 23),
  at most 5.6 M states and 2.4 s.
