# Mind Bender Solver (web)

A static page that solves Chuzzle Mind Bender boards in the browser: paint a start and a
goal board (normal or with the 2x2 piece), press Solve, and get every shortest solution
of up to 10 moves, with a step-by-step replay. Nothing is uploaded, and there is no server
code: the solver is C++ compiled to WebAssembly and runs in Web Workers.

## Run it locally

Browsers block workers on `file://` pages, so serve the folder (`serve.py` turns caching off, so a
reload never mixes old CSS with new sprite sheets):

```bash
python web/serve.py
```

Then open http://localhost:8770/.

## Host it

Copy the `web/` folder's files to any static host (GitHub Pages, Netlify, Cloudflare Pages,
or a plain web server): `index.html`, `style.css`, `app.js`, `solver-worker.js`,
`mb_solver.js`, `puzzles.js` and `assets/`. No special headers are needed, because the workers share
nothing (see below).

## Files

| File | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | the page: board editor, presets, solve orchestration, results and replay |
| `solver-worker.js` | one solver per Web Worker; talks to `app.js` by messages (the page fetches it and `mb_solver.js` once and starts every worker from one blob, so 16 workers cost no extra downloads; one worker is kept warm between solves) |
| `mb_solver.js` | the solver as WebAssembly, embedded in one JS file (built, but committed so the folder is deployable) |
| `puzzles.js` | the game's 100 boards with their known optimum or lower bound (generated) |
| `build_wasm.ps1` | rebuilds `mb_solver.js` with Emscripten (`emcc` on PATH) |
| `assets/` | the Chuzzle sprites, eyes, rainbow board frame and backdrop tiles (generated; not in git: they are the game's copyrighted art, so run `gen_sprites.py` with the textures from a Chuzzle install in `python/deluxe_clone/textures/`) |
| `font.js` | the game's Bauhaus 93 bitmap-font metrics (generated with `assets/bauhaus36.webp`) |
| `serve.py` | local server with caching off |
| `gen_sprites.py` | regenerates `assets/` from the game textures in `python/deluxe_clone/textures/`, the way `python/deluxe_clone/engine/textures.py` builds them: grey body x colour + 60 x highlight, cut by the alpha mask (yellow has its own sheet), then eyes and pupils |
| `gen_puzzles.py` | regenerates `puzzles.js` from `src/chuzzles/levels*.hpp`, `build/levels_final/`, `build/README.md` |

The solver itself is `src/solver/web/web_solver.hpp` (plain C++, no BMI2, no threads), with
the WebAssembly entry points in `src/programs/web/wasmSolver.cpp`.

## How it searches

- Exact distance layers from both ends: S_0..S_4 from the start, G_0..G_4 from the goal
  (layer k = neighbours of layer k-1 that are in neither k-1 nor k-2).
- 0-6 moves: one worker meets layers 0-3 (S_a against G_b, a + b = D).
- 7-10 moves: every worker builds G_4 and its own hash partition of S_4. 7 and 8 moves meet
  S_4 with G_3 / G_4; 9 and 10 moves stream each S_4 state 1 or 2 more moves into G_4. Each
  shortest path crosses S_4 exactly once, so the partitions split the work with no shared
  memory, and their solution lists just concatenate.
- All shortest paths are listed: the paths start -> s (walking the S layers back), the middle
  moves, and the paths y -> goal (walking the G layers forward). The 2-move middle stream
  visits one order of each commuting pair (two rows or two columns) and lists both. Every
  listed path is replayed against the boards.
- On normal boards the rotation lane bound (as in the desktop solver) prunes the middle stream.
- Worker count = the thread count, capped so that the workers' tables fit a RAM budget
  (about 1.6 GB on desktops, less where `navigator.deviceMemory` is small).

## Checking it

`testWebSolver` (CMake target, run from the repo root) runs the same C++ on every solved
puzzle up to 10 moves and compares the solution set with `build/levels_final/`:

```bash
build/testWebSolver.exe --max 10 --parts 3
```

On 2026-10-13 all 45 puzzles of 10 moves or fewer matched exactly (normal and 2x2, split into
3 partitions as the workers do).

Measured in Chrome on the i7-10700F (with a 16-thread solver job running alongside):
up to 8 moves under ~2 s; 4-2 (2x2, 9 moves) 1.2 s; 8-1 (10 moves) 11 s on 12 workers;
6-5 (2x2, 10 moves, the slowest kind: the bound is off on 2x2 boards) 19 s on 16 workers.

## Look

The boards use the game's own art: each ball is a Chuzzle sprite with the idle sway (20 frames, the
2x2 piece 10) and the game's stretched frame while a lane slides; eye shape and pupil colour stay with
each Chuzzle through a replay. The board sits in the game's rainbow frame, and the page background is
the backdrop pattern of the puzzle's world. The replay is guided: the game's wand cursor glides to a Chuzzle in the lane, presses, and drags
the lane the short way round (right 5 = drag left 1) while it slides along; between moves an arrow on
the board shows the next drag. The art is PopCap's (from the game files), so a public
host redistributes it.

## Layout

One layout for every screen: the app fills the window (header, info bar, boards, control dock) and
`app.js` measures the board area and sets `--u`, one pixel of the game's 800x600 screen. Boards are
drawn in the game's own coordinates (`python/deluxe_clone/objects/board.py`): 80-unit cells, the 2x2
Chuzzle's 179-unit sprite at (-10, -12), the 535-unit border 26 / 31 units out from the cells. Two
boards side by side when that gives at least 3/4 of the size of one, else one with Start / Goal tabs;
on short wide screens the dock moves to a column on the right. The solutions list is below the app.
Eyes follow `engine/chuzzle.py`: idle looks every 5-14 s, blinks every 3-10 s, pupils turn to the
mouse over their chuzzle, and drop half a pixel on sway frames 10-14. Text above the boards uses the
game's Bauhaus 93 bitmap font (`textures/fonts`), drawn on canvases.
