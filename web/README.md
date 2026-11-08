# Mind Bender Solver (web)

A static page that solves Chuzzle Mind Bender boards in the browser. The default is a custom
puzzle: paint the goal (normal or with the 2x2 piece), then make the start, the scrambled board,
by dragging its Chuzzles (drop one on another and they swap; tap two to swap; drag the 2x2) or
with Mix. The start is only ever rearranged, never painted, so it always holds the goal's
Chuzzles: repainting the goal repaints them on the start too. Solve gives every shortest
solution of up to 12 moves, with a step-by-step replay. A three-step tutorial in the info bar
shows the way on a first visit (Help: Show the tutorial again). The game's own 100 puzzles are
under Game puzzles: there the boards are fixed and View replays every shortest solution, loaded
from `solutions/` (all 100 are solved). The goal is on the left and the start on the right, as in
the game. Nothing is uploaded, and there is no server
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
`mb_solver.js`, `puzzles.js`, `font.js`, `solutions/` and `assets/`. No special headers are needed, because the workers share
nothing (see below).

## Files

| File | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | the page: board editor, presets, solve orchestration, results and replay |
| `solver-worker.js` | one solver per Web Worker; talks to `app.js` by messages (the page fetches it and `mb_solver.js` once and starts every worker from one blob, so 16 workers cost no extra downloads; one worker is kept warm between solves) |
| `mb_solver.js` | the solver as WebAssembly, embedded in one JS file (built, but committed so the folder is deployable) |
| `puzzles.js` | the game's 100 boards with their known optimum or lower bound (generated) |
| `solutions/` | `<puzzle>.txt`: every shortest solution of each of the game's 100 puzzles, 1-indexed (generated from `build/levels_final/`, 5.6 MB; one is loaded when its puzzle is viewed) |
| `build_wasm.ps1` | rebuilds `mb_solver.js` with Emscripten (`emcc` on PATH) |
| `assets/` | the Chuzzle sprites, eyes, rainbow board frame and backdrop tiles (generated; not in git: they are the game's copyrighted art, so run `gen_sprites.py` with the textures from a Chuzzle install in `python/deluxe_clone/textures/`) |
| `font.js` | the game's Bauhaus 93 bitmap-font metrics (generated with `assets/bauhaus36.webp`) |
| `serve.py` | local server with caching off |
| `gen_sprites.py` | regenerates `assets/` from the game textures in `python/deluxe_clone/textures/`, the way `python/deluxe_clone/engine/textures.py` builds them: grey body x colour + 60 x highlight, cut by the alpha mask (yellow has its own sheet), then eyes and pupils |
| `gen_puzzles.py` | regenerates `puzzles.js` and `solutions/` from `src/chuzzles/levels*.hpp`, `build/levels_final/`, `build/README.md` (run it after a new solve) |
| `tools/` | CPU measurement: `cpu_measure.mjs` (headless Chrome, CDP) and `ff_eval.mjs` (headless Firefox, WebDriver BiDi) load the served page, optionally run a scene script (`t_scrub.js` a lane sliding every frame, `t_dragloop.js` a chuzzle dragged round), then report the browser's CPU over N seconds as a share of one core; `JS=...` flips the `MB_DEBUG` switches first, `PROF=<file>` records a Gecko profile of the Firefox run; `cdp_eval.mjs` runs a scene in headless Chrome and screenshots it (`--shot`; `WIN=`, `EXTRA=` set the window and other Chrome flags), with `t_stars.js` (the outline on a hovered goal cell, `?m=drag` a 2x2 dragged, `?m=gone` the marks taken away), `t_dir.js` (a replay's first move: a 2x2 lane's direction) and `t_thumb.js` (the tabs' little boards) |

The solver itself is `src/solver/web/web_solver.hpp` (plain C++, no BMI2, no threads), with
the WebAssembly entry points in `src/programs/web/wasmSolver.cpp`.

## How it searches

- Exact distance layers from both ends: S_0..S_4 from the start, G_0..G_4 from the goal
  (layer k = neighbours of layer k-1 that are in neither k-1 nor k-2).
- 0-6 moves: one worker meets layers 0-3 (S_a against G_b, a + b = D).
- 7-12 moves: every worker builds G_4 and its own hash partition of S_4. 7 and 8 moves meet
  S_4 with G_3 / G_4; 9 to 12 moves stream each S_4 state m = 1 to 4 more moves into G_4. Each
  shortest path crosses S_4 exactly once, so the partitions split the work with no shared
  memory, and their solution lists just concatenate.
- The stream visits each state once per canonical order: a run of row moves (or column
  moves) by ascending lane, never one lane twice in a row, each lane stopping at its period
  (the first shift that gives it back).
- On normal boards the rotation lane bound (as in the desktop solver) prunes every level of
  the stream: a state with r moves still to go is dropped when its bound exceeds r. A row
  move never changes a row's term (each is the same for all of a lane's rotations), so a
  row-move child only recomputes the columns' half, and when a node's rows' half alone is
  over, none of its row-move children is made. A 2x2 bound (ceil(R / 2) + ceil(C / 2)) was
  tried: it is exact but rarely passes 4, and pruned 0.5% of 6-5 at 10 moves.
- The last two moves are class lookups (the desktop's free-block join, `--block`): a row
  move leaves every row's class (its smallest rotation) as it was, so the goal-layer states
  one or two row moves from a state share its row class key and differ from it in just those
  rows. G_4 is indexed by row class and by column class (a counting sort into buckets, 9 bytes
  a state each), so one lookup finds every row-move finish, one every column-move finish; the
  2x2's own moves change its place and are made one by one. Each worker builds the indexes;
  the page drops them when they would leave room for less than half as many workers.
- A hit is a pair (s in S_4, y in G_4). All shortest paths are listed, once each: the paths
  start -> s (walking the S layers back), every m-move path s -> y (half from each end,
  joined), and the paths y -> goal (walking the G layers forward). Any D-move path is a
  shortest one when nothing shorter exists, so each is kept; every one is replayed against
  the boards.
- One search at a time, and it belongs to the boards it started on: other puzzles can be opened,
  viewed and edited meanwhile. The orb is then the shown puzzle's (View works; Solve waits for
  the search), the status line follows the search ("show it" goes back), and a search that ends
  on another puzzle keeps its result until its puzzle is shown again.
- Worker count = the thread count, capped so that the workers' tables fit a RAM budget
  (1.6 GB on desktops, 2.4 GB with 12 or more threads, less where `navigator.deviceMemory`
  is small; a worker's heap is about 1.5 times what its tables hold).

## Checking it

`testWebSolver` (CMake target, run from the repo root) runs the same C++ on every solved
puzzle up to `--max` moves and compares the solution set with `build/levels_final/`
(`--join 0` turns the class lookups off):

```bash
build/testWebSolver.exe --max 12 --parts 3
```

On 2026-11-08 all 45 puzzles of 10 moves or fewer matched exactly split into 3 partitions as
the workers do, and so did every 11-move puzzle and the 12-move 9-2, 11-1, 14-2 and 13-1.
`tools/wasm_bench.mjs` runs the WebAssembly build itself in Node and checks the counts;
`tools/t_background.js` (with `cdp_eval.mjs`) starts a search, views 16-4 meanwhile and goes back.

One native thread on the i7-10700F, the whole search (every depth up to the answer):

| Puzzle | Moves | Before 2026-11-08 | Now |
|---|---|---|---|
| 8-1 | 10 | 13 s | 4.1 s |
| 6-5 (2x2) | 10 | 51 s | 21 s |
| 15-5 | 11 | - | 29 s |
| 7-5 | 11 | - | 197 s (477 s without the class lookups) |
| 9-2 | 12 | - | 27 s before the class lookups |
| 14-2 | 12 | - | 74 s before the class lookups |
| 13-1 | 12 | - | 1347 s |

In the page (Chrome, 16 threads, a quiet machine): 4-2 (2x2, 9 moves) 0.8 s; 8-1 (10) 8.1 s on
11 workers; 6-5 (2x2, 10) 8.2 s (19 s before); 11-2 (11) 4.0 s; 15-5 (11) 10 s; 7-5 (11) 46 s
on 10 workers; 9-2 (12) 7.5 s (Firefox 7.9 s); 14-2 (12) 20 s on 11 workers. 12 moves depends on
how much the bound prunes: 13-1 (four colours, one of them 16 cells) would take minutes.
2x2 boards at 12 moves are out of reach: nothing prunes them, and each move costs ~33x.

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
Chuzzle's 179-unit sprite at (-9.5, -8.5), the 535-unit border 26 / 31 units out from the cells. Two
boards side by side when that gives at least 3/4 of the size of one, else one with Start / Goal tabs;
on short wide screens the dock moves to a column on the right. The solutions list is below the app.
Each Chuzzle's look is random, as in the game's `Chuzzle.cpp`: eye shape, pupil colour, the face
0-2 pixels lower, one of three bounce speeds; the face bobs 1 pixel with the bounce (2 on the 2x2).
Eyes follow `engine/chuzzle.py`: idle looks every 5-14 s, blinks every 3-10 s (the lids in the
Chuzzle's own colour, as the game draws them), pupils turn to the mouse over their chuzzle (the
logo's and the palette's only while the mouse is inside their square). Clicking the logo or a
palette Chuzzle again and again makes it perform the game's antics (`Chuzzle.cpp`, frame by
frame at 100 updates a second): a poke, the oggle, a cough, a small sneeze, then the huge sneeze
that blows its fuzz off (bald for a while, the page shaking); the 2x2 squeezes, hurks with puffed
cheeks, and on the 11th click puts on its beauty show with stars, a shockwave and 200 tufts. A
right click pokes one (a squash and a note). The voices are the game's (`assets/sounds`). A
picked Chuzzle, a drop's target and the spots it empties, and what a click on the goal would
paint (the whole 2x2 footprint with the 2x2 tool) are marked with the game's match outline
(`Board::SparkleGroups`): a chain of fizzy stars in the Chuzzle's colour running clockwise round
them, 45 pixels out, dropping the points inside or crowded by a neighbour so it runs in and out
between them (round a 2x2 footprint a cloverleaf, the four cells in clockwise order; round the
2x2 Chuzzle itself a circle). Each mark's chain is a loop that its stars run round for as long as
the mark stands, a trail of some 27 fading stars behind the runner, every mark's loop at once. The palette item in use wears the game's hint circle (`Animator_Hint`:
`HintCircle`, black shadow 4 pixels off, yellow and red by turns). The Solve candy carries the
Scrambulator's two arrows, out and back as in the game's no-moves panel.
Text above the boards uses the game's Bauhaus 93 bitmap font (`textures/fonts`), drawn on canvases.

The controls are the game's too: the round buttons are `GameButton.png` (blue, purple under the
pointer, pink pressed) with white symbols in a dark outline, labelled in Bauhaus 93; Solve is the
Scrambulator's candy wrapper (`ScrambulatorButtonBG.png`) holding a button that fills from the
bottom while the solver runs; the move limit is a - / + stepper; moves, drop-downs and the help
dialog use the navy-and-gold of the game's `DialogBox.png`.

Smoothness and idle cost: every Chuzzle on the page is drawn from the game's sprite sheets onto a
canvas (`drawChuzzle`): each board's on one canvas, the logo's and the palette's on a small one
each, the held one on a small canvas of its own, the particles on a layer over the page. Every
sprite is cut from its sheet once at its screen size and kept as an ImageBitmap (`cut`), so a
Chuzzle is a few 1:1 blits (a scaled draw from a 4800-pixel sheet costs 9-10x a blit, and Firefox
drew most of its CPU there). A Chuzzle is redrawn only when its pose changes (a body frame every
~87 ms, a blink, the pupils moving), only its own cell is cleared and drawn (the 2x2 with the
ring of cells its sprite reaches into), and while nothing but the idle sway is going on, every
canvas waits for the same 80 ms tick and the frame loop rests between ticks, so the page composes
at most 12 frames a second instead of one per screen refresh (in Firefox the composite, not the
drawing, is what an idle page costs: headless, 46% of a core at 25 composites a second, 29% at 10).
Nothing on the page animates endlessly in CSS, and no CSS filter wraps a canvas that changes (the
palette's shadows are baked into the canvases, the Scrambulator arrows sit beside the candy, not
inside its drop-shadow), because a filter is redone every time what is under it changes. The held
Chuzzle is drawn once per pose onto its own small canvas and moved by a CSS transform on every raw
pointer event (`pointerrawupdate`), so a drag costs the compositor a shifted texture, not a
full-window canvas cleared and composed per mouse event. The cells are see-through elements over
the canvas that take the pointer.
The panels are solid (a see-through blur over the backdrop made every repaint slow), the board size
is snapped so 20 game pixels are a whole number of screen pixels (every sprite frame starts on a
pixel, so the Chuzzles sway in place instead of jittering), and the info bar, status line and Solve
row have fixed sizes, so messages coming and going never move the boards.
