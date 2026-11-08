# Mindbender-Solver

Optimal solver for Chuzzle's Mind Bender puzzles (100 puzzles, 6x6 boards,
row/column cyclic shifts, some with a 2x2 piece). It finds every shortest
solution and proves nothing shorter exists. C++ solver plus Python tools.

Status (2026-11-07): **100/100 solved**, all on this PC (the i7; before its RAM and GPU upgrades for the early ones,
on the RTX 5060 Ti since 2026-11-01) except 18-3 and 14-3, solved on the friend's PC (Ryzen 5
5600X, three Polaris GPUs over Vulkan), and 15-3 and 16-3, solved on rented vast.ai RTX 5090s
(`mbq fleet`; ~18 h of up to nine machines, $131.54 in all, docs/changes-2026-11-05-07.md
section 10). The last: 16-3 at 18, 3524 solutions, `<5,6,6>` + `--block` reverse, 30 fleet units
(31.5 machine-hours), recovery split over 5 machines (2026-11-07 12:05 PM).
`docs/run_history.csv` starts in 2026-10, so the 61
earlier answers have no row there. Latest: 15-3 at 18, 4384 solutions, `<5,5,7>` + `--block`, 76 fleet
units on 9 rented 5090s (13 h 37 min wall, 2026-11-07; recovery split over 3 machines, ~1 h);
all 22 game-accepted placements solved (2026-11-07). Before: 19-2 at 17, 3960 solutions, `<5,4,7>` + `--block`, 5060 Ti, 9 h 50 min (2026-11-05); 15-3 has no 17-move solution (>= 18, 11 h 28 min, 2026-11-06). Earlier: 15-1, 19-3, 17-3, 12-1 at 16 moves on the 5060 Ti; 6-4 at 16, 68
solutions, `<5,4,7>`, 5060 Ti, 6 h 45 min; 15-4 at 16, 356 solutions, `<5,4,7>`, 5060 Ti,
9 h 15 min; 18-3 at 17, 896 solutions, `<5,6,6>`, friend's PC, 2 h 50 min; 16-1 at 16, 12
solutions, `<5,4,7>`, 5060 Ti, 8 h 45 min; 14-3 at 17, 1790 solutions, `<5,6,6>`, friend's PC,
3 h 39 min; 19-4 at 16, 40 solutions, `<5,4,7>`, 5060 Ti, 9 h 36 min (13 h 15 min with two
holds) (each run's machine: column `machine` of the run history);
16-3 has no 17-move solution (`<5,6,6>`, friend's PC, 16 h 22 min, 2026-11-03); solved at 18 on
the fleet 2026-11-07.
19-2 at 17 on the friend's PC was stopped 2026-11-04 (unit 0 of 80 unfinished after 4 h 17 min,
~1 month projected); the friend's PC is not available for long runs any more. `build/README.md` lists the unsolved ones with their
"at least N moves" bounds (updated by the scheduler); `build/levels_final/` holds every answer.
History of what changed and why: `docs/changes-*.md` (newest: `changes-2026-11-05-07.md`, 19-2 at 17
and 15-3 at 18 solved, all 22 placements solved, the rented fleet (`mbq fleet`), Linux container
limits, recovery at left depth 11/12 and split over machines, the GPU output pipe (off);
`changes-2026-11-04.md`, 19-4 solved,
a pass's cost measured, the GPU survivor check ~18% faster, a 2x2 split bound (cannot prune),
2x2 boards at 17 moves: `<5,5,7>`, `--d7-groups` work units, the Vulkan 2-s job limit; `changes-2026-11-03.md`, the Linux port, Vulkan on
AMD, several GPUs, the friend's-PC runs, game-accepted goals, Chuzzle Suite;
`changes-2026-11-02.md`, 2x2 boards on
the GPU and faster recovery; `changes-2026-11-01.md`, the GPU
solver and its speedups; `changes-2026-10-30.md`, the first-move rule, before it).

## Build and test

- Needs CMake >= 3.24 (system one is 3.23; use CLion's:
  `C:/Users/jerrin/AppData/Local/JetBrains/CLion 2022.3/bin/cmake/win/x64/bin/cmake.exe`),
  Ninja (CLion's), MinGW g++ at `C:/mingw64` (put `C:/mingw64/bin` on PATH), Release.
  The configured build dir is `cmake-build-release`:
  `cmake --build cmake-build-release --target <name> -j 8`. Executables land in `build/`
  (`-DMB_OUTPUT_DIR=<dir>` sends them elsewhere, e.g. for test builds that must not replace the
  exe the scheduler copies). Portable build for other PCs: `-DMB_MARCH=x86-64-v3
  -DMB_OUTPUT_DIR=<repo>/build_portable` (no AVX-512).
- Every program compiles the shared sources itself, so building all targets takes ~20 min.
- Tests: `testExactSolverRegression` (14 cases; `--recovery11 build/levels_final/19-3_c16_202.txt`
  checks recovery at left depth 11, as 17-move runs use it; `--recovery 10 <2x2 16-move file>`
  left depth 10 on 2x2 boards, as `<5,5,7>` uses it), `testRotations` (112), `testJVec`,
  `testKnownSolutions` (replays every file in `<repo>/levels`; run from `build/`),
  `testWebSolver` (the web solver vs `build/levels_final`, all puzzles up to 10 moves; run from
  the repo root),
  `experimentBoundSample --check` (bounds admissible on all 1.9 M known-optimal-path states; run
  from the repo root), `python -m unittest discover -s python/chuzzle_suite/tests`.
- Exactness check for solver changes: run the solver on a solved puzzle with
  `--solutions build/levels_final/<file>` and look for `EXACT MATCH` and `IDENTICAL SET`
  (9-3 and 10-3 at 13 moves take minutes; 16-4 at 14 is the big one, 519 solutions).
- A running exe is locked: copy it before long runs (the scheduler does).
- GPU module (`src/gpu/`): `src\gpu\build.cmd` (CUDA 13.0 nvcc + VS 2022 MSVC 14.44, sm_120)
  builds `build_gpu/mbgpu.dll` and `test_gpu_stream.exe`. Check: `exportGpuVectors` (CMake
  target) writes `build_gpu/vectors.bin`, then `build_gpu\test_gpu_stream.exe
  build_gpu\vectors.bin` must print PASS; `--bench` on `build_gpu/bench_16-2.bin` times the
  stream (`--seeds 200000 --filter-gb 3`: 0.74 s). The solver build with --gpu is
  `cmake-build-test` -> `build_test/` (the solver passes `mbgpu_stats` to the DLL: rebuild both
  when `mbgpu.h` changes). Profilers: Nsight Systems 2025.3.2, Nsight Compute 2025.3.0 (GPU
  counters allowed for all users).
- Vulkan module (`src/gpu/vulkan/`, any vendor): `src\gpu\vulkan\build.cmd` (Vulkan SDK
  1.4.341 glslc + VS 2022 MSVC) builds `build_gpu/mbvk.dll`, `test_vk_stream.exe` (the same
  test, `--device 5060|auto`) and `vk_devices.exe`. Same C interface; `--gpu` tries CUDA, then
  Vulkan (`cuda:` / `vulkan:` prefixes force one; `auto` picks devices; several Vulkan GPUs
  share the seeds). `--gpu-list` prints the devices. Built 2026-11-02, not yet run on a GPU:
  its README lists the checks.

## Everyday commands (cmd)

```bat
cd /d C:\Users\jerrin\CLionProjects\Mindbender-Solver
python python\chuzzle_suite\run_suite.py
python python\scheduler\mbq.py daemon
python python\scheduler\mbq.py resume
python python\scheduler\mbq.py status
```

`status` is also a live page at http://127.0.0.1:8766/ (Hold solver / Pause queue buttons).
At logon, `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\mindbender_startup.cmd`
runs the Parsec watchdog, the user's Discord bots (`E:\runner.bat`) and Chuzzle Suite. Since
2026-10-30 it no longer starts `mbq daemon` / `mbq resume` (lines commented out), and the queue
is paused: the solver runs only when the user asks. A job cut off by a reboot is re-queued first.

## Layout

- `src/chuzzles/`: board (`B1B2` = two u64), moves, lane shifts (`lane_rotations.hpp`, pext/pdep),
  canonical move streams (`perm_stream*`, with the optional PARENT_PRUNE hook), level data
  (`levels.hpp`, `levels_cells.hpp`).
- `src/solver/`:
  - `frontier_builder.hpp`, `sorter.hpp`: exact frontiers (shared by every solver).
  - `inram/`: the in-RAM meet-in-the-middle solver (`solver_base`, `solver_direct`,
    `solver_frontier` (exact recovery from middle states, `recoverSolutionsFromMiddles`; the SSD
    solver uses it too), `right_frontier_table`, `right_frontier_index`, `memory_perm_gen`).
  - `bounds/`: pruning. `lane_bound.hpp` (colour-mix and rotation lane bounds, lane tables),
    `pattern_db.hpp` (grouped pattern databases), `parent_prune.hpp` (PermStream hooks),
    `audit.hpp` (run-start safety audits), `calibrate.hpp` (per-run mode timing).
  - `ssd/`: the large solver in parts, `namespace ssd` (its README lists them).
  - `web/`: `web_solver.hpp`, the browser solver (plain C++, up to 10 moves; see `web/README.md`).
- `src/programs/`:
  - `solvers/`: `findSolutions` (in-RAM, 3-12 moves, reads `config.json` from cwd) and
    `experimentDiskDepth6` (12-16 moves; `main` + every flag documented at its top). The name is
    historical: it no longer needs disk tables. Rename later (scheduler, exports, docs use it).
  - `tests/` (incl. `exportGpuVectors`, the GPU's CPU reference data), `benchmarks/`
    (`benchmarkLeftStream`), `tools/` (`expandAllLevels`,
    `generate_ml_training_data`, `analyzeLevelsFinalHeuristic`, `DecompMindbenderScramble`),
    `experiments/` (`experimentBoundSample`: `--check`, sampling, `--bench`, `--deep --kind`;
    `experimentFatSplitBound`: move-type-split tables for 2x2 boards, research, see
    docs/changes-2026-11-04.md section 13).
- `src/gpu/`: the CUDA module (`mbgpu.h` C interface, `mbgpu.cu` kernels + host driver,
  `test_gpu_stream.cu`, `build.cmd`, `README.md`); the solver side is `src/solver/ssd/gpu_stream.hpp`.
  `src/gpu/vulkan/`: the Vulkan module (`mbvk.cpp` host, `shaders/*.comp` GLSL ports of the
  kernels, `build.cmd`, `README.md`).
  `build_gpu/` (ignored by git): the DLL, the test exe, `vectors.bin`, `bench_16-2.bin`, run logs.
- `src/utils/`: `progress_reporter.hpp` (cheap progress JSON), `parallel_for.hpp`, `timer.hpp`,
  `intrinsics/` (`my_clz`, `my_ctz64`, `my_pext_u64`, `my_popcount`).
- `old/`: code that no longer builds or is unreferenced, kept for history (not built; README).
- `build/levels_final/`: canonical answers `<p>_c<moves>_<count>.txt` (tracked, 0-indexed moves).
  `build/levels_final/game_accepts/`: placements the game also accepts. The game counts a board
  as solved when every colour matches, so the 2x2 piece may also end on another 2x2 block of its
  colour (7 puzzles: 4-4, 9-1, 13-5, 15-3, 17-2, 19-2, 19-4; 22 placements). Each is its own
  puzzle `<p>@r<row>c<col>` (the piece's top-left cell, 1-indexed: `4-4@r2c3`), solved
  separately, never counted as the official answer (`BoardLookup::gameAcceptedFats`,
  `getBoardPair("4-4@r2c3")`; their bounds are in their own section of `build/README.md`).
  `levels/` = scratch output.
- `python/scheduler/mbq.py`: job queue + dashboard; `README.md` there documents every command
  (add/`--until`/`--front`, cancel, pause, hold, shards + export/import). Use it for long runs.
  `mbq fleet` (`fleet.py`, `fleet_worker/*.sh`, 2026-11-06): one search's shards on rented Linux
  GPU machines over SSH (key `~/.ssh/mindbender_fleet`); they build from source, pass the GPU
  self-test, run units detached; results come back to `state/shards/`, the merge job runs here.
  CUDA on Linux: `src/gpu/build.sh`. Used 2026-11-06..07 on 9 vast.ai RTX 5090s (15-3 at 18, the
  placements, 16-3 at 18); unit logs come home to `docs/run_logs/fleet/`. `--recover-from` needs
  every shard file once, so a recovery split over machines = full file sets with disjoint middles
  (docs/changes-2026-11-05-07.md section 6).
- `docs/run_history.csv` + `docs/run_logs/`: one row and the full log per solver run (written by
  the scheduler via `python/scheduler/run_history.py`). Use it for timings.
- `web/`: static solver site (board editor, WebAssembly solver in Web Workers, replay). Serve with
  `python -m http.server 8770 --directory web`; rebuild the wasm with `web/build_wasm.ps1`.
- `python/chuzzle_suite/`: Chuzzle Suite, the pygame app (the demo app before 2026-11-03; Start menu
  shortcut from `install_shortcut.py`, icon from `assets/make_icon.py`; solutions, real-game playback via posted window messages, saves).
- `python/viewer/`: mouse-route studio; `mouse/`: C mouse-path library (`libchuzzle_mouse.dll`).
- `python/deluxe_clone/`, `python/automation/legacy/`: older Chuzzle 1 projects merged in.
- Tables: `X:/mindbender_d6/<puzzle>_goal_d6`, only for `--disk-table`. The default modes write
  nothing there. Emptied 2026-11-02 (6-4's 197.5 GB of tables, all for solved puzzles); keep the
  folder itself (the scheduler's table root and free-space figure).

## How the large solver works (experimentDiskDepth6)

A search `<S, T2, T>` = S + T2 + T moves: exact depth-S left seeds, each streamed T2 more moves
(`--stream-depth`, 3-6), every state looked up in the goal side's depth-T set. The scheduler
picks: normal boards `<5,3,6>` up to 14 moves, `<5,4,6>` at 15, `<5,5,6>` at 16 (one pass; a
depth-7 goal side is ~800 GB on normal boards: 18-3 at `<5,4,7>` = 86 passes), `<5,6,6>` at 17
(--gpu only; 18-3 solved with it 2026-11-03); 2x2 boards `<4,3,7>` at 14, `<5,3,7>` at 15,
`<5,4,7>` at 16, `<5,5,7>` at 17 (2026-11-04: recovery at left depth 10 checked on 2x2 boards,
`testExactSolverRegression --recovery 10 <file>`; ~32x a 16-move run, so seed shards).
- Goal side, no table written (default): depth 6 = the depth-5 frontier in RAM, index slices
  built per pass from regenerated keys; depth 7 = a count pass, then per pass the canonical
  2-move stream fills the pass's buckets (owner-routed keys, no atomics; the bucket counts
  wait in a temp file, each pass reads its slice; normal boards drop goal states with
  h(start) > left depth). Hits are confirmed by regenerating. An index larger than the RAM budget means several passes, each re-streaming the
  whole left side (2x2 15-move runs: 6-7 passes).
- Pruning (normal boards): h = worst row + worst column lane distance, rotation kind by default
  (per lane, the fewest cells wrong over its 6 rotations); |dh| <= 1 per move, so a node with r
  moves still below it is dropped when h > T + r. Pattern databases add to it at parents. Each
  run audits the bound on its own seeds (fast == reference, |dh| <= 1, identical wanted leaves)
  and times every eligible mode, keeping the fastest. On 2x2 boards the bound is halved and
  prunes nothing, so it is off.
- The first streamed move from each seed, and the first move of the goal side's expansion
  from its depth-5 frontier, follow the same canonical-order rules as the moves below them
  (default; `--no-seed-last-move` turns it off): each state's own last move, of its
  lexicographically smallest path, is passed in as the previous move (`ssd::buildLastMoves`,
  children from `perm_stream_detail::forEachFirstMove`). Exact; ~1.47x fewer streamed states on
  normal boards (60 first moves -> ~41), ~1.45x on 2x2 boards (48 -> ~33). Proposed, with a
  patch and the exactness argument, by the user's teacher `algorithm0r` (Chris Marriott) in
  GitHub issue 1 (also the lane-period `break` in the normal stream); his patch was reworked
  rather than applied as is, so the credit is in the commit messages (`b5431bc`, `9173a0f`,
  `ed871d8`), the code comments and docs/changes-2026-10-30.md. The issue is still open.
- Seeds are visited in a strided order (4096-seed chunks), so the timed ones represent the list.
- `--block` (2026-11-04, 2x2 boards, depth-7 goal side, CUDA): the free-block join. Leaves at depth
  L - 8 are looked up by row and column class (`src/solver/ssd/class_key.hpp`: rows turned to their
  smallest rotation, unchanged by row moves) in an index of the depth-7 goal states' classes; a goal
  state of the same class one lane away is a middle. One move fewer on the left: 19-4 at 16 as
  `<5,3,7>` + 1 in 34 min (was 9 h 36 min), EXACT MATCH, IDENTICAL SET. GPU key mode 2 gives each
  class once per parent (`mbgpu_set_key_mode`). docs/changes-2026-11-04.md section 15.
- `--direction auto|forward|reverse` (2026-11-04): reverse streams from the goal. auto (default, --gpu,
  normal boards) samples each direction's seeds on the GPU and keeps the one that expands fewer nodes
  (`src/solver/ssd/direction.hpp`; 16-2: forward, 15.6x; 16-3 at 18: reverse, 3.3x). `--block` also
  runs on normal boards (depth-6 goal side, bound cut-off one higher); the scheduler turns it on
  with --gpu from 16 moves (2x2) / 15 (normal). docs/changes-2026-11-04.md section 16.
- 18 moves (2026-11-06): recovery at left depth 12 (normal boards, `<5,6,6>` + block; 5 shared + 7
  per middle, checked on 18-3 and 14-3's 17-move answers: 2.5 min / 77 min) and 2x2 boards at
  left depth 11 (`<5,5,7>` + block; 6 per middle unfiltered, ~45 min per middle per thread;
  19-2's 3960 at 11 + 6). `testExactSolverRegression --recovery 12|11 <file>`.
- Work units: `--shard k/N`, `--middles-out`, `--recover-from` (exact; see the scheduler README).
  Depth-7 runs: `--d7-groups G` makes unit k = goal partition group k / (N/G) x seed shard
  k % (N/G), so each unit builds one pass's index (`mbq add ... --shards N --d7-groups G`).
- `--gpu 5060`: the RTX 5060 Ti (`src/gpu/`, loaded from `build_gpu/mbgpu.dll`; normal and 2x2
  boards, forward) streams the left side and Bloom-tests it (CPU threads finish the survivors
  meanwhile), audits its own bound (normal boards), builds the goal-side index (depth 6, or
  depth 7 on 2x2 boards: counts and each pass's index on the GPU, no spill file) and confirms
  candidates by regenerating it. A pass's Bloom blocks must also fit the VRAM (2x2 depth-7
  runs: 5-6 passes).
  Large pages are off with --gpu. The GTX 1660 drives the desktop and is never used.
- Recovery keeps only start-side paths that can still reach a middle (rotation lane bound to
  each middle; normal boards, <= 64 middles): 16-2 at 15 moves 58 s -> 6.3 s, exact. Each
  middle's own start half keeps only states the start root can still reach (any number of
  middles): 12-1 at 16 (119 middles) 2 h -> 56 s, exact.

## Conventions (from the user)

- **Don't replace `StateHash`.** Its locality is deliberate; a fully mixed hash made the in-RAM
  probe 2.7x slower. The SSD pipeline uses a separate `mixKey` for partitioning.
- **Tune per device at run time**, never hardcode a value that won one benchmark (cached
  profile + flag override). RAM budgets are measured at run start.
- **Reuse existing tools** (e.g. `testKnownSolutions`) before writing new checkers.
- Moves shown to the user are **1-indexed** (`R14` = row 1 shifted right 4; `C453` = 2x2 columns
  4+5 down 3). Stored files are 0-indexed. Report each new solve with one example solution.
- Performance changes need an A/B measurement on a **quiet machine**: pause the queue and hold or
  stop the running job first. The user's bound for overhead features is 0.5%.
- Never run two solver processes at once on the same table folder (it corrupted a table once).
  The scheduler enforces this.
- Restarts happen between jobs (pause the queue, let the job finish). No sharding just for
  restart safety (13-33% overhead per run); checkpoint/resume is for when runs take days.
- Code style: includes are always `"src/..."` paths; use the `src/utils/intrinsics/` helpers,
  not compiler builtins; dead code goes to `old/` (never deleted); the repo mixes CRLF and LF,
  and scripted edits must keep each file's line endings.
- Level data (`src/chuzzles/levels_cells.hpp`) must have the same colour counts in start and
  goal; 16-1 had a one-cell typo until 2026-10-12, which made it unsolvable. Board data is
  compiled in: rebuild the solvers after any change to it.
- Leave alone: the `mouse/` C library, Windows system/security settings (Windows Update, auto
  login are the user's), the MuMu emulator (it costs ~6% CPU; it stays on).
- The solver runs only when the user asks (no autostart, queue paused). Test runs on solved
  puzzles for exactness checks are fine.
- GPU work only on the RTX 5060 Ti (`--gpu 5060`), at most 12.5 GB of its VRAM; the GTX 1660
  drives the desktop. CPU parts of GPU runs use `--threads 6` (the user's
  choice, 4-6: 16 threads took the i7 to 100 C, 8 kept it at 85-90 C).

## Measured facts (2026-10)

Machine: i7-10700F (8 cores / 16 threads), 96 GB DDR4 (2 x 16 GB Crucial 3200 + 2 x 32 GB
Corsair, all at JEDEC 2133; the Lenovo BIOS has no XMP; don't flash a modded BIOS), X: = Team
T253 1 TB SATA SSD (32-70 MB/s sustained writes once its cache fills, recovers when idle).
GPUs (driver 610.88 on both): RTX 5060 Ti 16 GB (PCIe bus 1, x8, CUDA device 0, sm_120, 36 SMs)
for the solver; GTX 1660 SUPER 6 GB (bus 4, x4, CUDA device 1) for the desktop.

| Run | Time |
|---|---|
| 14 moves, normal board, rotation bound | 16-3: 24 min; 16-4: 1 h 27 min (was 10.8 h with colour mix + pattern tables) |
| 14 moves, normal board, colour mix (before 2026-10-12) | 32 min (14-3) to 10.7 h (17-3) |
| 14 moves, 2x2 board, `<4,3,7>` | 48-68 min |
| 15 moves, 2x2 board, `<5,3,7>` | 7.8-10.5 h (6-7 passes, ~2.6-2.8 T left states each, ~560 M/s) |
| 15 moves, normal board, `<5,4,6>` | 0.4 h (18-3) to 8.5 h (17-1), one pass |
| 16 moves, normal board, `<5,5,6>` | 8.5 h (18-3, 4.9 ms/seed) to 36.5 h (20-5, 12.6 ms/seed, 1.0e13 left states); one pass; recovery ~5 min (CPU, before the reach filter) |
| `--gpu 5060` (2026-11-01, src/gpu/README.md) | 16-2 at 15: 42 min -> 54 s (stream 12.6 s); 16-4 at 14: 1 h 27 min -> 2 min 18 s; 18-3 at 16: stream ~2.5 min from a 1/64 shard (whole run ~4 min) |
| `--gpu 5060`, 16 moves, normal boards (2026-11-02, 6 threads) | 15-1 19 min, 16-3 16 min, 17-3 38 min, 19-3 1 h 04 min, 12-1 2 h 13 min (2 h of it recovery; now 56 s) |
| `--gpu 5060`, 2x2 boards (2026-11-02) | 17-4 at 13: 294 s -> 36 s; 17-2 at 14 (5 passes): 56 min -> 5 min 50 s; 18-4 at 15 `<5,3,7>` (6 passes): 10.0 h -> 25 min; 6-4 at 16 `<5,4,7>` (disk depth-7 index, 6 passes x ~4000 s, 3.1e14 left states): 6 h 45 min |
| `--gpu 5060`, 2x2 boards at 16 `<5,4,7>` (2026-11-02..04) | 6-4 6 h 45 min, 16-1 8 h 45 min, 15-4 9 h 15 min, 15-3 ~9.3 h (none), 19-4 ~9 h 36 min, 19-2 ~10 h 37 min (none), without holds; 5-6 passes x ~5.2e13 left states; per pass ~1 260 s fixed + the Bloom tests and the CPU survivor check (about equal at 10 bits); survivor check ~18% faster since 2026-11-04 (docs/changes-2026-11-04.md sections 11-12) |
| 15 moves, 6-4 (disk depth-7 index) | 13.4 h, 10 passes |
| 13 moves, normal board, rotation bound | 10-3: 1 min 43 s; 9-3: ~6 min (colour mix: 5-25 min) |
| First-move rule (2026-10-29/30, docs/changes-2026-10-16.md 13-14) | 18-3 at 15: stream 1103 s -> 759 s; 17-4 at 13 (2x2): 468 s -> 313 s; 17-2 at 14 (2x2, depth 7): 6010 s -> 3345 s |

- Rotation bound, states generated vs colour mix: 16-3 68% -> 0.28%, 17-5 97% -> 3.9%, 14-5
  86% -> 5.8%, 17-3 at 15 moves 64% -> 0.72%. Timing tests: "both" 53x (9-3), 188-229x (10-3),
  293x (16-3) vs no bound. Tables 0.2-1.2 MB, same 3-loads-per-lane lookup as colour mix.
- Index sizes: depth-6 fingerprint index ~4.6 B/state; depth-7 ~4.8 B/state (2x2 15-move runs:
  209-237 GB, 6-7 passes in a ~35 GB budget).
- Large pages: often unavailable once memory is fragmented (a reboot restores them); they made
  no measurable difference on 6-4 at 13 moves. Allocating 18.8 GB of them took 9.6 s on 16-4
  (2026-11-01); they are off with --gpu.
- GPU (2026-11-01): node kernel ~0.63 ns per parent node (16-2: 17.7 G nodes in 11.9 s); 66 C,
  133 W at full load. Details: `src/gpu/README.md`, `docs/changes-2026-11-01.md`.
- Progress reporting costs +0.03%; the probe-variant calibration picks 128/plain here (-1.1%).
- Background CPU users and held/test runs slow jobs and their run-history timings (e.g. 16-4's
  colour-mix run streamed at 149 M/s instead of ~250-350).

## Roadmap

Done (2026-10-04 to 10-12; details in docs/changes-*.md):
1. Index buffers allocated once and reused across passes; RAM budget = free RAM at start minus a
   reserve (max(3 GB, 10%)) minus the run's own allocations.
2. Probe self-calibration of batch size / pipelining, cached per machine.
3. No-table depth 6 and depth 7 (goal side in RAM, nothing written; 2x2 14-move runs 2.6 h -> ~1 h).
4. Colour-mix lane bound, every-level pruning (PermStream PARENT_PRUNE hook, no-op when off),
   `--stream-depth 4` (`<5,4,6>` for 15-move normal boards), grouped pattern databases.
5. Mode calibration rewritten: one mode per block on all threads, paired rounds, seeds spread
   over the list (it had rated 16-4's pdbparent 1.1x where full runs show 1.5x).
6. Rotation lane bound (default): 5-25x faster 14-move normal runs.
7. Work units (`--shard`, middles files, `mbq add --shards --remote`, `mbq export`/`import`).
8. Scheduler: `--until` escalation, re-queue after reboot, Hold solver, run history.
9. Code reorganized (bounds/, ssd/, inram/, programs/ folders, old/, `src/` includes).

Next:
1. Bloom bits per goal state: more is faster (2026-11-04: 14 bits, 8 passes instead of 6, ~28%
   less time per 2x2 run, 7x fewer candidates; fewer bits is much slower; docs/changes-2026-11-04.md
   sections 11 and 14). Make it a per-device choice at run start (timed sample, cached) instead of
   the fixed default 10; until then pass `--bloom-bits 14`. More ideas: docs/ideas-2026-11-04.md.
2. A bound that prunes on 2x2 boards (no longer blocking: every 2x2 board is solved; 19-2 took 17,
   15-3 18): the halved
   lane bound tops out at 6 and pruning needs > 7, and no lane bound can do better (cover
   argument in docs/changes-2026-10-12.md section 7). Tables over several lanes do not get there
   either: all-moves pattern databases ~0%, move-type-split tables capped at 4 + 4 (2026-11-04,
   `experimentFatSplitBound`, docs/changes-2026-11-04.md section 13). Open: dissection-type search.
3. 16 moves on 2x2 boards: done (2026-11-04; 6-4, 15-4, 16-1, 19-4 solved, 15-3 and 19-2 none).
   17 moves on 2x2 boards: `<5,4,7> --block` (19-2 solved, 9 h 50 min; 15-3 none, 11 h 28 min, both
   on the 5060 Ti). 18 moves on 2x2 boards: `<5,5,7> --block`, 76 fleet units (15-3, 2026-11-07,
   docs/changes-2026-11-05-07.md). Before
   `--block`: `<5,5,7>`, ~12 days each on the 5060 Ti; run as `--d7-groups` units
   (`mbq add 19-2 --moves 17 --gpu 5060 --threads 6 --shards 80 --d7-groups 5`). 16 moves were
   `<5,4,7>` with `--gpu 5060` (`mbq add ... --gpu 5060 --threads 6`),
   ~7-11 h each (6-4: 6 h 45 min, 6 passes x ~4000 s of stream, 3.1e14 left states; CPU:
   ~200-300 h). Fewer passes need a
   smaller Bloom filter per state (VRAM caps a pass at ~8.7 G goal states) and a smaller index
   (item 1). Normal boards run `<5,5,6>` (~8.5 h); the goal-side
   filter by h(start) keeps 54% on 18-3, too little for a one-pass depth-7 goal side. 17 moves
   on normal boards: the 6-move stream `<5,6,6>` (item 5).
4. Vulkan module (2026-11-02, `src/gpu/vulkan/`): built, device discovery checked; first run
   `test_vk_stream` and a solved puzzle with `--gpu vulkan:5060` (README checks). Ran on a
   friend's PC (Linux, three Polaris GPUs, 2026-11-02..04): solved 18-3 and 14-3 at 17, 16-3 >= 18;
   that PC is no longer available for long runs. Still open: the checks on the 5060 Ti.
5. GPU port: normal boards done (2026-11-01), 2x2 boards done (2026-11-02; stream, depth-6 and
   depth-7 goal sides, candidate check; the scheduler passes --gpu). 17 moves on normal boards
   (14-3, 16-3, 18-3) as `<5,6,6>`: built 2026-11-02 (docs/changes-2026-11-02.md section 5),
   recovery checked on the CPU, not yet run end to end: first (machine idle) a CPU run of
   10-3 at 13 as `<1,6,6>` with `--solutions`, then `build_gpu\test_gpu_stream.exe
   build_gpu\vectors.bin` (now with two 6-move cases) and a 6-move GPU run on a solved puzzle
   (16-2 at 15 as `<3,6,6>`), then deploy `build_test/experimentDiskDepth6.exe` to `build/` and
   restart the scheduler daemon between jobs. Projected ~5-7 h each (16-3's 16-move stream
   took 14 min; one more move costs ~25-30x).
6. Checkpoint/resume inside a run (pass, finished-seed watermark, candidates, chosen modes),
   once single runs take days.
7. Smaller: AVX2 batched StateHash + mixKey for the left side; incremental bound evaluation
   (a row move leaves the row half of h unchanged); goal symmetry (12-1's goal has 144
   symmetries) to shrink goal tables.
8. Rename `experimentDiskDepth6` (after the 2026-10-15..22 trip; touches the scheduler,
   exported packages and docs).
