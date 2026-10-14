# Mindbender-Solver

Optimal solver for Chuzzle's Mind Bender puzzles (100 puzzles, 6x6 boards,
row/column cyclic shifts, some with a 2x2 piece). It finds every shortest
solution and proves nothing shorter exists. C++ solver plus Python tools.

Status (2026-10-13): **79/100 solved**. `build/README.md` lists the unsolved ones with their
"at least N moves" bounds (updated by the scheduler); `build/levels_final/` holds every answer.
History of what changed and why: `docs/changes-*.md` (newest: `changes-2026-10-13.md`).

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
- Tests: `testExactSolverRegression` (13 cases), `testRotations` (112), `testJVec`,
  `testKnownSolutions` (replays every file in `<repo>/levels`; run from `build/`),
  `testWebSolver` (the web solver vs `build/levels_final`, all puzzles up to 10 moves; run from
  the repo root),
  `experimentBoundSample --check` (bounds admissible on all 1.9 M known-optimal-path states; run
  from the repo root), `python -m unittest discover -s python/demo_app/tests`.
- Exactness check for solver changes: run the solver on a solved puzzle with
  `--solutions build/levels_final/<file>` and look for `EXACT MATCH` and `IDENTICAL SET`
  (9-3 and 10-3 at 13 moves take minutes; 16-4 at 14 is the big one, 519 solutions).
- A running exe is locked: copy it before long runs (the scheduler does).

## Everyday commands (cmd)

```bat
cd /d C:\Users\jerrin\CLionProjects\Mindbender-Solver
python python\demo_app\run_demo.py
python python\scheduler\mbq.py daemon
python python\scheduler\mbq.py resume
python python\scheduler\mbq.py status
```

`status` is also a live page at http://127.0.0.1:8766/ (Hold solver / Pause queue buttons).
At logon, `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\mindbender_startup.cmd`
runs `mbq daemon` + `mbq resume`, the user's Discord bots (`E:\runner.bat`) and the demo app.
A job cut off by a reboot is re-queued first.

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
  - `tests/`, `benchmarks/` (`benchmarkLeftStream`), `tools/` (`expandAllLevels`,
    `generate_ml_training_data`, `analyzeLevelsFinalHeuristic`, `DecompMindbenderScramble`),
    `experiments/` (`experimentBoundSample`: `--check`, sampling, `--bench`, `--deep --kind`).
- `src/utils/`: `progress_reporter.hpp` (cheap progress JSON), `parallel_for.hpp`, `timer.hpp`,
  `intrinsics/` (`my_clz`, `my_ctz64`, `my_pext_u64`, `my_popcount`).
- `old/`: code that no longer builds or is unreferenced, kept for history (not built; README).
- `build/levels_final/`: canonical answers `<p>_c<moves>_<count>.txt` (tracked, 0-indexed moves).
  `levels/` = scratch output.
- `python/scheduler/mbq.py`: job queue + dashboard; `README.md` there documents every command
  (add/`--until`/`--front`, cancel, pause, hold, shards + export/import). Use it for long runs.
- `docs/run_history.csv` + `docs/run_logs/`: one row and the full log per solver run (written by
  the scheduler via `python/scheduler/run_history.py`). Use it for timings.
- `web/`: static solver site (board editor, WebAssembly solver in Web Workers, replay). Serve with
  `python -m http.server 8770 --directory web`; rebuild the wasm with `web/build_wasm.ps1`.
- `python/demo_app/`: pygame app (solutions, real-game playback via posted window messages, saves).
- `python/viewer/`: mouse-route studio; `mouse/`: C mouse-path library (`libchuzzle_mouse.dll`).
- `python/deluxe_clone/`, `python/automation/legacy/`: older Chuzzle 1 projects merged in.
- Tables: `X:/mindbender_d6/<puzzle>_goal_d6`, only for `--disk-table` and 6-4's kept depth-6
  table + 166 GB depth-7 index (keep them). The default modes write nothing there.

## How the large solver works (experimentDiskDepth6)

A search `<S, T2, T>` = S + T2 + T moves: exact depth-S left seeds, each streamed T2 more moves
(`--stream-depth`, 3 or 4), every state looked up in the goal side's depth-T set. The scheduler
picks: normal boards `<5,3,6>` up to 14 moves, `<5,4,6>` at 15; 2x2 boards `<4,3,7>` at 14,
`<5,3,7>` at 15; `<5,4,7>` at 16.
- Goal side, no table written (default): depth 6 = the depth-5 frontier in RAM, index slices
  built per pass from regenerated keys; depth 7 = a count pass, then per pass the canonical
  2-move stream fills the pass's buckets (owner-routed keys, no atomics). Hits are confirmed by
  regenerating. An index larger than the RAM budget means several passes, each re-streaming the
  whole left side (2x2 15-move runs: 6-7 passes).
- Pruning (normal boards): h = worst row + worst column lane distance, rotation kind by default
  (per lane, the fewest cells wrong over its 6 rotations); |dh| <= 1 per move, so a node with r
  moves still below it is dropped when h > T + r. Pattern databases add to it at parents. Each
  run audits the bound on its own seeds (fast == reference, |dh| <= 1, identical wanted leaves)
  and times every eligible mode, keeping the fastest. On 2x2 boards the bound is halved and
  prunes nothing, so it is off.
- Seeds are visited in a strided order (4096-seed chunks), so the timed ones represent the list.
- Work units: `--shard k/N`, `--middles-out`, `--recover-from` (exact; see the scheduler README).

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

## Measured facts (2026-10)

Machine: i7-10700F (8 cores / 16 threads), 64 GB DDR4 at JEDEC 2133 (the Lenovo BIOS has no
XMP; don't flash a modded BIOS), X: = Team T253 1 TB SATA SSD (32-70 MB/s sustained writes once
its cache fills, recovers when idle). GPU: GTX 1660 SUPER now; an RTX 5060 Ti from ~2026-10-23.

| Run | Time |
|---|---|
| 14 moves, normal board, rotation bound | 16-3: 24 min; 16-4: 1 h 27 min (was 10.8 h with colour mix + pattern tables) |
| 14 moves, normal board, colour mix (before 2026-10-12) | 32 min (14-3) to 10.7 h (17-3) |
| 14 moves, 2x2 board, `<4,3,7>` | 48-68 min |
| 15 moves, 2x2 board, `<5,3,7>` | 8.9-10.5 h (6-7 passes, ~2.6-2.8 T left states each, ~560 M/s) |
| 15 moves, 6-4 (disk depth-7 index) | 13.4 h, 10 passes |
| 13 moves, normal board, rotation bound | 10-3: 1 min 43 s; 9-3: ~6 min (colour mix: 5-25 min) |

- Rotation bound, states generated vs colour mix: 16-3 68% -> 0.28%, 17-5 97% -> 3.9%, 14-5
  86% -> 5.8%, 17-3 at 15 moves 64% -> 0.72%. Timing tests: "both" 53x (9-3), 188-229x (10-3),
  293x (16-3) vs no bound. Tables 0.2-1.2 MB, same 3-loads-per-lane lookup as colour mix.
- Index sizes: depth-6 fingerprint index ~4.6 B/state; depth-7 ~4.8 B/state (2x2 15-move runs:
  209-237 GB, 6-7 passes in a ~35 GB budget).
- Large pages: often unavailable once memory is fragmented (a reboot restores them); they made
  no measurable difference on 6-4 at 13 moves.
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
1. Smaller index for multi-pass runs: a binary fuse filter at ~2 B/state instead of 4.8 would
   halve the passes of 2x2 15-move runs (~10 h -> ~5 h) and 6-4 at 16. Generation dominates
   those runs (each pass re-streams the left side), so 3 reads per lookup instead of 1 hardly
   matters. Prototype and A/B.
2. A bound that prunes on 2x2 boards (6-4, 15-3, 15-4, 19-2 need >= 16, all 2x2): the halved
   lane bound tops out at 6 and pruning needs > 7, and no lane bound can do better (cover
   argument in docs/changes-2026-10-12.md section 7). It needs bounds over several lanes at once
   (pattern tables, e.g. split by move type), or the time has to come from fewer passes / the GPU.
3. 16 moves: `<5,4,7>` works with today's code. 2x2 boards ~300 h each on this CPU; 6-4 with its
   disk index ~150-160 h (~70-80 h with item 1). Normal boards at 16: try pruning the goal side
   with the bound measured from the start (states must be within the left depth of the start),
   which could make a depth-7 goal side small enough for one pass.
4. GPU port of the stream + lookup loop (est. 10-20x): CUDA on the RTX 5060 Ti first, kept
   HIP-portable. Starting points: `src/chuzzles/rotations_cuda.cpp`, `docs/gpu_impl.md`,
   `tests/testLayeredDFS.cpp` (CUDA-only target). Needs MSVC Build Tools + CUDA Toolkit.
5. Checkpoint/resume inside a run (pass, finished-seed watermark, candidates, chosen modes),
   once single runs take days.
6. Smaller: AVX2 batched StateHash + mixKey for the left side; incremental bound evaluation
   (a row move leaves the row half of h unchanged); goal symmetry (12-1's goal has 144
   symmetries) to shrink goal tables.
7. Rename `experimentDiskDepth6` (after the 2026-10-15..22 trip; touches the scheduler,
   exported packages and docs).
