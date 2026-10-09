# Mindbender-Solver

Optimal solver for Chuzzle's Mind Bender puzzles (100 puzzles, 6x6 boards,
row/column cyclic shifts, some with a 2x2 piece). It finds every shortest
solution and proves nothing shorter exists. C++ solver plus Python tools.
Current status: `build/README.md` (unsolved list with "at least N moves"
bounds); history of what changed and why: `docs/changes-*.md`.

## Build and test

- Needs CMake >= 3.24 (system one is 3.23; use CLion's:
  `C:/Users/jerrin/AppData/Local/JetBrains/CLion 2022.3/bin/cmake/win/x64/bin/cmake.exe`),
  Ninja (CLion's), MinGW g++ at `C:/mingw64`, Release.
  Configure once: `cmake -S . -B <builddir> -G Ninja -DCMAKE_BUILD_TYPE=Release`, then
  `cmake --build <builddir> --target <name> -j 8`. Executables land in `build/`.
- Tests: `testExactSolverRegression` (13 cases), `testRotations`,
  `testKnownSolutions` (replays every file in `<repo>/levels`; run from `build/`),
  `python -m unittest discover -s python/demo_app/tests`.
- A running exe is locked: copy it before long runs (the scheduler does).

## Everyday commands (cmd)

```bat
cd /d C:\Users\jerrin\CLionProjects\Mindbender-Solver
python python\demo_app\run_demo.py
python python\scheduler\mbq.py daemon
python python\scheduler\mbq.py resume
python python\scheduler\mbq.py status
```

`daemon` starts the queue server; it does not start by itself after a reboot.
`status` is also available as a live page at http://127.0.0.1:8766/.

## Layout

- `src/chuzzles/`: board (`B1B2` = two u64), moves, lane shifts (`lane_rotations.hpp`, pext/pdep),
  canonical move streams (`perm_stream*`), level data (`levels.hpp`, `levels_cells.hpp`).
- `src/solver/`: meet-in-the-middle solvers; `solver_frontier.hpp` has the exact
  recovery from middle states (`recoverSolutionsFromMiddles`).
- `src/programs/findSolutions.cpp`: in-RAM exhaustive solver (3-12 moves), reads `config.json` from cwd.
- `src/programs/experimentDiskDepth6.cpp`: SSD pipeline. `<S,3,6>` (12-14 moves) with a depth-6
  goal table on disk; `--table-depth 7` = index-only depth-7 table for 15 moves. Tables in
  `X:/mindbender_d6/<puzzle>_goal_d6`. `--progress-file` for the scheduler.
- `src/utils/progress_reporter.hpp`: cheap progress JSON (one atomic add per seed).
- `build/levels_final/`: canonical answers `<p>_c<moves>_<count>.txt` (tracked). `levels/` = scratch output.
- `python/scheduler/mbq.py`: job queue + dashboard (http://127.0.0.1:8766/). Use it for long runs.
- `docs/run_history.csv` + `docs/run_logs/`: one row and the full log per solver run (written by
  the scheduler via `python/scheduler/run_history.py`; backfilled from 2026-10-04). Use it for timings.
- `python/demo_app/`: pygame app (solutions, real-game playback via posted window messages, saves).
- `python/viewer/`: mouse-route studio; `mouse/`: C mouse-path library (`libchuzzle_mouse.dll`).
- `python/deluxe_clone/`, `python/automation/legacy/`: older Chuzzle 1 projects merged in.

## Conventions (from the user)

- **Don't replace `StateHash`.** Its locality is deliberate; a fully mixed hash made the in-RAM
  probe 2.7x slower. The SSD pipeline uses a separate `mixKey` for partitioning.
- **Tune per device at run time**, never hardcode a value that won one benchmark (cached
  profile + flag override). RAM budgets are measured at run start.
- **Reuse existing tools** (e.g. `testKnownSolutions`) before writing new checkers.
- Moves shown to the user are **1-indexed** (`R14` = row 1 shifted right 4; `C453` = 2x2 columns
  4+5 down 3). Stored files are 0-indexed.
- Repo mixes CRLF and LF; scripted edits must preserve each file's line endings.
- Never run two solver processes at once on the same table folder (it corrupted a table once).
  The scheduler enforces this.
- Performance changes need an A/B measurement; the user's bound for overhead features is 0.5%.

## Performance roadmap

Measured facts (2026-10):
- 13-move normal puzzle (12-1): depth-6 build 7.5 min, index 2.7 min, stream ~355 s per full
  pass (~1.2 G left states/s), lookups ~3.4 ns each with normal pages. 15-move 6-4: 10 passes of
  2.59 T states, 58-78 min each, 13.4 h total.
- Large pages: unavailable on this machine once memory is fragmented (even fresh processes got
  normal pages after the long runs; a reboot restores them). On 6-4 at 13 moves they made no
  difference (298.6 s vs 298-300 s); the 1.4x seen once was on 6-2.
- Progress reporting overhead: +0.03% (A/B, 6-4 13-move).

Done (2026-10-06, verified):
1. Index buffers allocated once, before the seeds, and reused across passes (`BigArray::ensure`,
   `FingerprintRange::reserve`). 6-2 at 12 moves forced to 4 passes: exact middles, identical 64
   solutions; passes 2-4 load in 0.25-0.6 s. Same speed single-pass (318.4 s vs 318.1 s).
2. RAM budget = free RAM at start - max(3 GB, 10% of RAM) - the run's own seeds, instead of
   60% of free: 16.6 GB vs 12.8 GB on this machine, so a ~13 GB index (12-1) needs 1 pass, not 2
   (saves one full left stream, ~6 min of ~47 for 12-1).
3. No-table depth 6 (2026-10-07, default for `--table-depth 6`): the depth-5 frontier stays in RAM,
   each probe pass builds its index slice in RAM from regenerated depth-6 keys, and hits are confirmed
   by regenerating depth 6 once more. Nothing is written, so X:'s slow sustained writes (32-70 MB/s)
   no longer matter. Exact on 6-2/9-2 (12 moves) and 9-3 (13 moves, 13 solutions); 9-3 index built in
   84 s, 24.6 min total vs 40+ min disk builds. An existing complete table (e.g. 6-4) is still reused;
   `--disk-table` forces the old path. De-dupe is by 64-bit key, so the "unique" count is a little
   below the exact state count (StateHash collides); the final check compares full states.
4. No-table depth 7 (2026-10-07, default for `--table-depth 7` without a disk index): depth-5 frontier
   in RAM; a count pass records raw fingerprints per bucket (u16), then each probe pass regenerates
   depth 7 with the canonical 2-move stream (PermStream, ~1575/parent on 2x2 boards, ~520 M/s) and
   fills its buckets; keys are routed to the thread owning their partition and grouped by bucket and
   by Bloom block before writing (no atomics: locked random RMWs ran at ~116 M/s). 6-2 <2,3,7>: exact
   (64 solutions), 38 min = count 6.3 min + 6 passes x 5.3 min (generate ~2 min + group/write ~3 min).
   14-move 2x2 puzzles use <4,3,7> (~50 min vs 2.6 h for <5,3,6>). Normal boards' depth-7 set is ~6x
   bigger (~30 passes), so they stay on <5,3,6> (~1.3x at best with depth 7).
5. Colour-mix lower bound (2026-10-07, `--bound auto`, default; src/solver/colour_mix_bound.hpp):
   h = worst row colour-multiset mismatch + worst column's (a row move keeps row multisets and
   changes one cell per column, and vice versa; halved on 2x2 boards, where it prunes nothing).
   Admissible: 0 violations on all 1.89 M states of known optimal paths (experimentBoundSample
   --check). Fast form: per-puzzle lane tables, ~7.5 ns/state. Leaf mode skips lookups with
   h > table depth; "both" also skips the last PermStream level under nodes with h > depth + 1
   (PARENT_PRUNE hook in stream_lanes_dfs; no-op by default, same output and speed). Each run
   audits it on its own seeds, then times off/leaf/both for 20 s and keeps the fastest. Exact on
   9-2, 9-3, 10-3; 9-3 13 moves 24.6 -> 6.3 min. Skip rates vary per puzzle (23-99.9% at 14 moves);
   normal-board 14-move queue estimate 12.6 -> ~4.7 days.
- Background CPU users slow runs: the MuMu Android emulator (`MuMuVMMHeadless`) cost ~6%.
- RAM upgraded 2026-10-06 to 64 GB (2x32 GB Corsair CMK64GX4M2E3200C16). The Lenovo BIOS has no
  XMP, so it runs at JEDEC 2133 MHz (old Crucial 2x16 GB ran at 2933). Measured anyway: 6-4 13-move
  search 311.8 / 307.4 s vs 318 s before (large pages available again after the reboot), RAM
  budget 48.5 GB vs 16.6 GB. Don't flash a modded BIOS for XMP (Boot Guard brick risk); a JEDEC-3200
  kit (e.g. Crucial CT2K32G4DFD832A) would run at 2933 if speed ever matters.

3a. Probe self-calibration (2026-10-06): the first ~20 s of every run's first probe pass rotate the
   real seeds through batch sizes 32/64/128 x plain/pipelined, then use the fastest; cached per
   machine in %LOCALAPPDATA%/MindbenderSolver/probe_profile.txt (`--probe-variant G:P`,
   `--no-calibrate`). Correct for every variant (6-2: identical 64 solutions). Measured on 6-4
   13-move, alternating A/B: 322.1 s -> 318.4 s (-1.1%, net of the calibration). It picks 128/plain
   here; pipelining never helped. Its own per-thread estimate (~10.8%) overstates the real gain, so
   the bottleneck is probably not memory-level parallelism (page walks / raw latency instead).

3b. Depth-6 build (2026-10-06): on big tables it is half the run (16-3: 43 min table + 12 min index,
   6.7 B states) and it is disk-bound. X: (Team T253 1 TB SATA) measured **32 MB/s sustained write**
   once its cache is exhausted. `--ram-build` (in-RAM sort, no raw spill) is correct (6-2: identical 64
   solutions) but slower here (475 s vs ~200 s) because the disk build's writes hit the OS file cache.
   **Done 2026-10-07 (no-table mode, see item 3 under "Done")**: keep the index in RAM (built from 8-byte keys per pass, like the
   depth-7 builder) and confirm candidates by regenerating depth 6 from depth 5 once. Est. 16-3 build
   55 min -> ~2-4 min, and ~140 GB less written per puzzle. Fall back to disk when the index > RAM.

Next (medium, days):
3. Smaller index: binary fuse filter with 16-bit fingerprints (~2.3 B/state vs ~3.7 now) ->
   ~40% fewer passes, but 3 memory reads per lookup instead of 1; prototype and measure.
4. Faster left side: batch StateHash + mixKey with AVX2 (4 states at a time); in multi-pass runs
   ~90% of states are generated, hashed and discarded.

Bigger:
5. GPU port of the stream + lookup loop (est. 10-20x). CUDA on the RTX 5060 Ti first (also usable
   on the current GTX 1660 SUPER for development), kept HIP-portable for an RX 9070 XT; Vulkan
   compute if one cross-vendor build is wanted. Existing start: `src/chuzzles/rotations_cuda.cpp`,
   `docs/gpu_impl.md`. Needs MSVC Build Tools + CUDA Toolkit.
6. More RAM: index passes scale with index size / RAM (64 GB ~ halves passes).
7. 16 moves: needs seed depth 6 support (`buildSeeds` stops at 5; recovery left depth up to 8 today).
   6-4's depth-6 table and depth-7 index are kept on X: for this.
8. Split a search across machines: **done 2026-10-08** as work units (`--shard k/N`,
   `--middles-out`, `--recover-from`; `mbq add --shards N --remote ...`, `mbq export`, `mbq import`;
   portable build via `-DMB_MARCH=x86-64-v3 -DMB_OUTPUT_DIR=build_portable`). See
   python/scheduler/README.md. Exact on 9-2 (3 shards, 1200 solutions).
9. Research: IDA* with pattern-database lower bounds (abstract colours) for 17+ moves.
