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
- Background CPU users slow runs: the MuMu Android emulator (`MuMuVMMHeadless`) cost ~6%.

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
8. Split a search across machines: shard seeds (`--shard i/N`) or partitions; only candidate lists
   cross the network.
9. Research: IDA* with pattern-database lower bounds (abstract colours) for 17+ moves.
