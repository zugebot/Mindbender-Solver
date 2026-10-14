# mbq: Mindbender solver queue

A small always-on scheduler for long solver runs: queue puzzles, ask how far
along the current run is, and let it validate and record the results.
Python standard library only (Windows).

```bash
python python/scheduler/mbq.py daemon      # start the server in the background (once)
python python/scheduler/mbq.py status      # running job, phase, %, ETA, queue
python python/scheduler/mbq.py add 12-1 12-5 --moves 13 --until 14
```

```text
(example output)
#7   6-4 15 moves  ssd <5,3,7>  [running]
  phase: probe pass 6/11 streaming  51.2%
  elapsed 5h 02m; this phase ends in 3h 40m
  whole job: about 3h 45m left, done around 21:40
```

## Dashboard

Open **http://127.0.0.1:8766/** while the server runs. It refreshes every 3
seconds and shows the running job (phase, progress bar, this phase's ETA and
the whole job's finish time, phase steps with their durations), the live
solver log, the queue with any wait reason, finished jobs with their results
and notes, the solved count, free disk space, and pause / resume / cancel
buttons. It is served by the scheduler itself: no internet, nothing to install.

The server only answers requests addressed to 127.0.0.1 / localhost, and
commands (POST) need an `X-MBQ: 1` header, so other web pages open in the
same browser cannot send it commands.

## Commands

| Command | What it does |
|---|---|
| `daemon` / `serve` | start the server in the background / in this console |
| `status` | running job (phase, percent, phase and whole-job ETA), queue, free disk |
| `add P [P...] --moves M` | queue puzzles. `--until N`: if no solution, re-queue one move deeper up to N. `--engine auto/ssd/inram`, `--ram-gb`, `--threads`, `--front`, `--keep-table`, `--no-promote` |
| `queue` / `history [-n N]` | queued and running jobs / finished jobs with their notes |
| `log ID [--tail N]` | the solver's log for a job (written live) |
| `cancel ID` | drop a queued job, or stop a running one |
| `pause` / `resume` | finish the current job but start no new ones / continue |
| `hold` / `unhold` | freeze the running solver in place (all cores free, e.g. to record the game) / let it carry on from the same point. Nothing is lost or redone; the ETA leaves the held time out. Survives a server restart, not a reboot |
| `shutdown` | stop the server; a running solver keeps going and is re-adopted on the next start |
| `add P --moves M --shards N [--remote k,...]` | one search as N work units: local shard jobs, then a merge job that waits for every shard's result (see Work units) |
| `export P --moves M --shards N --which k,...` | a folder for another PC: the portable solver + one `.bat` per shard |
| `import FILE...` | file shard results (`.mid`) returned from other PCs |

The same is available as JSON on `http://127.0.0.1:8766` (`GET /status`,
`GET /jobs`, `GET /jobs/ID`, `GET /jobs/ID/log?tail=N`, `POST /jobs`,
`POST /jobs/ID/cancel`, `POST /pause`, `/resume`, `/hold`, `/unhold`, `/shutdown`). It listens on
localhost only.

## What it does with a job

- **Engine**: `inram` (`build/findSolutions.exe`, exhaustive) for 3-12 moves,
  `ssd` (`build/experimentDiskDepth6.exe`) for 10-16 moves, split as
  `<seed, stream, table>`: normal boards `<S,3,6>` up to 14 and `<5,4,6>` at 15;
  2x2 boards `<4,3,7>` at 14, `<5,3,7>` at 15; 16 moves `<5,4,7>`
  (`--table-depth`, `--stream-depth` override). It runs a copy of the
  exe from `state/bin`, so rebuilding `build/` never clashes with a run.
- **One at a time, safely**: a job never starts while any other solver process
  is running (including one started by hand), and the queue is a single
  worker, so two runs can never share a table folder.
- **No disk wait**: the default no-table modes keep the goal side in RAM and
  write no tables, so jobs do not wait for free space on the table drive.
- **Progress**: the solver writes `state/jobs/ID/progress.json` every 5 s
  (`--progress-file`); its log is flushed line by line. The whole-job ETA
  adds the median duration of later phases from finished jobs of the same
  kind (engine, table depth, 2x2 piece, moves), so it gets better over time.
- **Results**: solutions are copied to `levels/` and replayed twice (the demo
  app's board simulator and `testKnownSolutions`). With promotion on (the
  default), a validated solve is copied to `build/levels_final` and removed
  from `build/README.md` / `build/config.json`'s unsolved lists; an empty
  search raises the README's "at least N moves" bound, but only when every
  shorter length was already excluded. Nothing is committed.
- **Cleanup**: a table the scheduler created is deleted when no queued job
  for that puzzle needs it (`--keep-table` keeps it). Tables it did not
  create are never deleted.
- **History**: every job that ran gets a row in `docs/run_history.csv` (times per
  phase, table and index sizes, passes, large pages, M states/s, probe variant,
  machine, solver commit) and its full log in `docs/run_logs/`. Both are tracked
  in git; `run_history.py backfill` added the runs from before the scheduler.
- **Restarts**: jobs live in `state/jobs.json` (git-ignored). If the server
  stops while a solver runs, the solver keeps going and the next server
  adopts it and records its result.

## Cost

The server sleeps between 5-second polls and reads a few small files. In the
solver, progress is one relaxed atomic add per left seed or partition (each
milliseconds of work) and a file write every 5 seconds from a background
thread; there is no printing or locking on the hot path.

## Work units (one search on several PCs)

A search streams every left seed against the same index, and the seeds are
independent, so `--shards N` cuts it into N work units: shard k searches the
seeds with index = k (mod N) and writes the middles it found (usually none)
to a small text file. A merge job waits until all N files exist, then runs
the solver with `--recover-from` on them: it checks that they are shards
0..N-1 of the same puzzle and split, each exactly once and complete, merges
them, recovers the solutions and validates and promotes them like any job.
Tested end to end on 9-2 (3 shards, one run through an exported package):
1200 solutions, replay and `testKnownSolutions` 1200/1200.

```bash
python python/scheduler/mbq.py add 14-3 --moves 15 --shards 8 --remote 6,7
python python/scheduler/mbq.py export 14-3 --moves 15 --shards 8 --which 6,7   # -> workunits/14-3_m15_n8/
# ... zip the folder, the other PC double-clicks run_*.bat, sends back the .mid ...
python python/scheduler/mbq.py import 14-3_m15_s6of8.mid 14-3_m15_s7of8.mid
```

The export uses `build_portable/experimentDiskDepth6.exe`, built for any
AVX2 + BMI2 CPU (Intel since 2013, AMD Zen 3+; older Ryzens run `pext` ~50x
slower):

```bash
cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DMB_MARCH=x86-64-v3 -DMB_OUTPUT_DIR=<repo>/build_portable
cmake --build <dir> --target experimentDiskDepth6
```

Results come back as files, so nothing on this PC is opened to the network.
A work unit's middles count is trusted as reported: a missing middle cannot
be detected without redoing that shard, so use machines you trust (or run a
shard twice on different PCs and compare the files).

## Solved alarm

When a job's solutions pass validation, the server loops a loud siren on this PC (a generated WAV at
full amplitude through `winsound`, so no browser is needed; the PC's volume applies) until it is
stopped or 15 minutes pass. The dashboard shows a flashing SOLVED banner with **Stop alarm**; its
**Sound in this tab** button makes the page play a siren too, and the page raises its own alarm if it
sees the solved count go up.

```bat
python python\scheduler\mbq.py alarm stop       :: silence it
python python\scheduler\mbq.py alarm test       :: sound it for 20 s
python python\scheduler\mbq.py alarm disable    :: no alarm for solves (enable turns it back on)
```
