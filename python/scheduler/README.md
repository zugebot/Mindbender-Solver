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
| `shutdown` | stop the server; a running solver keeps going and is re-adopted on the next start |

The same is available as JSON on `http://127.0.0.1:8766` (`GET /status`,
`GET /jobs`, `GET /jobs/ID`, `GET /jobs/ID/log?tail=N`, `POST /jobs`,
`POST /jobs/ID/cancel`, `POST /pause`, `/resume`, `/shutdown`). It listens on
localhost only.

## What it does with a job

- **Engine**: `inram` (`build/findSolutions.exe`, exhaustive) for 3-12 moves,
  `ssd` (`build/experimentDiskDepth6.exe`) for 10-15 moves: `<S,3,6>` up to
  14, `<5,3,7>` with the index-only depth-7 table at 15. It runs a copy of the
  exe from `state/bin`, so rebuilding `build/` never clashes with a run.
- **One at a time, safely**: a job never starts while any other solver process
  is running (including one started by hand), and the queue is a single
  worker, so two runs can never share a table folder.
- **Waits for disk**: a job that has to build a table waits until the table
  drive has room (~100 GB for depth 6, ~550 GB for depth 7).
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
- **Restarts**: jobs live in `state/jobs.json` (git-ignored). If the server
  stops while a solver runs, the solver keeps going and the next server
  adopts it and records its result.

## Cost

The server sleeps between 5-second polls and reads a few small files. In the
solver, progress is one relaxed atomic add per left seed or partition (each
milliseconds of work) and a file write every 5 seconds from a background
thread; there is no printing or locking on the hot path.
