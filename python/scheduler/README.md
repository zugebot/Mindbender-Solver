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

Not at logon (since 2026-10-30): the logon script no longer starts the server or resumes the
queue, and the queue is paused. Start it by hand when runs are wanted:
`mbq.py daemon`, then `mbq.py resume`.

GPU runs: `mbq add P --moves M --gpu 5060 --threads 8` passes `--gpu 5060` to the solver and
copies `build_gpu/mbgpu.dll` next to its exe copy in `state/bin` (a depth-6 goal side, or depth 7
on 2x2 boards; `src/gpu/README.md`). `build/experimentDiskDepth6.exe` and the DLL must be built
together (2026-11-02 or later for 2x2 boards). Re-queues after a reboot and `--until` escalations keep the setting.

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
| `add P [P...] --moves M` | queue puzzles. `--until N`: if no solution, re-queue one move deeper up to N. `--engine auto/ssd/inram`, `--ram-gb`, `--threads`, `--gpu NAME`, `--front`, `--keep-table`, `--no-promote` |
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
  `ssd` (`build/experimentDiskDepth6.exe`) for 10-17 moves, split as
  `<seed, stream, table>`: normal boards `<S,3,6>` up to 14, `<5,4,6>` at 15,
  `<5,5,6>` at 16 and `<5,6,6>` at 17 (`--gpu` required); 2x2 boards `<4,3,7>` at 14,
  `<5,3,7>` at 15, `<5,4,7>` at 16 and `<5,5,7>` at 17 (~1-2 weeks of GPU time: run it as
  `--shards N --d7-groups G`, below) (`--table-depth`, `--stream-depth` 3-6 override).
- **Block join** (2026-11-04): with `--gpu` (CUDA) the solver's `--block` is on by default from 16
  moves on 2x2 boards and 15 on normal boards: the left side is one move shorter (`<5,4,7>` + block
  at 17 on 2x2 boards, ~9 h instead of ~10 days; `docs/changes-2026-11-04.md` sections 15-16).
  `mbq add ... --block` forces it, `--no-block` turns it off; `status` shows `+block`.
- **Depth-7 work units** (`--shards N --d7-groups G`, N a multiple of G): unit k searches goal
  partition group k / (N/G) against seed shard k % (N/G). A plain seed shard rebuilds every
  pass's index (5-6 per 2x2 run); a grouped unit builds one, so long runs split with little
  overhead. G = the run's pass count (or more). The merge job recovers from all N files, and
  the solver checks that every unit is there exactly once. It runs a copy of the
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
- **Results**: solutions are copied to `levels/` and replayed twice (Chuzzle
  Suite's board simulator and `testKnownSolutions`). With promotion on (the
  default), a validated solve is copied to `build/levels_final` and removed
  from `build/README.md` / `build/config.json`'s unsolved lists; an empty
  search raises the README's "at least N moves" bound, but only when every
  shorter length was already excluded. Nothing is committed.
- **Game-accepted placements** (`mbq add 4-4@r2c3 ...`: the 2x2 piece on another 2x2 block of
  its colour, which the game also counts as solved): solutions go to
  `build/levels_final/game_accepts/` instead, and only that placement's own line in
  `build/README.md` changes (the official unsolved list and `config.json` stay as they are).
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

## Fleet (rented GPU machines)

`mbq fleet` runs every shard of one search on rented Linux GPU machines
(`fleet.py`, `fleet_worker/`). This PC stays in charge: it connects out to
each machine over SSH (no public address needed), copies the sources over,
hands out units, collects the middles files into `state/shards/` and runs the
merge job (recovery, validation, promotion) here, like any sharded search.

What to rent (vast.ai, RunPod, ...): one NVIDIA GPU per machine, RTX 5090
(32 GB) best, RTX 4090 next; 128 GB RAM (more RAM and VRAM = fewer goal
partition groups = less work); 8+ cores; 100 GB disk; SSH access. Image:
`nvidia/cuda:12.8.1-devel-ubuntu24.04` (nvcc, g++ 13, cmake 3.28; driver
570+). Rent identical machines: the unit layout is fixed per search.

```bash
python python/scheduler/mbq.py fleet key        # once: prints the public key for the rental account
python python/scheduler/mbq.py fleet add m1 root@1.2.3.4 -p 41234 --price 0.62
python python/scheduler/mbq.py fleet status     # setup: build + GPU self-test, ~5-10 min per machine
python python/scheduler/mbq.py fleet run 15-3 --moves 18 --units 60 --d7-groups 3
python python/scheduler/mbq.py fleet run 16-3 --moves 18 --units 40 --direction reverse
python python/scheduler/mbq.py fleet remove m1  # after its current unit (--now: at once)
```

- **Setup** (`fleet_worker/setup.sh`, on the machine): installs what is
  missing, builds `experimentDiskDepth6` (`-march=native`) and the CUDA module
  (`src/gpu/build.sh`, `-arch=native`), runs `test_gpu_stream` on
  `build_gpu/vectors.bin` and only then takes units.
- **Units** (`fleet_worker/run_unit.sh`): detached, one at a time per machine,
  so they keep running while this PC sleeps or the server restarts. Each poll
  (every 60 s, `status.sh`) reads progress and exit codes; a finished unit's
  file is checked (puzzle, split, shard, block) before it is filed.
- **Failures**: a unit that fails, disappears, stops making progress for an
  hour, or whose machine does not answer for 15 min goes back to the queue (at
  most 4 attempts). A finished unit found later is still collected; the first
  copy wins and a different second copy is reported.
- **Checks**: `--checks C` (default 2) units run again on another machine and
  must be byte-identical to the first copy; a mismatch is reported in `fleet
  status`. A missing middle cannot otherwise be detected.
- **Direction**: normal boards need `--direction forward|reverse`, decided once
  (the units must agree); 2x2 boards stream forward.
- **Merge**: the merge job is an ordinary queued job, so it waits while the
  queue is paused (`mbq resume`). Recovery runs on this PC (left depth 12 on
  normal boards at 18 moves: minutes to ~1.5 h, more middles take longer).
- Cost: `--price` (dollars per hour) gives the "spent so far" figure; stop
  paying at the rental site after `fleet remove`.
- `fleet run` options added 2026-11-06: `--front` (ahead of the other
  searches), `--only m4[,m5]` (only those machines take it), `--no-promote`
  (merge and validate, promote by hand). Fleet searches always use the SSD
  engine, so short (11-12 move) searches can be sharded too.
- Draining: `fleet remove NAME` lets the current unit finish, then the machine
  leaves; `fleet add NAME ...` with the same name brings it back (setup
  re-runs, ~5-10 min).
- Logs: every unit's full log comes home when it ends
  (`docs/run_logs/fleet/<puzzle>_m<M>_n<N>/<unit>.<machine>.ok|exit<code>.log`),
  and each machine's setup and self-test logs go to `docs/run_logs/fleet/machines/`.
- Recovery on rented machines (done by hand for 15-3 at 18, 2026-11-07):
  `--recover-from` needs every shard's file exactly once, so give each machine
  all N files with only its share of the unique middles left in (lower
  `count`); solutions through different middles differ, so the union of the
  machines' solution files is the answer. Scripts and logs:
  `docs/run_logs/fleet/15-3_m18_n76/recovery/`; docs/changes-2026-11-05-07.md
  section 6.
- vast.ai notes: image `nvidia/cuda:12.8.2-devel-ubuntu24.04` (template with
  SSH), use the **direct** SSH address, not the proxy; the instances page shows
  each rental's end date, and the site mails "expiring soon" before it.

## Solved alarm

When a job's solutions pass validation, the server loops the game's trophy jingles on this PC, loud (`GetTrophy` with
`TrophyJoy` mixed in on its final chord at 0.93 s, then 3 s of quiet, from `alarm.py`, through `winsound`, so no
browser is needed; the PC's volume applies) until it is
stopped or 15 minutes pass. The dashboard shows a flashing SOLVED banner with **Stop alarm**; its
**Sound in this tab** button makes the page play them too, and the page raises its own alarm if it
sees the solved count go up.

```bat
python python\scheduler\mbq.py alarm stop       :: silence it
python python\scheduler\mbq.py alarm test       :: sound it for 20 s
python python\scheduler\mbq.py alarm disable    :: no alarm for solves (enable turns it back on)
```
