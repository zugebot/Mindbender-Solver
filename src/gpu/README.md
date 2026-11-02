# src/gpu/

The GPU module for `experimentDiskDepth6 --gpu NAME` (CUDA; RTX 5060 Ti here).

| File | What |
|---|---|
| `mbgpu.h` | C interface (plain C types: the solver is MinGW, this module MSVC + nvcc) |
| `mbgpu.cu` | kernels and host driver |
| `test_gpu_stream.cu` | checks the GPU stream against the CPU's, seed by seed; `--bench` times a full split |
| `build.cmd` | builds `build_gpu/mbgpu.dll` and `build_gpu/test_gpu_stream.exe` |

The solver loads `mbgpu.dll` at run time (`src/solver/ssd/gpu_stream.hpp`): next to the exe, else
`<exe dir>/../build_gpu`, or `--gpu-dll PATH`.

## What runs where

- **Probe:** per pass, the GPU gets the pass's Bloom blocks, streams every left seed `T2` moves
  and hands back the leaves that pass the Bloom filter (~1%); CPU threads finish those with the
  same index code as the CPU probe (`FingerprintRange::probeGroup`) while the GPU streams on
  (a bounded queue, `ProbeQueue`).
- **Bound audit** (`gpuAuditBound`, in place of `bounds::auditLaneBound`): the audit kernel
  (`expandKernel`) computes h from its definition: h equals `lane_bound::referenceBound` on
  ~160 k sampled states; on every edge of the full streams of 256 seeds (2048 below stream
  depth 5) h changes by at most 1; per seed the pruned stream (the node kernel, incremental
  bound) keeps exactly the full stream's leaves with h <= limit. A failure turns the bound off,
  as on the CPU.
- **Goal-side depth-6 index** (`no_table_d6.hpp`): the GPU counts the children per partition,
  and per group of partitions generates, sorts (CUB) and de-duplicates their keys and writes the
  fingerprints, bucket offsets and Bloom blocks straight into the index arrays, in the CPU's
  layout.
- **Candidate check** (`mbgpu_match_children`): the GPU regenerates depth 6 and hands back the
  children whose key is a candidate's; the CPU compares the full states.
- Still on the CPU: the depth-5 frontiers and their last moves, recovery.

## The node kernel (`expandNodes`)

The stream re-implements the CPU's rules exactly (`perm_stream_detail::stream_lanes_dfs` with
the seed's last move, `bounds::LanePruneLeaves`): same-direction lanes ascending, a lane stops at
its period, commuting column-row pairs skipped, and the rotation lane bound at every level (a
node with r moves below it is dropped when h > limit + r).

- One thread per node tries all 12 lanes x 5 amounts, so the threads of a warp run the same loop.
- The bound is incremental. A lane's count word holds, in nibble k, how many of its cells differ
  from its target lane at rotation k; its distance is the smallest nibble. A row move changes
  one cell of every column, so a child's column words are the parent's minus the old cell's
  mismatches plus the new cell's: one shared-memory lookup per column (`c_mc` / `c_mr`, built
  by `mbgpu_set_target`). The moved row's own distance does not change.
- Children are staged per warp in shared memory and written out in contiguous runs.
- Work is level by level. Each level is done in chunks sized from the measured children per
  node (80% of the next level's buffer); a chunk that overflows its buffer is run again
  smaller, before anything reads it. Every chunk's subtree is finished before the next chunk,
  so each level has one fixed buffer and every kernel launch is short (Windows resets a GPU
  whose kernel runs ~2 s).
- Separate builds for inner levels and leaves; 80 registers, 3 blocks per SM (4 is slower:
  the register cap spills and shared memory allows only 3).

Not yet: 2x2 boards, reverse searches, stream depth 6 on the solver side, the colour-mix bound
and pattern databases (the GPU always uses the rotation bound in "both" mode).

## Safety

- Only the CUDA device whose name contains NAME is used (`--gpu 5060`); it refuses others.
- At most `--gpu-vram-gb` (default 12.5) of its memory: level buffers 1.3 GB, the pass's Bloom
  blocks (10 bits per goal state, 3-8 GB on the unsolved normal boards), output 0.1 GB.
- Any CUDA error or buffer overflow that cannot be retried stops the run with an error; nothing
  is silently skipped.

## Checks

```
cmake --build cmake-build-test --target exportGpuVectors
build_test\exportGpuVectors.exe build_gpu\vectors.bin
src\gpu\build.cmd
build_gpu\test_gpu_stream.exe build_gpu\vectors.bin
build_gpu\test_gpu_stream.exe build_gpu\bench_16-2.bin --bench [--seeds 200000] [--filter-gb 3]
```

`exportGpuVectors` streams sample seeds of 1-1, 9-3, 10-3, 18-3 and 16-2 with the CPU code
(one thread, a few seconds), unpruned and pruned, with and without last moves, and records per
seed the leaf count and the sum and xor of the leaf keys; `test_gpu_stream` must match every
seed. `--bench` (a file from `exportGpuVectors --bench`) times the stream; `--filter-gb` adds a
random Bloom filter at a real one's density. End to end: `--gpu 5060 --solutions
build/levels_final/<file>` (EXACT MATCH and IDENTICAL SET), as for any solver change.

Profiling: Nsight Systems (`nsys profile --trace=cuda --stats=true ...`) for launches and gaps;
Nsight Compute (`ncu --set full --import-source yes`) needs "Allow access to the GPU
performance counters to all users" (NVIDIA Control Panel, Desktop > Enable Developer Settings,
Developer > Manage GPU Performance Counters).

## Measured (2026-11-01, RTX 5060 Ti, i7-10700F, CPU parts on 8 threads)

| Run | CPU (16 threads) | GPU, first version | GPU now |
|---|---|---|---|
| 16-2 at 15 `<4,5,6>`: stream + probe (17.7 G nodes) | 2196 s (~1500 s with the first-move rule) | 122 s | 12.6 s |
| 16-2 at 15: bound audit | 156 s (2 seeds) | 5.9 s (256 seeds) | 5.5 s |
| 16-2 at 15: candidate check | 15.8 s | 15.8 s (CPU) | 1.2 s |
| 16-2 at 15: recovery (CPU, reach filter) | 58 s | 58 s | 6.3 s |
| 16-2 at 15: whole run, EXACT MATCH, IDENTICAL SET (366) | 42 min | 3 min 45 s | 54 s |
| 10-3 at 13 `<4,3,6>`: whole run, IDENTICAL SET (1648) | 95 s | 23 s | 14.6 s |
| 16-4 at 14 `<5,3,6>`: whole run, IDENTICAL SET (519) | 1 h 27 min | - | 2 min 18 s |
| 18-3 at 16 `<5,5,6>`: stream, from a 1/64 shard | 8.5 h (whole run) | - | ~2.5 min (whole run ~4 min) |

Where the stream's time went (16-2, 200 k seeds, 7.81 s -> 0.74 s): one thread per (node,
lane) kept ~10 of 32 threads of a warp busy, every child's bound took 6 rotations x 6 lanes,
and 18% of the wall time was host round trips (169 k launches for the full run, now 5 k). The
node kernel: 8.2x; warp staging of the children: 7%; precomputed lane cells: 5%; register cap
for 3 blocks per SM: 12%.
