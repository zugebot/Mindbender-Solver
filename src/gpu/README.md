# src/gpu/

The GPU left stream for `experimentDiskDepth6 --gpu NAME` (CUDA; RTX 5060 Ti here).

| File | What |
|---|---|
| `mbgpu.h` | C interface (plain C types: the solver is MinGW, this module MSVC + nvcc) |
| `mbgpu.cu` | kernels and host driver |
| `test_gpu_stream.cu` | checks the GPU stream against the CPU's, seed by seed; `--bench` times a full split |
| `build.cmd` | builds `build_gpu/mbgpu.dll` and `build_gpu/test_gpu_stream.exe` |

The solver loads `mbgpu.dll` at run time (`src/solver/ssd/gpu_stream.hpp`): next to the exe, else
`<exe dir>/../build_gpu`, or `--gpu-dll PATH`.

## What runs where

Per probe pass, the GPU gets the pass's Bloom blocks, streams every left seed `T2` moves and
hands back the leaves that pass the Bloom filter (~1%); the CPU finishes those with the same index
code as the CPU probe (`FingerprintRange::probeGroup`). The goal-side index is still built on the
CPU, and recovery is unchanged.

The stream re-implements the CPU's rules exactly (`perm_stream_detail::stream_lanes_dfs` with
the seed's last move, `bounds::LanePruneLeaves`): same-direction lanes ascending, a lane stops at
its period, commuting column-row pairs skipped, and the rotation lane bound at every level (a
node with r moves below it is dropped when h > limit + r). It works level by level: one thread
per (node, lane) computes the lane's five shifts and appends the kept children to the next
level's buffer; each level is done in chunks whose children always fit, and every chunk's
subtree is finished before the next chunk, so each level has one fixed buffer and every kernel
launch is short (Windows resets a GPU whose kernel runs ~2 s).

Not yet: 2x2 boards, reverse searches, the colour-mix bound and pattern databases (the GPU always
uses the rotation bound in "both" mode).

## Safety

- Only the CUDA device whose name contains NAME is used (`--gpu 5060`); it refuses others.
- At most `--gpu-vram-gb` (default 12.5) of its memory: level buffers 1.3 GB, the pass's Bloom
  blocks (10 bits per goal state, 3-8 GB on the unsolved normal boards), output 0.1 GB.
- Every launch is short (chunks of at most 140 k nodes).
- Any CUDA error or buffer overflow stops the run with an error; nothing is silently skipped.

## Checks

```
cmake --build cmake-build-test --target exportGpuVectors
build_test\exportGpuVectors.exe build_gpu\vectors.bin
src\gpu\build.cmd
build_gpu\test_gpu_stream.exe build_gpu\vectors.bin
```

`exportGpuVectors` streams sample seeds of 1-1, 9-3, 10-3, 18-3 and 16-2 with the CPU code
(one thread, a few seconds), unpruned and pruned, with and without last moves, and records per
seed the leaf count and the sum and xor of the leaf keys; `test_gpu_stream` must match every
seed. End to end: `--gpu 5060 --solutions build/levels_final/<file>` (EXACT MATCH and
IDENTICAL SET), as for any solver change.

## Measured (2026-11-01, RTX 5060 Ti, i7-10700F)

| Run | CPU (16 threads) | GPU |
|---|---|---|
| 16-2 at 15 `<4,5,6>`, stream of 3.29 M seeds, 17.7 G nodes, 273 M leaves | ~1500 s (2196 s before the first-move rule) | 123 s |
| 10-3 at 13 `<4,3,6>`, stream + probe, 13.9 M left states | 10.0 s | 0.87 s, EXACT MATCH, IDENTICAL SET |
