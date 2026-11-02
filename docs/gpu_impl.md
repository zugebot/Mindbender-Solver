


there are 22 SM's,
each SM can control 1 warp at a time,
1 warp can control 32 threads.






## Implemented (2026-11-01)

The GPU solver exists now: `src/gpu/` (CUDA, RTX 5060 Ti: 36 SMs, sm_120), used by
`experimentDiskDepth6 --gpu 5060`. Its design and measurements are in `src/gpu/README.md`, its
history in `docs/changes-2026-11-01.md`. The earlier CUDA prototypes
(`src/chuzzles/rotations_cuda.cpp`, `src/programs/tests/testLayeredDFS.cpp`) were not used.
