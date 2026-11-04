# src/gpu/vulkan: the Vulkan GPU module (mbvk.dll)

The same GPU work as the CUDA module (`src/gpu/`, `mbgpu.dll`), for any GPU with a Vulkan 1.2
driver: AMD, Intel and NVIDIA. It implements the whole C interface in `src/gpu/mbgpu.h`:
the left stream with the Bloom test, the per-seed checksums, the bound audit, the goal-side
index build and the candidate check. The solver loads whichever module fits. The CUDA module
and its kernels are unchanged, so NVIDIA runs are exactly as fast as before.

Status (2026-11-02): checked on Linux on a friend's PC (Ryzen 5 5600X, two Radeon Pro Duo =
four Polaris 10 GPUs of 16 GB, Ubuntu 24.04, Mesa RADV 25.2): `test_vk_stream` PASS (every
stream case, every audit, the sort), and EXACT MATCH / IDENTICAL SET on 10-3 at 13, 16-2 at 15
and 17-4 at 13 (2x2), on one GPU and on several. Not yet run on the RTX 5060 Ti (the queue held
it).

## Files

| File | What |
|---|---|
| `mbvk.cpp` | the host side: device discovery and choice, buffers, the level-by-level driver (a port of `mbgpu.cu`'s host code), several GPUs at once |
| `mbvk.h` | the one extra export, `mbvk_list_devices` |
| `shaders/common.glsl` | board layout, hashes, the bound from its definition, Bloom and match tests |
| `shaders/expand.comp` | node kernel, normal boards (`expandNodes`: incremental bound from packed mismatch counts) |
| `shaders/expand_fat.comp` | node kernel, 2x2 boards (`expandFat`) |
| `shaders/audit.comp` | the audit's kernel (`expandKernel`: one invocation per node and lane, h on every edge) |
| `shaders/bound.comp` | `mbgpu_bounds` |
| `shaders/keys.comp` | radix sort, scan, unique, partition bounds, fingerprints, bucket starts, Bloom blocks (CUB's job in CUDA) |
| `vk_devices.cpp` | `vk_devices.exe`: lists the devices |
| `build.cmd` | builds everything into `build_gpu/` |

## Build

Needs the Vulkan SDK (`VULKAN_SDK`; 1.4.341 here) and Visual Studio 2022's C++ compiler.

```bat
src\gpu\vulkan\build.cmd
```

The output, in `build_gpu/`:
- `mbvk.dll`
- `test_vk_stream.exe` (the CUDA test program, `src/gpu/test_gpu_stream.cu`, compiled as C++ against `mbvk.cpp`)
- `vk_devices.exe`
- `vk_spv/*.inc` (the kernels as SPIR-V, embedded into all three)

The DLL links the C runtime statically and loads `vulkan-1.dll` at run time, so it needs
nothing but a GPU driver. On a PC without Vulkan, `mbgpu_open` reports that instead of failing
to load.

## Choosing devices

`--gpu` takes `[cuda:|vulkan:]NAME`:
- **No prefix:** the solver tries `mbgpu.dll` (CUDA) first, then `mbvk.dll`. It moves on only
  when the CUDA module has no such device (for example no NVIDIA driver); any other CUDA error
  stops the run.
- **`NAME = auto`:**
  - CUDA takes its first device, which CUDA orders fastest first.
  - Vulkan takes every discrete GPU that drives no display (Windows: DXGI adapters with
    outputs; Linux: the PCI addresses of connected DRM outputs). With the monitor off or
    unplugged, no output is connected and `auto` takes every GPU, the monitor's one too
    (it overheated that way on the friend's PC on 2026-11-03): name the devices there. If there are none, it falls back to the discrete GPUs with displays, then other
    GPUs, never a CPU implementation.
- **`all`:** every usable device.
- **`#N`:** Vulkan device N.
- **Anything else:** every device whose name contains it (`5060`, `Radeon`).
- **Several:** separate them with commas.

`experimentDiskDepth6 --gpu-list` (or `build_gpu\vk_devices.exe`) prints every Vulkan device,
whether it can run the kernels and why not, and which ones `auto` takes. It opens nothing. On
this machine:

```
#0 NVIDIA GeForce GTX 1660 SUPER (discrete, drives a display, NVIDIA 610.88, subgroup 32, 6.3 GB, 5.5 GB free): usable
#1 NVIDIA GeForce RTX 5060 Ti (discrete, NVIDIA 610.88, subgroup 32, 16.8 GB, 16.0 GB free): usable [auto]
```

The "free" figure is this process's budget (`VK_EXT_memory_budget`). It does not subtract what
other programs hold (the queue job had 8 GB at the time), so keep `--gpu-vram-gb` (default
12.5) when the GPU is shared.

## Several GPUs

Every chosen device is used.
- **Shared work:** the stream, checksums, audit, child counts and candidate check hand out
  seeds in chunks from one counter, so a faster card takes more. Leaves that pass go to the
  solver's sink one batch at a time.
- **Bloom blocks:** each pass's blocks go to every device. Each device needs the whole filter,
  so more GPUs make passes faster, not fewer.
- **First device only:** the goal-side index build (`mbgpu_child_keys`, `mbgpu_build_index`)
  and `mbgpu_bounds`, which need one sorted set.

## How it differs from the CUDA kernels (not in what is computed)

- **Buffers:** every buffer is reached by device address (`GL_EXT_buffer_reference`). There
  are no descriptor sets; addresses travel in push constants and in one `Globals` block.
- **Node storage:** a node's previous move and seed index share one u32
  (`prev << 24 | seed within the chunk`; a chunk is at most 8 M seeds), so nodes are 20 bytes
  and no 8-bit storage is needed.
- **Counters:** the counters are 32-bit and set by the host before each launch; the host adds
  them up.
- **Checksums:** the per-seed checksums and partition counts are 64-bit sums built from two
  32-bit atomics with a carry, so no 64-bit atomics are needed.
- **Subgroups:** subgroup operations replace warp shuffles, for any subgroup size that divides
  256 (NVIDIA 32, AMD 64 or 32, Intel 8-32). Staging is per subgroup, 5 slots per invocation.
- **Sorting:** the goal-side keys are sorted by an LSD radix sort, 16 passes of 4 bits. Each
  pass is a histogram per 4096-key tile, a scan, and a stable scatter that ranks each round of
  256 keys by a workgroup prefix over packed digit counters. Duplicates are then removed by
  flags and a scan.
- **Short submissions:** drivers end a compute job that runs too long, and the device is lost
  (`VkResult -4`). Linux amdgpu (kernel 7.0) allows about 2 s: on the friend's Polaris GPUs one
  radix pass over 348 M keys (histogram, scan and scatter in one submission) took 2029 ms and
  was ended (2026-11-04, a 2x2 depth-7 index build). Now the histogram, the scan and the scatter
  are separate submissions, the histogram and the scatter go in chunks of tiles, and the Bloom
  inserts in chunks of keys; each chunk size is halved above 250 ms and doubled below 60 ms, per
  device, at run time. Any submission over 1 s is reported on stderr with the kernels it ran
  (`mbvk: <device>: one submission took N ms (...)`; `MBVK_SLOW_MS` sets the threshold, 0 turns
  it off). Windows has a similar limit (TDR, 2 s by default).
- **Big buffers:** a buffer is one allocation. When a driver caps allocations
  (`maxMemoryAllocationSize`, `maxBufferSize`), `mbgpu_vram_left` and
  `mbgpu_child_key_capacity` shrink to fit, so the solver plans smaller passes instead of
  failing. NVIDIA reports no cap.

Requirements, checked per device (a device that lacks one is listed with the reason):
- Vulkan 1.2
- `shaderInt64`
- `bufferDeviceAddress`
- subgroup basic, arithmetic and ballot in compute shaders
- 28 KB of shared memory

## Checks before the first real run (machine idle, queue paused)

1. `build_gpu\test_vk_stream.exe build_gpu\vectors.bin --device 5060` must print PASS for
   every case: the per-seed checksums against the CPU stream, the filter checks, the audit
   cases, and the child counts and keys (the sort). This covers normal and 2x2 boards and the
   6-move cases.
2. Run a solved puzzle on Vulkan and look for `EXACT MATCH` and `IDENTICAL SET`:
   ```bat
   build_test\experimentDiskDepth6.exe 16-2 --seed-depth 5 --stream-depth 4 --out %TEMP%\mbvk --gpu vulkan:5060 --threads 6 --solutions build\levels_final\16-2_c15_366.txt
   build_test\experimentDiskDepth6.exe 17-4 --seed-depth 4 --out %TEMP%\mbvk --gpu vulkan:5060 --threads 6 --solutions build\levels_final\17-4_c13_166.txt
   ```
   The first is 16-2 at 15 moves as `<5,4,6>`, a normal board; the second is 17-4 at 13 moves
   as `<4,3,6>`, a 2x2 board.
3. Time the same runs with `--gpu cuda:5060`. The Vulkan port has not been tuned with Nsight
   the way the CUDA kernels were; expect it to be slower on NVIDIA. CUDA stays the default
   there.

## Measured (2026-11-02, Radeon Pro WX 7100 = one Polaris 10 GPU of a Radeon Pro Duo)

| Run | One GPU | Three GPUs | Four GPUs | RTX 5060 Ti (CUDA) |
|---|---|---|---|---|
| bench 16-2, 200 k seeds, 3 GB filter (first port) | 8.04 s | | | 0.74 s |
| the same, kernels laid out for GCN | 4.43 s | | | 0.74 s |
| the same, only crossed lanes at the limit tested | 3.27 s | | | 0.74 s |
| 16-2 at 15, stream + probe (first port) | 127 s | 45.4 s | | 12.6 s |
| the same, kernels laid out for GCN | | 27.1 s | 21.2 s | 12.6 s |
| the same, only crossed lanes at the limit tested | | | 13.1 s (16.6 s with the filter copies) | 12.6 s |
| 17-4 at 13 (2x2), stream + probe | | 15.1-15.5 s | | |

What made the difference on GCN (64-wide waves, 256 VGPRs per lane per SIMD, 64 KB LDS per
CU): registers. The first port kept a lane's 5 child boards (and keys) in arrays, as the CUDA
kernel does, and RADV gave the node kernels 128 VGPRs: 2 waves per SIMD. Keeping a lane's
children as a 5-bit mask and rebuilding each child from its parent when it is written or
tested, and staging 4 bytes per child (parent slot, move) with the parents in LDS instead of
20-byte copies, brought them to 64 VGPRs (normal boards) and 40 (2x2), LDS 28 KB -> 12.8 KB.
Normal boards got 1.8x faster; 2x2 boards did not change (their leaves are bound by the random
Bloom reads: about 1 G per second per GPU, the same with 16-byte loads or with five reads in
flight per invocation, so GDDR5's random-access rate).

Then less work per node. Running each part twice (same results, so the extra time is that
part's cost) showed the prune test of each lane's children at two thirds of the time, the
subgroup scans and count words at a few percent. A move changes one cell of every lane it
crosses, so a crossed lane's distance changes by at most 1: only crossed lanes already at the
room left can be pushed over it, and only those are looked up now (usually one or two of six,
where all six were). 1.36x more, exact (every audit identical). The CUDA kernel tests all six
too, so the same change should speed it up. Four GPUs scale linearly (1016 M kept nodes/s
against 252 M on one, before this change); the clocks hold at 1243 MHz at 80-97 W (cap 98 W). Register use per kernel: `RADV_DEBUG=shaderstats` with
`MESA_SHADER_CACHE_DISABLE=true` (RADV caches compiled kernels and only prints new ones).

## On another PC

Linux: build with CMake (`-DMB_MARCH=native -DMB_OUTPUT_DIR=<dir>`) and `build.sh`
(glslc, libvulkan-dev, g++), which writes `libmbvk.so` next to the solver. The friend's-PC
setup and runs are in `docs/changes-2026-11-03.md`. Windows:

The other PC needs only a GPU driver. Copy over:
- `experimentDiskDepth6.exe`: the portable build, `-DMB_MARCH=x86-64-v3`, which needs AVX2 and
  BMI2. A CPU without them needs a `x86-64-v2` build.
- `mbvk.dll`, in the same folder.

Then:

```bat
experimentDiskDepth6.exe --gpu-list
experimentDiskDepth6.exe 16-2 --seed-depth 5 --stream-depth 4 --out work --gpu auto --threads 6 --solutions 16-2_c15_366.txt
```

First run a solved puzzle with `--solutions`, as above, on that PC's GPUs. Then run
`test_vk_stream.exe vectors.bin --device auto` there. A driver that miscompiles a kernel shows
up as a FAIL in either check.
