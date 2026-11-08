#!/bin/bash
# python/scheduler/fleet_worker/setup.sh: prepares a rented Linux GPU machine for fleet work units.
# mbq copies the package to ~/mbfleet (this script, run_unit.sh, status.sh; pkg/ = the sources) and starts
# this script there in the background. It installs what is missing (Ubuntu: g++, cmake, ninja,
# python3), builds the solver for this CPU and the CUDA module for this GPU, runs the GPU
# self-test (the GPU's stream against the CPU's, build_gpu/vectors.bin) and writes setup.ok
# (machine facts, one key=value per line) or setup.failed (the reason). Log: setup.log.
# Best image: nvidia/cuda:12.8.1-devel-ubuntu24.04 (nvcc, g++ 13, cmake 3.28 from apt).
set -u
cd "$(dirname "$0")"
ROOT=$(pwd)
exec > "$ROOT/setup.log" 2>&1
rm -f "$ROOT/setup.ok" "$ROOT/setup.failed"
fail() { echo "FAILED: $*"; echo "$*" > "$ROOT/setup.failed"; exit 1; }
echo "setup started $(date '+%F %T') on $(hostname)"

SUDO=""
[ "$(id -u)" != 0 ] && command -v sudo >/dev/null && SUDO="sudo -n"
missing=""
command -v g++ >/dev/null || missing="$missing g++"
command -v ninja >/dev/null || missing="$missing ninja-build"
command -v python3 >/dev/null || missing="$missing python3"
command -v cmake >/dev/null || missing="$missing cmake"
if [ -n "$missing" ]; then
    command -v apt-get >/dev/null || fail "missing:$missing (no apt-get)"
    $SUDO apt-get update -q && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -q $missing \
        || fail "apt-get install$missing"
fi
# CMake >= 3.24 (Ubuntu 22.04 has 3.22): the pip one if apt's is older
cmv=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [ "$(printf '%s\n3.24\n' "$cmv" | sort -V | head -1)" != "3.24" ]; then
    python3 -m pip install -q --user cmake || python3 -m pip install -q --user --break-system-packages cmake \
        || fail "cmake $cmv is older than 3.24 and pip could not install a newer one"
    export PATH="$HOME/.local/bin:$PATH"
fi
gv=$(g++ -dumpversion | cut -d. -f1)
[ "$gv" -ge 12 ] || fail "g++ $gv is too old (C++23 needs 12+; use an Ubuntu 24.04 image)"
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || fail "no nvcc (use a CUDA devel image, e.g. nvidia/cuda:12.8.1-devel-ubuntu24.04)"
nvidia-smi >/dev/null || fail "nvidia-smi does not work: no usable NVIDIA GPU"

echo "== solver"
cmake -S pkg -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMB_MARCH=native -DMB_OUTPUT_DIR="$ROOT/bin" \
    -DSTATIC_MINGW_RUNTIME=OFF || fail "cmake configure"
cmake --build build --target experimentDiskDepth6 -j "$(nproc)" || fail "solver build"
echo "== CUDA module"
OUT="$ROOT/bin" sh pkg/src/gpu/build.sh || fail "CUDA module build"
echo "== GPU self-test"
# --device "": the first CUDA device (rented machines have one GPU, of any model)
"$ROOT/bin/test_gpu_stream" pkg/build_gpu/vectors.bin --device "" 2>&1 | tee "$ROOT/selftest.log"
grep -q PASS "$ROOT/selftest.log" || fail "GPU self-test did not PASS (selftest.log)"

gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i 0)
{
    echo "gpu=$gpu"
    echo "gpus=$(nvidia-smi -L | wc -l)"
    echo "cores=$(nproc)"
    # RAM this container may use: MemTotal, capped by the cgroup limit (v2 or v1)
    lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
    echo "ram_gb=$(awk -v lim="$lim" '/MemTotal/ {t = $2 * 1024; if (lim ~ /^[0-9]+$/ && lim + 0 < t) t = lim; printf "%.0f", t / 1073741824}' /proc/meminfo)"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "disk_free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)"
    echo "finished=$(date '+%F %T')"
} > "$ROOT/setup.ok.part"
mv "$ROOT/setup.ok.part" "$ROOT/setup.ok"
echo "setup done $(date '+%F %T')"
