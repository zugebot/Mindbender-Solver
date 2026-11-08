#!/bin/sh
# src/gpu/build.sh: builds the CUDA GPU module on Linux with nvcc + g++ (build.cmd is the Windows
# one). Outputs in <repo>/build_gpu, or $OUT:
#   libmbgpu.so      the module the solver loads at run time
#   test_gpu_stream  the GPU-vs-CPU stream check (see README.md)
# Needs the CUDA Toolkit (12.8 or newer for RTX 50-series cards) and g++.
# ARCH picks the GPU code: "native" (default: the GPUs in this machine) or e.g.
# "-gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120".
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-$REPO/build_gpu}
mkdir -p "$OUT"
case "${ARCH:-native}" in
    native) ARCHFLAGS="-arch=native" ;;
    *) ARCHFLAGS="$ARCH" ;;
esac

FLAGS="-O3 -std=c++17 $ARCHFLAGS -I$REPO -Xcompiler -fPIC,-O2 -lineinfo"
nvcc $FLAGS -shared -o "$OUT/libmbgpu.so" "$REPO/src/gpu/mbgpu.cu"
nvcc $FLAGS -o "$OUT/test_gpu_stream" "$REPO/src/gpu/test_gpu_stream.cu" "$REPO/src/gpu/mbgpu.cu"
echo "built $OUT/libmbgpu.so and test_gpu_stream"
