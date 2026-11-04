#!/bin/sh
# src/gpu/vulkan/build.sh: builds the Vulkan GPU module on Linux with glslc + g++ (build.cmd is
# the Windows one). Outputs in <repo>/build_gpu, or $OUT:
#   libmbvk.so      the module the solver loads at run time (any Vulkan 1.2 GPU)
#   test_vk_stream  the GPU-vs-CPU stream check (src/gpu/test_gpu_stream.cu, on mbvk)
#   vk_devices      lists the Vulkan devices and which ones --gpu auto takes
#   vk_spv/*.inc    the compiled kernels, embedded into all three
# Needs glslc (Ubuntu: glslc), the Vulkan headers (libvulkan-dev) and g++.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$REPO/build_gpu}
SPV=$OUT/vk_spv
SH=$HERE/shaders
mkdir -p "$SPV"

G="glslc --target-env=vulkan1.2 -O -mfmt=c"
$G -DLEAF=0 "$SH/expand.comp" -o "$SPV/expand0.inc"
$G -DLEAF=1 "$SH/expand.comp" -o "$SPV/expand1.inc"
$G -DLEAF=0 "$SH/expand_fat.comp" -o "$SPV/fat0.inc"
$G -DLEAF=1 "$SH/expand_fat.comp" -o "$SPV/fat1.inc"
$G "$SH/audit.comp" -o "$SPV/audit.inc"
$G "$SH/bound.comp" -o "$SPV/bound.inc"
for k in 1 2 3 4 5 6 7 8 9 10; do
    $G -DKERNEL=$k "$SH/keys.comp" -o "$SPV/keys$k.inc"
done

FLAGS="-O2 -std=c++17 -Wall -I$REPO -I$SPV -pthread"
g++ $FLAGS -shared -fPIC "$HERE/mbvk.cpp" -o "$OUT/libmbvk.so" -ldl
g++ $FLAGS -x c++ "$REPO/src/gpu/test_gpu_stream.cu" -x none "$HERE/mbvk.cpp" -o "$OUT/test_vk_stream" -ldl
g++ $FLAGS "$HERE/vk_devices.cpp" "$HERE/mbvk.cpp" -o "$OUT/vk_devices" -ldl
echo "built $OUT/libmbvk.so, test_vk_stream and vk_devices"
