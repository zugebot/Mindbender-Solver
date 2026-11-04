// src/gpu/vulkan/vk_devices.cpp: prints every Vulkan device, whether the kernels can run on it
// and which ones --gpu auto takes (mbvk_list_devices). Opens no device, allocates nothing.
// Built by build.cmd as build_gpu/vk_devices.exe.

#include "src/gpu/vulkan/mbvk.h"

#include <cstdio>
#include <vector>

int main() {
    std::vector<char> text(1 << 16);
    if (mbvk_list_devices(text.data(), static_cast<int>(text.size())) != 0) {
        std::fprintf(stderr, "%s\n", mbgpu_last_error());
        return 1;
    }
    std::fputs(text.data(), stdout);
    return 0;
}
