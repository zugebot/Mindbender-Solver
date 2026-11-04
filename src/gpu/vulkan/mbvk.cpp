// src/gpu/vulkan/mbvk.cpp
//
// The Vulkan GPU module (mbvk.dll): the same C interface as the CUDA module
// (src/gpu/mbgpu.h), so the solver loads either one, on any GPU with a Vulkan 1.2 driver
// (NVIDIA, AMD, Intel). The kernels are GLSL ports of src/gpu/mbgpu.cu
// (src/gpu/vulkan/shaders/, compiled to SPIR-V by build.cmd and embedded here); the host
// logic below follows mbgpu.cu's step for step (level buffers, chunks sized from the measured
// children per node, retries, the output flush). See src/gpu/vulkan/README.md.
//
// What Vulkan adds:
// - Device choice (mbgpu_open's name): "auto" = every discrete GPU that drives no display
//   (else the display GPUs, else integrated ones), "all", "#N" (Vulkan device index N), or
//   a part of the name ("5060", "Radeon"); several comma-separated. Every chosen device is
//   used: seeds are handed out in chunks from a shared queue, so faster devices take more.
//   The goal-side index build (mbgpu_child_keys, mbgpu_build_index) and mbgpu_bounds run on
//   the first device; the Bloom filter is copied to every device.
// - The VRAM cap per device is the smaller of the caller's cap and the free VRAM the driver
//   reports (VK_EXT_memory_budget) minus 256 MB, as the CUDA module does.
// - Requirements, checked per device with the reason given when one is skipped: Vulkan 1.2,
//   shaderInt64, bufferDeviceAddress, subgroup basic/arithmetic/ballot in compute shaders,
//   a subgroup size dividing 256, 28 KB of shared memory. No 64-bit atomics, no 8-bit storage.
// - The Vulkan loader (vulkan-1.dll, libvulkan.so.1) is loaded at run time, so the module loads
//   on a machine without Vulkan and mbgpu_open then says why it cannot run.
// - Windows (mbvk.dll, build.cmd) and Linux (libmbvk.so, build.sh). Which GPUs drive a display:
//   DXGI adapters with outputs on Windows, connected DRM outputs' PCI addresses on Linux.

#define MBGPU_BUILD_DLL
#include "src/gpu/vulkan/mbvk.h"

#define VK_NO_PROTOTYPES
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi.h>
#else
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#pragma comment(lib, "dxgi.lib")
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

namespace {

    // ---------------- SPIR-V (build.cmd: glslc -mfmt=c) ----------------

    const uint32_t kExpand0[] =
#include "expand0.inc"
            ;
    const uint32_t kExpand1[] =
#include "expand1.inc"
            ;
    const uint32_t kFat0[] =
#include "fat0.inc"
            ;
    const uint32_t kFat1[] =
#include "fat1.inc"
            ;
    const uint32_t kAudit[] =
#include "audit.inc"
            ;
    const uint32_t kBound[] =
#include "bound.inc"
            ;
    const uint32_t kKeys1[] =
#include "keys1.inc"
            ;
    const uint32_t kKeys2[] =
#include "keys2.inc"
            ;
    const uint32_t kKeys3[] =
#include "keys3.inc"
            ;
    const uint32_t kKeys4[] =
#include "keys4.inc"
            ;
    const uint32_t kKeys5[] =
#include "keys5.inc"
            ;
    const uint32_t kKeys6[] =
#include "keys6.inc"
            ;
    const uint32_t kKeys7[] =
#include "keys7.inc"
            ;
    const uint32_t kKeys8[] =
#include "keys8.inc"
            ;
    const uint32_t kKeys9[] =
#include "keys9.inc"
            ;
    const uint32_t kKeys10[] =
#include "keys10.inc"
            ;

    enum Pipe {
        P_EXPAND0, P_EXPAND1, P_FAT0, P_FAT1, P_AUDIT, P_BOUND,
        P_RADIX_HIST, P_RADIX_SCATTER, P_SCAN_LOCAL, P_SCAN_ADD, P_UNIQUE_FLAGS, P_UNIQUE_SCATTER,
        P_PART_BOUNDS, P_FINGERPRINTS, P_BUCKET_STARTS, P_BLOOM, P_COUNT
    };

    struct Spv {
        const uint32_t* code;
        size_t bytes;
    };

    const Spv kSpv[P_COUNT] = {
            {kExpand0, sizeof(kExpand0)}, {kExpand1, sizeof(kExpand1)}, {kFat0, sizeof(kFat0)},
            {kFat1, sizeof(kFat1)},       {kAudit, sizeof(kAudit)},     {kBound, sizeof(kBound)},
            {kKeys1, sizeof(kKeys1)},     {kKeys2, sizeof(kKeys2)},     {kKeys3, sizeof(kKeys3)},
            {kKeys4, sizeof(kKeys4)},     {kKeys5, sizeof(kKeys5)},     {kKeys6, sizeof(kKeys6)},
            {kKeys7, sizeof(kKeys7)},     {kKeys8, sizeof(kKeys8)},     {kKeys9, sizeof(kKeys9)},
            {kKeys10, sizeof(kKeys10)},
    };

    constexpr u32 NO_PREV = 255;
    constexpr u64 MAX_CHILDREN = 60;
    constexpr u64 FAT_BOOL_BIT = 1ULL << 57;
    constexpr u32 BLOCK = 256;
    constexpr u64 TILE = 4096;                   // keys.comp: keys per workgroup
    constexpr u32 SHARED_NEEDED = 28 * 1024;     // expand.comp's inner kernel: stage + tables
    constexpr u64 STAGING_BYTES = 64ULL << 20;
    constexpr u64 RESERVE = 256ULL << 20;        // VRAM kept clear of the driver's free figure

    enum LeafMode : int { LEAF_FILTER = 0, LEAF_CHECKSUM = 1, LEAF_KEYS = 2, LEAF_COUNTS = 3 };

    // shaders/common.glsl: Counters (u32, set per launch)
    struct Counters {
        u32 next, leaves, inRange, passed, edges, violations, overLimit, overflow;
    };

    // shaders/common.glsl: Globals (std430)
    struct Globals {
        u64 rowTarget[6];
        u64 colTarget[6];
        u64 cnt, sum, xr;
        u64 bloom, match, matchCount, blocksPerPartition;
        u64 ob1, ob2, okey, oseed;
        u64 partCounts;
        u64 bloomBuf[8];           // the Bloom blocks, bloomPartsPerBuf partitions per buffer
        int32_t limit, mode;
        u32 pBegin, pEnd, oCap, bloomPartsPerBuf, pad1, pad2;
        u32 mc[288];
        u32 mr[288];
    };
    static_assert(offsetof(Globals, cnt) == 96, "std430 layout");
    static_assert(offsetof(Globals, bloomBuf) == 192, "std430 layout");
    static_assert(offsetof(Globals, limit) == 256, "std430 layout");
    static_assert(offsetof(Globals, mc) == 288, "std430 layout");
    static_assert(sizeof(Globals) % 4 == 0 && sizeof(Globals) <= 65536, "vkCmdUpdateBuffer limit");

    // shaders/expand_push.glsl
    struct ExpandPush {
        u64 globals, counters, inB1, inB2, inPk, outB1, outB2, outPk;
        u32 n, outCap;
        int32_t rem;
        u32 prune, seedBase;
    };

    // shaders/bound.comp
    struct BoundPush {
        u64 globals, states, h;
        u32 n;
    };

    // shaders/keys.comp
    struct KeysPush {
        u64 a0, a1, a2, a3, u0, u1, u2;
        u32 w0, w1;
    };
    static_assert(sizeof(ExpandPush) <= 128 && sizeof(KeysPush) <= 128 && sizeof(BoundPush) <= 128, "push constants");

    // ---------------- errors ----------------

    thread_local std::string t_error;   // the failing step's message, per thread
    std::string g_error;                // what mbgpu_last_error returns
    std::mutex g_sinkMutex;             // sink calls from several device threads, one at a time

    bool fail(const std::string& what) {
        t_error = what;
        return false;
    }

    int finish(const bool ok) {
        g_error = ok ? std::string() : (t_error.empty() ? std::string("unknown error") : t_error);
        return ok ? 0 : 1;
    }

#define VKC(call)                                                                                  \
    do {                                                                                           \
        const VkResult r_ = (call);                                                                \
        if (r_ != VK_SUCCESS) {                                                                    \
            return fail(std::string(#call) + " failed: VkResult " + std::to_string(static_cast<int>(r_))); \
        }                                                                                          \
    } while (0)

    // ---------------- Vulkan entry points (loaded from the Vulkan loader at run time) ----------------

#define MBVK_FNS(X)                                                                                \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)           \
    X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2)                              \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceMemoryProperties2)                 \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkEnumerateDeviceExtensionProperties)            \
    X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer)  \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory)     \
    X(vkMapMemory) X(vkUnmapMemory) X(vkGetBufferDeviceAddress) X(vkCreateShaderModule)            \
    X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                  \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateCommandPool)                        \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)                    \
    X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkCmdBindPipeline) X(vkCmdPushConstants)       \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdFillBuffer)                 \
    X(vkCmdUpdateBuffer) X(vkQueueSubmit) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences)    \
    X(vkResetFences) X(vkDeviceWaitIdle)

#define MBVK_DECLARE(f) PFN_##f f = nullptr;
    MBVK_FNS(MBVK_DECLARE)
#undef MBVK_DECLARE

    void* g_lib = nullptr;
    VkInstance g_instance = VK_NULL_HANDLE;

    bool loadVulkan() {
        if (g_instance != VK_NULL_HANDLE) {
            return true;
        }
#ifdef _WIN32
        const char* loader = "vulkan-1.dll";
        if (g_lib == nullptr) {
            g_lib = reinterpret_cast<void*>(LoadLibraryA(loader));
        }
        void* entry = g_lib != nullptr ? reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(g_lib),
                                                                                  "vkGetInstanceProcAddr"))
                                       : nullptr;
#else
        const char* loader = "libvulkan.so.1";
        if (g_lib == nullptr) {
            g_lib = dlopen(loader, RTLD_NOW | RTLD_LOCAL);
        }
        void* entry = g_lib != nullptr ? dlsym(g_lib, "vkGetInstanceProcAddr") : nullptr;
#endif
        if (g_lib == nullptr) {
            return fail(std::string(loader) + " not found (no Vulkan driver installed)");
        }
        const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(entry);
        if (gipa == nullptr) {
            return fail(std::string(loader) + " has no vkGetInstanceProcAddr");
        }
        const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
        const auto enumVersion =
                reinterpret_cast<PFN_vkEnumerateInstanceVersion>(gipa(nullptr, "vkEnumerateInstanceVersion"));
        u32 version = VK_API_VERSION_1_0;
        if (enumVersion != nullptr) {
            enumVersion(&version);
        }
        if (createInstance == nullptr || version < VK_API_VERSION_1_2) {
            return fail("the Vulkan loader is older than 1.2 (update the GPU driver)");
        }
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "mbvk";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        VKC(createInstance(&ci, nullptr, &g_instance));
#define MBVK_LOAD(f)                                                                               \
        f = reinterpret_cast<PFN_##f>(gipa(g_instance, #f));                                       \
        if (f == nullptr) {                                                                        \
            return fail("Vulkan entry point " #f " missing");                                      \
        }
        MBVK_FNS(MBVK_LOAD)
#undef MBVK_LOAD
        return true;
    }

    // ---------------- device discovery ----------------

    // Keys of the adapters that drive a display, to leave them to the desktop: on Windows the
    // DXGI adapters with an output (their LUID in hex), on Linux the PCI addresses of the DRM
    // cards with a connected output ("0000:06:00.0").
#ifdef _WIN32
    std::string hexKey(const uint8_t* bytes, const int n) {
        std::string s;
        char b[3];
        for (int i = 0; i < n; ++i) {
            std::snprintf(b, sizeof(b), "%02x", bytes[i]);
            s += b;
        }
        return s;
    }

    std::vector<std::string> displayAdapters() {
        std::vector<std::string> out;
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            return out;
        }
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1* adapter = nullptr;
            if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            IDXGIOutput* output = nullptr;
            if (SUCCEEDED(adapter->EnumOutputs(0, &output))) {
                DXGI_ADAPTER_DESC1 desc{};
                adapter->GetDesc1(&desc);
                out.push_back(hexKey(reinterpret_cast<const uint8_t*>(&desc.AdapterLuid), sizeof(LUID)));
                output->Release();
            }
            adapter->Release();
        }
        factory->Release();
        return out;
    }
#else
    std::vector<std::string> displayAdapters() {
        namespace fs = std::filesystem;
        std::vector<std::string> out;
        std::error_code ec;
        for (const fs::directory_entry& e : fs::directory_iterator("/sys/class/drm", ec)) {
            const std::string name = e.path().filename().string();   // card1-DP-2
            const std::size_t dash = name.find('-');
            if (name.rfind("card", 0) != 0 || dash == std::string::npos) {
                continue;
            }
            std::ifstream status(e.path() / "status");
            std::string state;
            if (!(status >> state) || state != "connected") {
                continue;
            }
            const fs::path dev = fs::read_symlink(e.path().parent_path() / name.substr(0, dash) / "device", ec);
            if (!ec) {
                out.push_back(dev.filename().string());
            }
        }
        return out;
    }
#endif

    struct Candidate {
        VkPhysicalDevice pd = VK_NULL_HANDLE;
        int index = -1;
        std::string name, driver;
        VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
        bool display = false;
        std::string whyNot;   // empty: usable
        u32 subgroup = 0, family = 0, maxGroupsX = 0, maxGroupsY = 0;
        VkPhysicalDeviceMemoryProperties mem{};
        int heap = -1;    // the largest device-local heap (VRAM)
        u64 heapBytes = 0, freeBytes = 0, maxAlloc = 0;
    };

    const char* typeName(const VkPhysicalDeviceType t) {
        switch (t) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
            default: return "other";
        }
    }

    bool hasExtension(const VkPhysicalDevice pd, const char* name) {
        u32 n = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> ext(n);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, ext.data());
        for (const VkExtensionProperties& e : ext) {
            if (std::strcmp(e.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    }

    // The memory type for a buffer whose allowed types are `bits`: device memory in the VRAM heap
    // (preferring a type the host cannot map), or host-visible coherent memory for staging and
    // readback (preferring cached, then not device-local). Drivers list look-alike types that
    // some buffers cannot use (Mesa RADV), so this is asked per buffer. -1: none.
    int pickMemoryType(const Candidate& c, const u32 bits, const bool host) {
        int best = -1, bestScore = -1;
        for (u32 t = 0; t < c.mem.memoryTypeCount; ++t) {
            if (!(bits & (1u << t))) {
                continue;
            }
            const VkMemoryPropertyFlags f = c.mem.memoryTypes[t].propertyFlags;
            int score = -1;
            if (host) {
                const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if ((f & hv) == hv) {
                    score = ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 2 : 0) +
                            ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 0 : 1);
                }
            } else if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                       static_cast<int>(c.mem.memoryTypes[t].heapIndex) == c.heap) {
                score = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? 0 : 1;
            }
            if (score > bestScore) {
                best = static_cast<int>(t);
                bestScore = score;
            }
        }
        return best;
    }

    std::vector<Candidate> listDevices() {
        std::vector<Candidate> out;
        u32 n = 0;
        if (vkEnumeratePhysicalDevices(g_instance, &n, nullptr) != VK_SUCCESS) {
            return out;
        }
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(g_instance, &n, pds.data());
        const std::vector<std::string> displays = displayAdapters();
        for (u32 i = 0; i < n; ++i) {
            Candidate c;
            c.pd = pds[i];
            c.index = static_cast<int>(i);
            VkPhysicalDeviceProperties base{};
            vkGetPhysicalDeviceProperties(c.pd, &base);
            c.name = base.deviceName;
            c.type = base.deviceType;
            if (base.apiVersion < VK_API_VERSION_1_2) {
                c.whyNot = "its driver offers Vulkan " + std::to_string(VK_VERSION_MAJOR(base.apiVersion)) + "." +
                           std::to_string(VK_VERSION_MINOR(base.apiVersion)) + ", 1.2 needed";
                out.push_back(c);
                continue;
            }
            VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
            VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            sg.pNext = &idp;
            m3.pNext = &sg;
            drv.pNext = &m3;
            p2.pNext = &drv;
            VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
            if (base.apiVersion >= VK_API_VERSION_1_3) {
                idp.pNext = &m4;   // Vulkan 1.3: the largest buffer
            }
            VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
            const bool hasPci = hasExtension(c.pd, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
            if (hasPci) {
                pci.pNext = p2.pNext;
                p2.pNext = &pci;
            }
            vkGetPhysicalDeviceProperties2(c.pd, &p2);
            const VkPhysicalDeviceLimits& lim = p2.properties.limits;
            c.driver = std::string(drv.driverName) + " " + drv.driverInfo;
            c.subgroup = sg.subgroupSize;
            c.maxAlloc = m3.maxMemoryAllocationSize;   // one buffer = one allocation here
            if (base.apiVersion >= VK_API_VERSION_1_3 && m4.maxBufferSize != 0) {
                c.maxAlloc = std::min<u64>(c.maxAlloc, m4.maxBufferSize);
            }
            c.maxGroupsX = lim.maxComputeWorkGroupCount[0];
            c.maxGroupsY = lim.maxComputeWorkGroupCount[1];
            std::string key;
#ifdef _WIN32
            if (idp.deviceLUIDValid) {
                key = hexKey(idp.deviceLUID, VK_LUID_SIZE);
            }
#else
            if (hasPci) {
                char b[32];
                std::snprintf(b, sizeof(b), "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice,
                              pci.pciFunction);
                key = b;
            }
#endif
            for (const std::string& d : displays) {
                c.display = c.display || (!key.empty() && d == key);
            }
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f12;
            vkGetPhysicalDeviceFeatures2(c.pd, &f2);
            const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                               VK_SUBGROUP_FEATURE_BALLOT_BIT;
            if (!f2.features.shaderInt64) {
                c.whyNot = "no 64-bit integers in shaders (shaderInt64)";
            } else if (!f12.bufferDeviceAddress) {
                c.whyNot = "no buffer device addresses";
            } else if (!(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (sg.supportedOperations & ops) != ops) {
                c.whyNot = "no subgroup arithmetic/ballot in compute shaders";
            } else if (c.subgroup < 4 || c.subgroup > 256 || BLOCK % c.subgroup != 0) {
                c.whyNot = "subgroup size " + std::to_string(c.subgroup) + " does not divide 256";
            } else if (lim.maxComputeSharedMemorySize < SHARED_NEEDED || lim.maxComputeWorkGroupInvocations < BLOCK ||
                       lim.maxComputeWorkGroupSize[0] < BLOCK) {
                c.whyNot = "too little shared memory or too small workgroups";
            }
            // a compute queue (prefer one without graphics)
            u32 nf = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nf, nullptr);
            std::vector<VkQueueFamilyProperties> fams(nf);
            vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nf, fams.data());
            int fam = -1;
            for (u32 f = 0; f < nf; ++f) {
                if (fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    if (fam < 0 || !(fams[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                        fam = static_cast<int>(f);
                    }
                }
            }
            if (fam < 0 && c.whyNot.empty()) {
                c.whyNot = "no compute queue";
            }
            c.family = fam < 0 ? 0 : static_cast<u32>(fam);
            // memory: the largest device-local heap, a pure VRAM type in it, a host-visible type
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            const bool hasBudget = hasExtension(c.pd, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            if (hasBudget) {
                mp2.pNext = &budget;
            }
            vkGetPhysicalDeviceMemoryProperties2(c.pd, &mp2);
            const VkPhysicalDeviceMemoryProperties& mp = mp2.memoryProperties;
            int heap = -1;
            for (u32 h = 0; h < mp.memoryHeapCount; ++h) {
                if ((mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                    (heap < 0 || mp.memoryHeaps[h].size > mp.memoryHeaps[heap].size)) {
                    heap = static_cast<int>(h);
                }
            }
            c.mem = mp;
            c.heap = heap;
            if ((heap < 0 || pickMemoryType(c, ~0u, false) < 0 || pickMemoryType(c, ~0u, true) < 0) &&
                c.whyNot.empty()) {
                c.whyNot = "no device-local or host-visible memory type";
            }
            if (heap >= 0) {
                c.heapBytes = mp.memoryHeaps[heap].size;
                c.freeBytes = hasBudget && budget.heapBudget[heap] > budget.heapUsage[heap]
                                      ? budget.heapBudget[heap] - budget.heapUsage[heap]
                                      : c.heapBytes * 9 / 10;
            }
            out.push_back(c);
        }
        return out;
    }

    std::string hexKeyU32(const u32 v) {
        char b[16];
        std::snprintf(b, sizeof(b), "%x", v);
        return b;
    }

    std::string lower(std::string s) {
        for (char& ch : s) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return s;
    }

    // The devices `spec` names (see the top of this file).
    bool chooseDevices(const std::string& spec, std::vector<Candidate>& all, std::vector<const Candidate*>& chosen) {
        auto add = [&](const Candidate& c) {
            for (const Candidate* x : chosen) {
                if (x->index == c.index) {
                    return;
                }
            }
            chosen.push_back(&c);
        };
        std::vector<std::string> tokens;
        std::string cur;
        for (const char ch : spec + ",") {
            if (ch == ',') {
                while (!cur.empty() && cur.back() == ' ') cur.pop_back();
                while (!cur.empty() && cur.front() == ' ') cur.erase(cur.begin());
                tokens.push_back(lower(cur));
                cur.clear();
            } else {
                cur += ch;
            }
        }
        for (const std::string& tok : tokens) {
            if (tok.empty() || tok == "auto") {
                // the first non-empty group: discrete GPUs without a display, discrete GPUs,
                // other GPUs without a display, other GPUs (never a CPU implementation)
                for (int group = 0; group < 4; ++group) {
                    const std::size_t before = chosen.size();
                    for (const Candidate& c : all) {
                        const bool discrete = c.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                        const bool ok = c.whyNot.empty() && c.type != VK_PHYSICAL_DEVICE_TYPE_CPU &&
                                        (group >= 2 || discrete) && (group % 2 == 1 || !c.display);
                        if (ok) {
                            add(c);
                        }
                    }
                    if (chosen.size() > before) {
                        break;
                    }
                }
            } else if (tok == "all") {
                for (const Candidate& c : all) {
                    if (c.whyNot.empty() && c.type != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                        add(c);
                    }
                }
            } else if (tok[0] == '#') {
                const int idx = std::atoi(tok.c_str() + 1);
                if (idx < 0 || idx >= static_cast<int>(all.size())) {
                    return fail("no Vulkan device " + tok);
                }
                if (!all[idx].whyNot.empty()) {
                    return fail("Vulkan device " + tok + " (" + all[idx].name + ") cannot run this: " + all[idx].whyNot);
                }
                add(all[idx]);
            } else {
                bool any = false;
                for (const Candidate& c : all) {
                    if (lower(c.name).find(tok) != std::string::npos) {
                        if (!c.whyNot.empty()) {
                            return fail("Vulkan device " + c.name + " cannot run this: " + c.whyNot);
                        }
                        add(c);
                        any = true;
                    }
                }
                if (!any) {
                    return fail("no Vulkan device named like \"" + tok + "\"");
                }
            }
        }
        if (chosen.empty()) {
            std::string list;
            for (const Candidate& c : all) {
                list += "; #" + std::to_string(c.index) + " " + c.name + (c.whyNot.empty() ? "" : " (" + c.whyNot + ")");
            }
            return fail("no usable Vulkan device" + (list.empty() ? std::string(" (none found)") : ":" + list.substr(1)));
        }
        return true;
    }

    // ---------------- one device ----------------

    struct Buffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        u64 bytes = 0;      // allocated (counted against the cap unless host memory)
        u64 addr = 0;       // device address
        void* map = nullptr;
        bool host = false;
    };

    struct Run;

    struct Gpu {
        Candidate cand;
        VkDevice dev = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipe[P_COUNT] = {};
        u64 cap = 0, used = 0;
        u64 levelCap = 0, outCap = 0;
        Buffer lb1[MBGPU_MAX_DEPTH], lb2[MBGPU_MAX_DEPTH], lpk[MBGPU_MAX_DEPTH];
        Buffer ctr, ctrRead, globals, staging;
        Buffer ob1, ob2, okey, oseed;
        // The pass's Bloom blocks, `partsPerBuf` partitions per buffer: a driver may cap one buffer
        // (Mesa RADV: 4 GB), and a pass's filter is often larger.
        static constexpr u32 MAX_BLOOM_BUFS = 8;
        Buffer bloom[MAX_BLOOM_BUFS];
        u32 bloomBufs = 0, partsPerBuf = 256;
        u64 blocksPerPartition = 0;
        u32 pBegin = 0, pEnd = 256;
        Globals gl{};

        ~Gpu() { destroy(); }

        // ---- memory ----

        bool alloc(Buffer& b, u64 bytes, const bool host = false) {
            bytes = std::max<u64>(bytes, 16);
            if (!host && used + bytes > cap) {
                return fail("VRAM cap reached on " + cand.name + ": " + std::to_string((used + bytes) >> 20) +
                            " MB wanted, cap " + std::to_string(cap >> 20) + " MB");
            }
            if (!host && cand.maxAlloc != 0 && bytes > cand.maxAlloc) {
                return fail("one buffer of " + std::to_string(bytes >> 20) + " MB exceeds " + cand.name +
                            "'s largest allocation (" + std::to_string(cand.maxAlloc >> 20) + " MB)");
            }
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VKC(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(dev, b.buf, &req);
            const int picked = pickMemoryType(cand, req.memoryTypeBits, host);
            if (picked < 0) {
                vkDestroyBuffer(dev, b.buf, nullptr);
                b.buf = VK_NULL_HANDLE;
                return fail("no suitable memory type for a buffer (allowed types 0x" +
                            hexKeyU32(req.memoryTypeBits) + ")");
            }
            const u32 type = static_cast<u32>(picked);
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &flags;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = type;
            const VkResult r = vkAllocateMemory(dev, &ai, nullptr, &b.mem);
            if (r != VK_SUCCESS) {
                vkDestroyBuffer(dev, b.buf, nullptr);
                b.buf = VK_NULL_HANDLE;
                return fail("allocating " + std::to_string(req.size >> 20) + " MB on " + cand.name +
                            " failed: VkResult " + std::to_string(static_cast<int>(r)));
            }
            VKC(vkBindBufferMemory(dev, b.buf, b.mem, 0));
            VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            da.buffer = b.buf;
            b.addr = vkGetBufferDeviceAddress(dev, &da);
            b.bytes = req.size;
            b.host = host;
            if (host) {
                VKC(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
            } else {
                used += b.bytes;
            }
            return true;
        }

        void release(Buffer& b) {
            if (b.buf != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev, b.buf, nullptr);
            }
            if (b.mem != VK_NULL_HANDLE) {
                if (b.map != nullptr) {
                    vkUnmapMemory(dev, b.mem);
                }
                vkFreeMemory(dev, b.mem, nullptr);
                if (!b.host) {
                    used -= b.bytes;
                }
            }
            b = Buffer{};
        }

        // ---- commands (one command buffer, recorded and waited for per step) ----

        // Submissions slower than this are reported on stderr with the shaders they ran: a driver
        // ends a compute job that runs past its timeout (Linux amdgpu: lockup_timeout), which
        // loses the device. MBVK_SLOW_MS overrides (0: off).
        std::string cmdPipes;
        double slowMs = [] {
            const char* v = std::getenv("MBVK_SLOW_MS");
            return v != nullptr ? std::atof(v) : 1000.0;
        }();

        double lastSubmitMs = 0.0;   // the last submission's wall time (sort chunk sizing)
        u64 sortChunkTiles = 1024;   // radix tiles per submission, tuned at run time (sortedChildKeys)
        u64 bloomChunkKeys = 16ULL << 20;   // Bloom inserts per submission, tuned at run time (buildIndex)

        bool begin() {
            cmdPipes.clear();
            VKC(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VKC(vkBeginCommandBuffer(cmd, &bi));
            return true;
        }

        bool submit() {
            VKC(vkEndCommandBuffer(cmd));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            const auto t0 = std::chrono::steady_clock::now();
            VKC(vkQueueSubmit(queue, 1, &si, fence));
            VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
            VKC(vkResetFences(dev, 1, &fence));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            lastSubmitMs = ms;
            if (slowMs > 0 && ms > slowMs) {
                std::fprintf(stderr, "mbvk: %s: one submission took %.0f ms (%s)\n", cand.name.c_str(), ms,
                             cmdPipes.empty() ? "copies" : cmdPipes.c_str());
            }
            return true;
        }

        void barrier() {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                               VK_ACCESS_TRANSFER_WRITE_BIT;
            const VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            vkCmdPipelineBarrier(cmd, st, st, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        void hostBarrier() {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        // `threads` invocations of `p` in workgroups of 256, split over x and y when there are
        // more workgroups than the device's x limit (shaders: groupLinear).
        bool dispatch(const Pipe p, const void* push, const u32 pushBytes, const u64 threads) {
            const u64 groups = (threads + BLOCK - 1) / BLOCK;
            if (groups == 0) {
                return true;
            }
            const u64 x = std::min<u64>(groups, std::max<u32>(1, cand.maxGroupsX));
            const u64 y = (groups + x - 1) / x;
            if (y > cand.maxGroupsY || groups * BLOCK > 0xFFFFFFFFULL) {
                return fail("launch too large: " + std::to_string(threads) + " invocations");
            }
            static const char* const kPipeNames[P_COUNT] = {
                    "expand0", "expand1", "fat0", "fat1", "audit", "bound", "radix_hist", "radix_scatter",
                    "scan_local", "scan_add", "unique_flags", "unique_scatter", "part_bounds", "fingerprints",
                    "bucket_starts", "bloom"};
            cmdPipes += (cmdPipes.empty() ? "" : " ") + std::string(kPipeNames[p]) + "x" + std::to_string(threads);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe[p]);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
            vkCmdDispatch(cmd, static_cast<u32>(x), static_cast<u32>(y), 1);
            return true;
        }

        bool upload(const Buffer& dst, const u64 dstOff, const void* src, const u64 bytes) {
            for (u64 done = 0; done < bytes;) {
                const u64 n = std::min(STAGING_BYTES, bytes - done);
                std::memcpy(staging.map, static_cast<const u8*>(src) + done, n);
                if (!begin()) {
                    return false;
                }
                VkBufferCopy region{0, dstOff + done, n};
                vkCmdCopyBuffer(cmd, staging.buf, dst.buf, 1, &region);
                if (!submit()) {
                    return false;
                }
                done += n;
            }
            return true;
        }

        bool download(void* dst, const Buffer& src, const u64 srcOff, const u64 bytes) {
            for (u64 done = 0; done < bytes;) {
                const u64 n = std::min(STAGING_BYTES, bytes - done);
                if (!begin()) {
                    return false;
                }
                VkBufferCopy region{srcOff + done, 0, n};
                vkCmdCopyBuffer(cmd, src.buf, staging.buf, 1, &region);
                hostBarrier();
                if (!submit()) {
                    return false;
                }
                std::memcpy(static_cast<u8*>(dst) + done, staging.map, n);
                done += n;
            }
            return true;
        }

        bool zero(const Buffer& b, const u64 bytes) {
            if (!begin()) {
                return false;
            }
            vkCmdFillBuffer(cmd, b.buf, 0, (bytes + 3) / 4 * 4, 0);
            barrier();
            return submit();
        }

        // ---- setup ----

        bool create(const Candidate& c, const u64 vramCapBytes) {
            cand = c;
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex = cand.family;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            f12.bufferDeviceAddress = VK_TRUE;
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f12;
            f2.features.shaderInt64 = VK_TRUE;
            VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            di.pNext = &f2;
            di.queueCreateInfoCount = 1;
            di.pQueueCreateInfos = &qi;
            VKC(vkCreateDevice(cand.pd, &di, nullptr, &dev));
            vkGetDeviceQueue(dev, cand.family, 0, &queue);
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = cand.family;
            VKC(vkCreateCommandPool(dev, &pi, nullptr, &pool));
            VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ci.commandPool = pool;
            ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ci.commandBufferCount = 1;
            VKC(vkAllocateCommandBuffers(dev, &ci, &cmd));
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VKC(vkCreateFence(dev, &fi, nullptr, &fence));
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            VKC(vkCreatePipelineLayout(dev, &li, nullptr, &layout));
            for (int p = 0; p < P_COUNT; ++p) {
                VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                mi.codeSize = kSpv[p].bytes;
                mi.pCode = kSpv[p].code;
                VkShaderModule mod = VK_NULL_HANDLE;
                VKC(vkCreateShaderModule(dev, &mi, nullptr, &mod));
                VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                cpi.stage.module = mod;
                cpi.stage.pName = "main";
                cpi.layout = layout;
                const VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe[p]);
                vkDestroyShaderModule(dev, mod, nullptr);
                if (r != VK_SUCCESS) {
                    return fail("building kernel " + std::to_string(p) + " for " + cand.name + " failed: VkResult " +
                                std::to_string(static_cast<int>(r)));
                }
            }
            cap = std::min<u64>(vramCapBytes, cand.freeBytes > RESERVE ? cand.freeBytes - RESERVE : 0);
            // level buffers: 20 B per node; up to 1/8 of the cap for all of them, at most 8 M nodes
            // per level (the seed index within a chunk must fit 24 bits)
            const u64 perNode = 8 + 8 + 4;
            levelCap = std::min<u64>(8ULL << 20, (cap / 8) / (perNode * MBGPU_MAX_DEPTH));
            outCap = std::max<u64>(levelCap, 1ULL << 20);
            if (levelCap < 1024) {
                return fail(cand.name + ": VRAM cap too small");
            }
            for (int d = 0; d < MBGPU_MAX_DEPTH; ++d) {
                if (!alloc(lb1[d], levelCap * 8) || !alloc(lb2[d], levelCap * 8) || !alloc(lpk[d], levelCap * 4)) {
                    return false;
                }
            }
            return alloc(ctr, sizeof(Counters)) && alloc(ctrRead, sizeof(Counters), true) &&
                   alloc(globals, sizeof(Globals)) && alloc(staging, STAGING_BYTES, true) &&
                   alloc(ob1, outCap * 8) && alloc(ob2, outCap * 8) && alloc(okey, outCap * 8) &&
                   alloc(oseed, outCap * 4);
        }

        void destroy() {
            if (dev == VK_NULL_HANDLE) {
                return;
            }
            vkDeviceWaitIdle(dev);
            for (int d = 0; d < MBGPU_MAX_DEPTH; ++d) {
                release(lb1[d]);
                release(lb2[d]);
                release(lpk[d]);
            }
            for (Buffer* b : {&ctr, &ctrRead, &globals, &staging, &ob1, &ob2, &okey, &oseed}) {
                release(*b);
            }
            for (Buffer& b : bloom) {
                release(b);
            }
            for (VkPipeline& p : pipe) {
                if (p != VK_NULL_HANDLE) {
                    vkDestroyPipeline(dev, p, nullptr);
                    p = VK_NULL_HANDLE;
                }
            }
            if (layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, layout, nullptr);
            if (fence != VK_NULL_HANDLE) vkDestroyFence(dev, fence, nullptr);
            if (pool != VK_NULL_HANDLE) vkDestroyCommandPool(dev, pool, nullptr);
            vkDestroyDevice(dev, nullptr);
            dev = VK_NULL_HANDLE;
        }

        bool uploadGlobals() {
            if (!begin()) {
                return false;
            }
            vkCmdUpdateBuffer(cmd, globals.buf, 0, sizeof(Globals), &gl);
            barrier();
            return submit();
        }

        u64 vramLeft() const { return cap > used ? cap - used : 0; }

        // Raw keys one sort can take: two key buffers, flags and scan scratch (~21 B per key;
        // 25 B kept, as the CUDA module) in what is left, each buffer within one allocation.
        u64 keyCapacity(const u64 reserve = 0) const {
            const u64 left = vramLeft() > reserve ? vramLeft() - reserve : 0;
            const u64 margin = 64ULL << 20;
            u64 keys = left > margin ? (left - margin) / 25 : 0;
            if (cand.maxAlloc != 0) {
                keys = std::min<u64>(keys, cand.maxAlloc / 8);
            }
            return std::min<u64>(keys, 0xFFFFFFFFULL);
        }

        bool launch(Run& run, int d, u64 off, u64 m, bool leaf, Counters& c);
        bool flushOutput(Run& run);
        bool processLevel(Run& run, int d, u64 n);
        bool streamAll(Run& run, const u64* states, const u8* prev, u64 count);
        bool checksumRun(Run& run, const u64* states, const u8* prev, u64 count, uint64_t* cntOut,
                         uint64_t* sumOut, uint64_t* xrOut);
        bool recordScan(u64 data, u64 len, u64 scratch);
        bool radixTiles(Pipe p, KeysPush push, u64 tiles);
        bool sortedChildKeys(const u64* states, const u8* prev, u64 count, int depth, u32 p0, u32 p1, u64 reserve,
                             Buffer& a, Buffer& b, u64* unique);
        bool childKeys(const u64* states, const u8* prev, u64 count, int depth, u32 p0, u32 p1, u64* out,
                       u64 outCap, u64* outCount);
        bool buildIndex(const u64* states, const u8* prev, u64 count, int depth, u32 p0, u32 p1, u32 bucketBits,
                        u64 bpp, u16* fpsOut, u64 fpsCap, u32* startsOut, u32* bloomOut, uint64_t* uniqueOut);
        bool bounds(const u64* states, u64 count, int32_t* h);
    };

    // ---------------- the module ----------------

    struct Module {
        bool open = false;
        std::vector<std::unique_ptr<Gpu>> gpus;
        int limit = -1;
        bool targetSet = false;
        u64 rowTarget[6] = {}, colTarget[6] = {};
        u32 mc[288] = {}, mr[288] = {};
    };

    Module g_m;

    // One stream call's state on one device (mbgpu.cu: Run).
    struct Run {
        int depth = 0;
        int prune = 1;
        int audit = 0;
        int mode = LEAF_FILTER;
        mbgpu_sink_fn sink = nullptr;
        void* ctx = nullptr;
        u64 seedOffset = 0;       // this run's first seed in the API call (multi-device chunks)
        mbgpu_stats* stats = nullptr;
        double ratio[MBGPU_MAX_DEPTH] = {60, 60, 60, 60, 60, 60};   // children per node, as measured
        bool fat = false;
        u64 passed = 0;           // leaves (or keys) in the output buffer, committed
        u64 oCap = 0;
        u32 seedBase = 0;         // the current chunk's first seed within this run
        u64 leaves = 0, inRange = 0, edges = 0, violations = 0, overLimit = 0;
        std::vector<u64> hb1, hb2, hkey;
        std::vector<u32> hseed;
    };

    bool Gpu::launch(Run& run, const int d, const u64 off, const u64 m, const bool leaf, Counters& c) {
        Counters init{};
        init.passed = static_cast<u32>(run.passed);
        ExpandPush p{};
        p.globals = globals.addr;
        p.counters = ctr.addr;
        p.inB1 = lb1[d].addr + off * 8;
        p.inB2 = lb2[d].addr + off * 8;
        p.inPk = lpk[d].addr + off * 4;
        if (!leaf) {
            p.outB1 = lb1[d + 1].addr;
            p.outB2 = lb2[d + 1].addr;
            p.outPk = lpk[d + 1].addr;
        }
        p.n = static_cast<u32>(m);
        p.outCap = static_cast<u32>(levelCap);
        p.rem = run.depth - (d + 1);
        p.prune = static_cast<u32>(run.prune);
        p.seedBase = run.seedBase;
        const Pipe which = run.audit ? P_AUDIT : run.fat ? (leaf ? P_FAT1 : P_FAT0) : (leaf ? P_EXPAND1 : P_EXPAND0);
        const u64 threads = run.audit ? m * 12 : m;
        if (!begin()) {
            return false;
        }
        vkCmdUpdateBuffer(cmd, ctr.buf, 0, sizeof(Counters), &init);
        barrier();
        if (!dispatch(which, &p, sizeof(p), threads)) {
            vkEndCommandBuffer(cmd);
            return false;
        }
        barrier();
        VkBufferCopy region{0, 0, sizeof(Counters)};
        vkCmdCopyBuffer(cmd, ctr.buf, ctrRead.buf, 1, &region);
        hostBarrier();
        if (!submit()) {
            return false;
        }
        std::memcpy(&c, ctrRead.map, sizeof(Counters));
        run.stats->launches++;
        if (c.overflow) {
            return fail("buffer overflow on the device");
        }
        return true;
    }

    bool Gpu::flushOutput(Run& run) {
        const u64 n = run.passed;
        if (n == 0) {
            return true;
        }
        run.hb1.resize(n);
        run.hb2.resize(n);
        run.hkey.resize(n);
        run.hseed.resize(n);
        if (!download(run.hb1.data(), ob1, 0, n * 8) || !download(run.hb2.data(), ob2, 0, n * 8) ||
            !download(run.hkey.data(), okey, 0, n * 8) || !download(run.hseed.data(), oseed, 0, n * 4)) {
            return false;
        }
        for (u32& s : run.hseed) {
            s += static_cast<u32>(run.seedOffset);
        }
        if (run.sink != nullptr) {
            std::lock_guard<std::mutex> lock(g_sinkMutex);
            run.sink(run.ctx, run.hb1.data(), run.hb2.data(), run.hkey.data(), run.hseed.data(), n);
        }
        run.stats->passed += n;
        run.passed = 0;
        return true;
    }

    // Expands level d (n nodes in level buffer d) down to the leaves, as mbgpu.cu's
    // processLevel: chunks sized from the children per node measured so far (80% of the next
    // buffer), a chunk whose children overflow it run again smaller; with LEAF_FILTER a leaf
    // chunk whose passing leaves overflow the output buffer is undone (its counters are simply
    // not taken), the leaves before it handed over, and run in halves.
    bool Gpu::processLevel(Run& run, const int d, const u64 n) {
        const int childDepth = d + 1;
        const bool leaf = childDepth == run.depth;
        u64 mMax = n;
        for (u64 off = 0; off < n;) {
            u64 m = std::min(n - off, mMax);
            if (run.audit) {
                m = std::min(m, std::max<u64>(1, levelCap / MAX_CHILDREN));
            } else if (!leaf) {
                const double r = std::max(run.ratio[d], 1e-3);
                m = std::min(m, std::max<u64>(1, static_cast<u64>(0.8 * static_cast<double>(levelCap) / r)));
            }
            Counters c{};
            if (!launch(run, d, off, m, leaf, c)) {
                return false;
            }
            if (!leaf) {
                const double r = static_cast<double>(c.next) / static_cast<double>(m);
                if (c.next > levelCap) {
                    if (run.audit || m == 1) {
                        return fail("level buffer overflow");
                    }
                    run.ratio[d] = std::max(run.ratio[d], r);
                    run.stats->retries++;
                    continue;
                }
                run.ratio[d] = r;
                run.edges += c.edges;
                run.violations += c.violations;
                run.stats->nodes[childDepth] += c.next;
                off += m;
                if (c.next != 0 && !processLevel(run, childDepth, c.next)) {
                    return false;
                }
                continue;
            }
            if (run.mode == LEAF_FILTER && c.passed > run.oCap) {
                if (m == 1) {
                    return fail("output buffer overflow");
                }
                if (!flushOutput(run)) {
                    return false;
                }
                mMax = m / 2;
                run.stats->retries++;
                continue;
            }
            run.passed = c.passed;
            run.leaves += c.leaves;
            run.inRange += c.inRange;
            run.edges += c.edges;
            run.violations += c.violations;
            run.overLimit += c.overLimit;
            off += m;
            if (run.mode == LEAF_FILTER && run.passed > run.oCap / 2 && !flushOutput(run)) {
                return false;
            }
        }
        return true;
    }

    bool Gpu::streamAll(Run& run, const u64* states, const u8* prev, const u64 count) {
        if ((run.prune || run.audit) && !g_m.targetSet) {
            return fail("mbgpu_set_target has not been called");
        }
        if (run.depth < 1 || run.depth > MBGPU_MAX_DEPTH) {
            return fail("depth must be 1.." + std::to_string(MBGPU_MAX_DEPTH));
        }
        if (count > 0xFFFFFFFFULL) {
            return fail("too many seeds for 32-bit seed indices");
        }
        run.fat = count != 0 && (states[0] & FAT_BOOL_BIT) != 0;
        if (run.fat && (run.audit || (run.prune && g_m.limit >= 0))) {
            return fail("2x2 boards: no bound on the GPU (the halved bound prunes nothing; the solver turns it off)");
        }
        std::memcpy(gl.rowTarget, g_m.rowTarget, sizeof(gl.rowTarget));
        std::memcpy(gl.colTarget, g_m.colTarget, sizeof(gl.colTarget));
        std::memcpy(gl.mc, g_m.mc, sizeof(gl.mc));
        std::memcpy(gl.mr, g_m.mr, sizeof(gl.mr));
        gl.limit = g_m.limit;
        gl.mode = run.mode;
        gl.oCap = static_cast<u32>(std::min<u64>(run.oCap, 0xFFFFFFFFULL));
        if (!uploadGlobals()) {
            return false;
        }
        std::vector<u64> hb1, hb2;
        std::vector<u32> hpk;
        for (u64 off = 0; off < count; off += levelCap) {
            const u64 m = std::min(levelCap, count - off);
            hb1.resize(m);
            hb2.resize(m);
            hpk.resize(m);
            for (u64 j = 0; j < m; ++j) {
                hb1[j] = states[2 * (off + j)];
                hb2[j] = states[2 * (off + j) + 1];
                const u32 pv = prev != nullptr ? prev[off + j] : NO_PREV;
                hpk[j] = (pv << 24) | static_cast<u32>(j);
            }
            if (!upload(lb1[0], 0, hb1.data(), m * 8) || !upload(lb2[0], 0, hb2.data(), m * 8) ||
                !upload(lpk[0], 0, hpk.data(), m * 4)) {
                return false;
            }
            run.seedBase = static_cast<u32>(off);
            run.stats->nodes[0] += m;
            if (!processLevel(run, 0, m)) {
                return false;
            }
        }
        if (run.mode == LEAF_FILTER && !flushOutput(run)) {
            return false;
        }
        run.stats->leaves += run.leaves;
        run.stats->inRange += run.inRange;
        return true;
    }

    // Per-seed checksum buffers on the device, copied out and freed at the end.
    bool Gpu::checksumRun(Run& run, const u64* states, const u8* prev, const u64 count, uint64_t* cntOut,
                          uint64_t* sumOut, uint64_t* xrOut) {
        const u64 bytes = std::max<u64>(1, count) * 8;
        Buffer cnt, sum, xr;
        const bool ok = alloc(cnt, bytes) && alloc(sum, bytes) && alloc(xr, bytes) && zero(cnt, bytes) &&
                        zero(sum, bytes) && zero(xr, bytes);
        if (ok) {
            gl.cnt = cnt.addr;
            gl.sum = sum.addr;
            gl.xr = xr.addr;
            run.mode = LEAF_CHECKSUM;
        }
        const bool done = ok && streamAll(run, states, prev, count) && download(cntOut, cnt, 0, count * 8) &&
                          download(sumOut, sum, 0, count * 8) && download(xrOut, xr, 0, count * 8);
        release(cnt);
        release(sum);
        release(xr);
        gl.cnt = gl.sum = gl.xr = 0;
        return done;
    }

    // One radix kernel (histogram or scatter) over `tiles` tiles, in chunks of sortChunkTiles
    // tiles per submission (push.u1 = the chunk's first tile). The chunk size follows the
    // device's measured time: halved above 250 ms, doubled below 60 ms.
    bool Gpu::radixTiles(const Pipe p, KeysPush push, const u64 tiles) {
        for (u64 t0 = 0; t0 < tiles;) {
            const u64 m = std::min(sortChunkTiles, tiles - t0);
            push.u1 = t0;
            if (!begin()) {
                return false;
            }
            if (!dispatch(p, &push, sizeof(push), m * BLOCK)) {
                vkEndCommandBuffer(cmd);
                return false;
            }
            barrier();
            if (!submit()) {
                return false;
            }
            t0 += m;
            if (m == sortChunkTiles) {
                if (lastSubmitMs > 250.0 && sortChunkTiles > 16) {
                    sortChunkTiles /= 2;
                } else if (lastSubmitMs < 60.0 && sortChunkTiles < (1ULL << 20)) {
                    sortChunkTiles *= 2;
                }
            }
        }
        return true;
    }

    // words of scan scratch for scanning len values (each level's block totals)
    u64 scanScratchWords(const u64 len) {
        const u64 blocks = (len + TILE - 1) / TILE;
        return blocks <= 1 ? 1 : blocks + scanScratchWords(blocks);
    }

    // Records an exclusive scan of data[0, len) (u32, in place) into the open command buffer.
    bool Gpu::recordScan(const u64 data, const u64 len, const u64 scratch) {
        if (len == 0) {
            return true;
        }
        const u64 blocks = (len + TILE - 1) / TILE;
        KeysPush p{};
        p.a0 = data;
        p.a1 = scratch;
        p.u0 = len;
        if (!dispatch(P_SCAN_LOCAL, &p, sizeof(p), blocks * BLOCK)) {
            return false;
        }
        barrier();
        if (blocks > 1) {
            if (!recordScan(scratch, blocks, scratch + blocks * 4)) {
                return false;
            }
            KeysPush a{};
            a.a0 = data;
            a.a1 = scratch;
            a.u0 = len;
            if (!dispatch(P_SCAN_ADD, &a, sizeof(a), len)) {
                return false;
            }
            barrier();
        }
        return true;
    }

    // The children's keys in partitions [p0, p1), sorted and unique, left in b (count *unique);
    // a is scratch of the same size. `reserve` bytes are kept free for the caller.
    bool Gpu::sortedChildKeys(const u64* states, const u8* prev, const u64 count, const int depth, const u32 p0,
                              const u32 p1, const u64 reserve, Buffer& a, Buffer& b, u64* unique) {
        const u64 cap = keyCapacity(reserve);
        if (cap < (1ULL << 20)) {
            return fail("too little VRAM left to sort keys");
        }
        if (!alloc(a, cap * 8) || !alloc(b, cap * 8)) {
            return false;
        }
        mbgpu_stats st{};
        Run run;
        run.depth = depth;
        run.prune = 0;
        run.stats = &st;
        run.mode = LEAF_KEYS;
        run.oCap = cap;
        gl.pBegin = p0;
        gl.pEnd = p1;
        gl.okey = a.addr;
        gl.bloom = gl.match = 0;
        const bool streamed = streamAll(run, states, prev, count);
        gl.okey = okey.addr;
        if (!streamed) {
            return false;
        }
        const u64 n = run.passed;
        if (n > cap) {
            return fail("raw keys of partitions [" + std::to_string(p0) + "," + std::to_string(p1) +
                        ") exceed the device buffer");
        }
        *unique = 0;
        if (n == 0) {
            return true;
        }
        const u64 tiles = (n + TILE - 1) / TILE;
        Buffer flags, hist, scratch;
        bool ok = alloc(flags, n * 4) && alloc(hist, 16 * tiles * 4) &&
                  alloc(scratch, scanScratchWords(std::max<u64>(16 * tiles, n)) * 4);
        // LSD radix sort, 16 passes of 4 bits: a -> b -> a ... ends in a. Histogram, scan and
        // scatter are separate submissions, and the histogram and scatter go in tile chunks.
        Buffer* src = &a;
        Buffer* dst = &b;
        for (u32 pass = 0; ok && pass < 16; ++pass) {
            KeysPush h{};
            h.a0 = src->addr;
            h.a1 = hist.addr;
            h.u0 = n;
            h.w0 = 4 * pass;
            h.w1 = static_cast<u32>(tiles);
            KeysPush s{};
            s.a0 = src->addr;
            s.a1 = dst->addr;
            s.a2 = hist.addr;
            s.u0 = n;
            s.w0 = 4 * pass;
            s.w1 = static_cast<u32>(tiles);
            ok = radixTiles(P_RADIX_HIST, h, tiles);
            if (ok) {
                ok = begin() && recordScan(hist.addr, 16 * tiles, scratch.addr);
                if (ok) {
                    barrier();
                    ok = submit();
                } else {
                    vkEndCommandBuffer(cmd);
                }
            }
            ok = ok && radixTiles(P_RADIX_SCATTER, s, tiles);
            std::swap(src, dst);
        }
        // unique: flags, their scan, the first key of each run into b; the count into ctr
        if (ok) {
            KeysPush f{};
            f.a0 = a.addr;
            f.a1 = flags.addr;
            f.u0 = n;
            KeysPush u{};
            u.a0 = a.addr;
            u.a1 = flags.addr;
            u.a2 = b.addr;
            u.a3 = ctr.addr;
            u.u0 = n;
            // three submissions (flags, their scan, the scatter): each stays short
            ok = begin() && dispatch(P_UNIQUE_FLAGS, &f, sizeof(f), n);
            if (ok) {
                barrier();
                ok = submit();
            }
            if (ok) {
                ok = begin() && recordScan(flags.addr, n, scratch.addr);
                if (ok) {
                    barrier();
                    ok = submit();
                }
            }
            if (ok) {
                ok = begin() && dispatch(P_UNIQUE_SCATTER, &u, sizeof(u), n);
                if (ok) {
                    barrier();
                    ok = submit();
                }
            }
        }
        u32 count32 = 0;
        ok = ok && download(&count32, ctr, 0, 4);
        release(flags);
        release(hist);
        release(scratch);
        *unique = count32;
        return ok;
    }

    bool Gpu::childKeys(const u64* states, const u8* prev, const u64 count, const int depth, const u32 p0, const u32 p1,
                        u64* out, const u64 outCap, u64* outCount) {
        Buffer a, b;
        u64 unique = 0;
        bool ok = sortedChildKeys(states, prev, count, depth, p0, p1, 0, a, b, &unique);
        if (ok && unique > outCap) {
            ok = fail("output buffer too small for the unique keys");
        }
        ok = ok && download(out, b, 0, unique * 8);
        release(a);
        release(b);
        if (ok) {
            *outCount = unique;
        }
        return ok;
    }

    // The per-partition device buffers of buildIndex: starts and Bloom blocks of one partition.
    u64 indexPartitionBytes(const u32 bucketBits, const u64 blocksPerPartition) {
        return ((1ULL << bucketBits) + 1) * 4 + blocksPerPartition * 32 + 257 * 8;
    }

    bool Gpu::buildIndex(const u64* states, const u8* prev, const u64 count, const int depth, const u32 p0,
                         const u32 p1, const u32 bucketBits, const u64 bpp, u16* fpsOut, const u64 fpsCap,
                         u32* startsOut, u32* bloomOut, uint64_t* uniqueOut) {
        const u64 dirPer = (1ULL << bucketBits) + 1;
        if (dirPer > 0x7FFFFFFFULL) {
            return fail("too many buckets per partition");
        }
        Buffer a, b, bnd, starts, bl;
        u64 unique = 0;
        bool ok = sortedChildKeys(states, prev, count, depth, p0, p1, indexPartitionBytes(bucketBits, bpp), a, b,
                                  &unique);
        if (ok && unique > fpsCap) {
            ok = fail("fingerprint buffer too small");
        }
        const u32 parts = p1 - p0;
        ok = ok && alloc(bnd, 257 * 8) && alloc(starts, dirPer * 4) && alloc(bl, bpp * 32);
        // partition boundaries in the sorted keys (in b)
        std::vector<u64> bounds(parts + 1);
        if (ok) {
            KeysPush p{};
            p.a0 = b.addr;
            p.a1 = bnd.addr;
            p.u0 = unique;
            p.w0 = p0;
            p.w1 = parts;
            ok = begin() && dispatch(P_PART_BOUNDS, &p, sizeof(p), parts + 1);
            if (ok) {
                barrier();
                ok = submit() && download(bounds.data(), bnd, 0, (parts + 1) * 8);
            }
        }
        // fingerprints of the whole group (in a, which is free now)
        if (ok && unique != 0) {
            KeysPush p{};
            p.a0 = b.addr;
            p.a1 = a.addr;
            p.u0 = unique;
            ok = begin() && dispatch(P_FINGERPRINTS, &p, sizeof(p), (unique + 1) / 2);
            if (ok) {
                barrier();
                ok = submit() && download(fpsOut, a, 0, unique * 2);
            }
        }
        for (u32 i = 0; ok && i < parts; ++i) {
            const u64 lo = bounds[i], hi = bounds[i + 1];
            uniqueOut[i] = hi - lo;
            ok = begin();
            if (!ok) {
                break;
            }
            if (hi > lo) {
                KeysPush s{};
                s.a0 = b.addr;
                s.a1 = starts.addr;
                s.u0 = lo;
                s.u1 = hi;
                s.w0 = bucketBits;
                ok = dispatch(P_BUCKET_STARTS, &s, sizeof(s), dirPer);
            } else {
                vkCmdFillBuffer(cmd, starts.buf, 0, dirPer * 4, 0);
            }
            vkCmdFillBuffer(cmd, bl.buf, 0, bpp * 32, 0);
            barrier();
            if (!ok) {
                vkEndCommandBuffer(cmd);
                break;
            }
            ok = submit();
            // the Bloom inserts in chunks, each its own submission (halved above 250 ms, doubled
            // below 60 ms), so none runs into a driver's job timeout
            for (u64 j0 = lo; ok && j0 < hi;) {
                const u64 m = std::min(bloomChunkKeys, hi - j0);
                KeysPush f{};
                f.a0 = b.addr;
                f.a1 = bl.addr;
                f.u0 = j0;
                f.u1 = j0 + m;
                f.u2 = bpp;
                ok = begin() && dispatch(P_BLOOM, &f, sizeof(f), m);
                if (!ok) {
                    vkEndCommandBuffer(cmd);
                    break;
                }
                barrier();
                ok = submit();
                j0 += m;
                if (m == bloomChunkKeys) {
                    if (lastSubmitMs > 250.0 && bloomChunkKeys > (1ULL << 16)) {
                        bloomChunkKeys /= 2;
                    } else if (lastSubmitMs < 60.0 && bloomChunkKeys < (1ULL << 30)) {
                        bloomChunkKeys *= 2;
                    }
                }
            }
            ok = ok && download(startsOut + i * dirPer, starts, 0, dirPer * 4) &&
                 download(bloomOut + i * bpp * 8, bl, 0, bpp * 32);
        }
        for (Buffer* x : {&a, &b, &bnd, &starts, &bl}) {
            release(*x);
        }
        return ok;
    }

    bool Gpu::bounds(const u64* states, const u64 count, int32_t* h) {
        Buffer s, out;
        bool ok = alloc(s, count * 16) && alloc(out, count * 4) && upload(s, 0, states, count * 16);
        if (ok) {
            std::memcpy(gl.rowTarget, g_m.rowTarget, sizeof(gl.rowTarget));
            std::memcpy(gl.colTarget, g_m.colTarget, sizeof(gl.colTarget));
            ok = uploadGlobals();
        }
        if (ok) {
            BoundPush p{};
            p.globals = globals.addr;
            p.states = s.addr;
            p.h = out.addr;
            p.n = static_cast<u32>(count);
            ok = begin() && dispatch(P_BOUND, &p, sizeof(p), count);
            if (ok) {
                barrier();
                ok = submit() && download(h, out, 0, count * 4);
            }
        }
        release(s);
        release(out);
        return ok;
    }

    // ---------------- several devices ----------------

    // Runs work(gpu, first, n) over seeds [0, total) on every device: chunks from a shared
    // counter, so a faster device takes more. Chunks shrink as the seeds run out (a quarter of
    // what is left per device, at least 1/256 of the share), so the devices finish together.
    // One device: one call with every seed.
    template<typename Work>
    bool splitSeeds(const u64 total, const Work& work) {
        std::vector<std::unique_ptr<Gpu>>& gpus = g_m.gpus;
        if (gpus.size() == 1 || total == 0) {
            return work(*gpus[0], 0, total);
        }
        u64 minLevel = ~0ULL;
        for (const auto& g : gpus) {
            minLevel = std::min(minLevel, g->levelCap);
        }
        const u64 devices = gpus.size();
        const u64 minChunk = std::max<u64>(1, std::min<u64>(minLevel, total / (256 * devices)));
        std::atomic<u64> next{0};
        std::atomic<bool> failed{false};
        std::mutex m;
        std::string firstError;
        std::vector<std::thread> threads;
        for (auto& gp : gpus) {
            Gpu* g = gp.get();
            threads.emplace_back([&, g] {
                for (;;) {
                    u64 a = next.load(), n = 0;
                    do {
                        if (a >= total || failed) {
                            return;
                        }
                        n = std::min<u64>(total - a, std::max<u64>(minChunk, std::min<u64>(minLevel, (total - a) / (4 * devices))));
                    } while (!next.compare_exchange_weak(a, a + n));
                    if (!work(*g, a, n)) {
                        std::lock_guard<std::mutex> lock(m);
                        if (!failed.exchange(true)) {
                            firstError = g->cand.name + ": " + t_error;
                        }
                        return;
                    }
                }
            });
        }
        for (std::thread& t : threads) {
            t.join();
        }
        return failed ? fail(firstError) : true;
    }

    void addStats(mbgpu_stats& to, const mbgpu_stats& s) {
        for (int d = 0; d <= MBGPU_MAX_DEPTH; ++d) {
            to.nodes[d] += s.nodes[d];
        }
        to.leaves += s.leaves;
        to.inRange += s.inRange;
        to.passed += s.passed;
        to.launches += s.launches;
        to.retries += s.retries;
    }

    double since(const std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    bool needOpen() {
        return g_m.open ? true : fail("mbgpu_open has not been called");
    }

    // The probe's and match_children's stream: leaves that pass go to the sink.
    bool filterStream(const u64* states, const u8* prev, const u64 count, const int depth, const int prune,
                      const mbgpu_sink_fn sink, void* ctx, mbgpu_stats* stats, const bool useMatch) {
        const auto t0 = std::chrono::steady_clock::now();
        mbgpu_stats total{};
        std::mutex m;
        const bool ok = splitSeeds(count, [&](Gpu& g, const u64 a, const u64 n) {
            mbgpu_stats st{};
            Run run;
            run.depth = depth;
            run.prune = prune;
            run.sink = sink;
            run.ctx = ctx;
            run.seedOffset = a;
            run.stats = &st;
            run.mode = LEAF_FILTER;
            run.oCap = g.outCap;
            if (!useMatch) {
                g.gl.bloom = g.bloomBufs != 0 ? g.bloom[0].addr : 0;
                for (u32 b = 0; b < Gpu::MAX_BLOOM_BUFS; ++b) {
                    g.gl.bloomBuf[b] = b < g.bloomBufs ? g.bloom[b].addr : 0;
                }
                g.gl.bloomPartsPerBuf = g.partsPerBuf;
                g.gl.blocksPerPartition = g.blocksPerPartition;
                g.gl.pBegin = g.pBegin;
                g.gl.pEnd = g.pEnd;
                g.gl.match = 0;
                g.gl.matchCount = 0;
            }
            g.gl.ob1 = g.ob1.addr;
            g.gl.ob2 = g.ob2.addr;
            g.gl.okey = g.okey.addr;
            g.gl.oseed = g.oseed.addr;
            const bool r = g.streamAll(run, states + 2 * a, prev != nullptr ? prev + a : nullptr, n);
            std::lock_guard<std::mutex> lock(m);
            addStats(total, st);
            return r;
        });
        if (stats != nullptr) {
            addStats(*stats, total);
            stats->seconds += since(t0);
        }
        return ok;
    }

    bool checksumStream(const u64* states, const u8* prev, const u64 count, const int depth, const int audit,
                        uint64_t* cntOut, uint64_t* sumOut, uint64_t* xrOut, mbgpu_stats* stats, Run* totals) {
        const auto t0 = std::chrono::steady_clock::now();
        mbgpu_stats total{};
        std::mutex m;
        const bool ok = splitSeeds(count, [&](Gpu& g, const u64 a, const u64 n) {
            mbgpu_stats st{};
            Run run;
            run.depth = depth;
            run.prune = audit ? 0 : 1;
            run.audit = audit;
            run.stats = &st;
            const bool r = g.checksumRun(run, states + 2 * a, prev != nullptr ? prev + a : nullptr, n, cntOut + a,
                                         sumOut + a, xrOut + a);
            std::lock_guard<std::mutex> lock(m);
            addStats(total, st);
            if (totals != nullptr) {
                totals->leaves += run.leaves;
                totals->edges += run.edges;
                totals->violations += run.violations;
                totals->overLimit += run.overLimit;
            }
            return r;
        });
        if (stats != nullptr) {
            addStats(*stats, total);
            stats->seconds += since(t0);
        }
        return ok;
    }

}  // namespace

extern "C" {

MBGPU_API int mbgpu_open(const char* nameMatch, const uint64_t vramCapBytes, char* info, const int infoLen) {
    t_error.clear();
    if (g_m.open) {
        mbgpu_close();
    }
    if (!loadVulkan()) {
        return finish(false);
    }
    std::vector<Candidate> all = listDevices();
    std::vector<const Candidate*> chosen;
    if (!chooseDevices(nameMatch != nullptr ? nameMatch : "auto", all, chosen)) {
        return finish(false);
    }
    g_m = Module{};
    std::string text = "Vulkan, " + std::to_string(chosen.size()) + " device" + (chosen.size() > 1 ? "s" : "") + ":";
    for (const Candidate* c : chosen) {
        auto g = std::make_unique<Gpu>();
        if (!g->create(*c, vramCapBytes)) {
            const std::string why = t_error;
            g_m.gpus.clear();
            return finish(fail(c->name + ": " + why));
        }
        char line[512];
        std::snprintf(line, sizeof(line),
                      " #%d %s (%s%s, %s, subgroup %u), %.1f GB, %.1f GB free, cap %.1f GB, level buffers %llu nodes;",
                      c->index, c->name.c_str(), typeName(c->type), c->display ? ", drives a display" : "",
                      c->driver.c_str(), c->subgroup, c->heapBytes / 1e9, c->freeBytes / 1e9, g->cap / 1e9,
                      static_cast<unsigned long long>(g->levelCap));
        text += line;
        g_m.gpus.push_back(std::move(g));
    }
    g_m.open = true;
    if (!text.empty() && text.back() == ';') {
        text.pop_back();
    }
    if (info != nullptr && infoLen > 0) {
        std::snprintf(info, static_cast<size_t>(infoLen), "%s", text.c_str());
    }
    return finish(true);
}

MBGPU_API int mbgpu_set_target(const uint64_t goalB1, const uint64_t goalB2, const int limit) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if ((goalB1 & FAT_BOOL_BIT) && limit >= 0) {
        return finish(fail("2x2 boards: no bound on the GPU (pass limit -1)"));
    }
    // mbgpu.cu: rowLane / colLane and the node kernel's mismatch tables
    for (u32 i = 0; i < 6; ++i) {
        g_m.rowTarget[i] = ((i < 3 ? goalB1 : goalB2) >> (36 - 18 * (i % 3))) & 0x3FFFFULL;
        u64 lane = 0;
        for (u32 r = 0; r < 6; ++r) {
            const u64 half = r < 3 ? goalB1 : goalB2;
            lane |= ((half >> (51 - 18 * (r % 3) - 3 * i)) & 7ULL) << (15 - 3 * r);
        }
        g_m.colTarget[i] = lane;
    }
    for (u32 a = 0; a < 6; ++a) {
        for (u32 p = 0; p < 6; ++p) {
            for (u32 v = 0; v < 8; ++v) {
                u32 ec = 0, er = 0;
                for (u32 k = 0; k < 6; ++k) {
                    const u32 q = (p + k) % 6;
                    ec |= (v != ((g_m.colTarget[a] >> (15 - 3 * q)) & 7u) ? 1u : 0u) << (4 * k);
                    er |= (v != ((g_m.rowTarget[a] >> (15 - 3 * q)) & 7u) ? 1u : 0u) << (4 * k);
                }
                g_m.mc[(a * 6 + p) * 8 + v] = ec;
                g_m.mr[(a * 6 + p) * 8 + v] = er;
            }
        }
    }
    g_m.limit = limit;
    g_m.targetSet = true;
    return finish(true);
}

MBGPU_API int mbgpu_set_filter(const uint32_t* bloomWords, const uint64_t blocksPerPartition, const uint32_t pBegin,
                               const uint32_t pEnd) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (pBegin >= pEnd || pEnd > 256) {
        return finish(fail("bad partition range"));
    }
    const u32 parts = pEnd - pBegin;
    const u64 partBytes = blocksPerPartition * 32;
    std::atomic<bool> failed{false};
    std::string firstError;
    std::mutex m;
    std::vector<std::thread> threads;
    for (auto& gp : g_m.gpus) {
        Gpu* g = gp.get();
        threads.emplace_back([&, g] {
            for (Buffer& b : g->bloom) {
                g->release(b);
            }
            g->bloomBufs = 0;
            g->partsPerBuf = 256;
            g->pBegin = pBegin;
            g->pEnd = pEnd;
            g->blocksPerPartition = blocksPerPartition;
            if (bloomWords == nullptr) {
                return;
            }
            // partitions per buffer: as many as one buffer may hold (MBVK_BLOOM_BUF_MB lowers it, to
            // test the split on small filters)
            u64 bufCap = g->cand.maxAlloc != 0 ? g->cand.maxAlloc : ~0ULL;
            if (const char* env = std::getenv("MBVK_BLOOM_BUF_MB")) {
                bufCap = std::min<u64>(bufCap, std::strtoull(env, nullptr, 10) << 20);
            }
            const u64 per = partBytes != 0 ? std::max<u64>(1, bufCap / partBytes) : parts;
            g->partsPerBuf = static_cast<u32>(std::min<u64>(per, parts));
            const u32 bufs = (parts + g->partsPerBuf - 1) / g->partsPerBuf;
            bool ok = partBytes <= bufCap && bufs <= Gpu::MAX_BLOOM_BUFS;
            if (!ok) {
                fail("the Bloom blocks of " + std::to_string(parts) + " partitions need more than " +
                     std::to_string(Gpu::MAX_BLOOM_BUFS) + " buffers on this device");
            }
            for (u32 b = 0; ok && b < bufs; ++b) {
                const u32 n = std::min(g->partsPerBuf, parts - b * g->partsPerBuf);
                const u64 off = static_cast<u64>(b) * g->partsPerBuf * partBytes;
                ok = g->alloc(g->bloom[b], n * partBytes) &&
                     g->upload(g->bloom[b], 0, reinterpret_cast<const u8*>(bloomWords) + off, n * partBytes);
                g->bloomBufs = b + 1;
            }
            if (!ok) {
                std::lock_guard<std::mutex> lock(m);
                if (!failed.exchange(true)) {
                    firstError = g->cand.name + ": " + t_error;
                }
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    return finish(failed ? fail(firstError) : true);
}

MBGPU_API int mbgpu_stream(const uint64_t* states, const uint8_t* prevMove, const uint64_t count, const int depth,
                           const mbgpu_sink_fn sink, void* ctx, mbgpu_stats* stats) {
    t_error.clear();
    return finish(needOpen() && filterStream(states, prevMove, count, depth, 1, sink, ctx, stats, false));
}

MBGPU_API int mbgpu_stream_checksums(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                     const int depth, uint64_t* leafCount, uint64_t* keySum, uint64_t* keyXor,
                                     mbgpu_stats* stats) {
    t_error.clear();
    return finish(needOpen() &&
                  checksumStream(states, prevMove, count, depth, 0, leafCount, keySum, keyXor, stats, nullptr));
}

MBGPU_API int mbgpu_audit_stream(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                 const int depth, uint64_t* wantedCount, uint64_t* wantedSum, uint64_t* wantedXor,
                                 mbgpu_audit_stats* stats) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (g_m.limit < 0) {
        return finish(fail("the audit needs a limit (mbgpu_set_target)"));
    }
    mbgpu_stats st{};
    Run totals;
    if (!checksumStream(states, prevMove, count, depth, 1, wantedCount, wantedSum, wantedXor, &st, &totals)) {
        return finish(false);
    }
    if (stats != nullptr) {
        stats->nodes = 0;
        for (int d = 1; d < depth; ++d) {
            stats->nodes += st.nodes[d];
        }
        stats->nodes += totals.leaves;
        stats->edges = totals.edges;
        stats->violations = totals.violations;
        stats->leaves = totals.leaves;
        stats->overLimit = totals.overLimit;
        stats->seconds = st.seconds;
    }
    return finish(true);
}

MBGPU_API int mbgpu_bounds(const uint64_t* states, const uint64_t count, int32_t* h) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (!g_m.targetSet) {
        return finish(fail("mbgpu_set_target has not been called"));
    }
    return finish(count == 0 || g_m.gpus[0]->bounds(states, count, h));
}

MBGPU_API int mbgpu_child_counts(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                 const int depth, uint64_t partitionCounts[256]) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    std::vector<u64> sum(256, 0);
    std::mutex m;
    const bool ok = splitSeeds(count, [&](Gpu& g, const u64 a, const u64 n) {
        Buffer counts;
        std::vector<u64> part(256);
        bool r = g.alloc(counts, 256 * 8) && g.zero(counts, 256 * 8);
        if (r) {
            mbgpu_stats st{};
            Run run;
            run.depth = depth;
            run.prune = 0;
            run.stats = &st;
            run.mode = LEAF_COUNTS;
            g.gl.partCounts = counts.addr;
            g.gl.pBegin = 0;
            g.gl.pEnd = 256;
            r = g.streamAll(run, states + 2 * a, prevMove != nullptr ? prevMove + a : nullptr, n) &&
                g.download(part.data(), counts, 0, 256 * 8);
            g.gl.partCounts = 0;
        }
        g.release(counts);
        if (r) {
            std::lock_guard<std::mutex> lock(m);
            for (int p = 0; p < 256; ++p) {
                sum[p] += part[p];
            }
        }
        return r;
    });
    if (ok) {
        std::memcpy(partitionCounts, sum.data(), 256 * 8);
    }
    return finish(ok);
}

MBGPU_API int mbgpu_child_keys(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                               const int depth, const uint32_t p0, const uint32_t p1, uint64_t* out,
                               const uint64_t outCap, uint64_t* outCount) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (p0 >= p1 || p1 > 256) {
        return finish(fail("bad partition range"));
    }
    return finish(g_m.gpus[0]->childKeys(states, prevMove, count, depth, p0, p1, out, outCap, outCount));
}

MBGPU_API int mbgpu_match_children(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                   const int depth, const uint64_t* sortedKeys, const uint64_t keyCount,
                                   const mbgpu_sink_fn sink, void* ctx, mbgpu_stats* stats) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (keyCount == 0 || count == 0) {
        return finish(true);
    }
    // the keys on every device
    std::vector<Buffer> keys(g_m.gpus.size());
    bool ok = true;
    for (std::size_t i = 0; ok && i < g_m.gpus.size(); ++i) {
        Gpu& g = *g_m.gpus[i];
        ok = g.alloc(keys[i], keyCount * 8) && g.upload(keys[i], 0, sortedKeys, keyCount * 8);
        g.gl.match = keys[i].addr;
        g.gl.matchCount = keyCount;
        g.gl.bloom = 0;
        g.gl.pBegin = 0;
        g.gl.pEnd = 256;
    }
    ok = ok && filterStream(states, prevMove, count, depth, 0, sink, ctx, stats, true);
    for (std::size_t i = 0; i < g_m.gpus.size(); ++i) {
        Gpu& g = *g_m.gpus[i];
        g.release(keys[i]);
        g.gl.match = 0;
        g.gl.matchCount = 0;
    }
    return finish(ok);
}

MBGPU_API uint64_t mbgpu_vram_left(void) {
    if (!g_m.open) {
        return 0;
    }
    // the Bloom blocks go to every device, split over up to 8 buffers (each within the driver's
    // buffer cap); the caller keeps 256 MB clear
    u64 left = ~0ULL;
    for (const auto& g : g_m.gpus) {
        u64 l = g->vramLeft();
        if (g->cand.maxAlloc != 0 && g->cand.maxAlloc < (~0ULL - RESERVE) / Gpu::MAX_BLOOM_BUFS) {
            l = std::min<u64>(l, g->cand.maxAlloc * Gpu::MAX_BLOOM_BUFS + RESERVE);
        }
        left = std::min(left, l);
    }
    return left;
}

MBGPU_API uint64_t mbgpu_child_key_capacity(void) {
    return g_m.open ? g_m.gpus[0]->keyCapacity() : 0;
}

MBGPU_API int mbgpu_build_index(const uint64_t* states, const uint8_t* prevMove, const uint64_t count,
                                const int depth, const uint32_t p0, const uint32_t p1, const uint32_t bucketBits,
                                const uint64_t bloomBlocksPerPartition, uint16_t* fpsOut, const uint64_t fpsCap,
                                uint32_t* startsOut, uint32_t* bloomOut, uint64_t* uniqueOut) {
    t_error.clear();
    if (!needOpen()) {
        return finish(false);
    }
    if (p0 >= p1 || p1 > 256 || bucketBits > 30 || bloomBlocksPerPartition == 0) {
        return finish(fail("bad index layout"));
    }
    return finish(g_m.gpus[0]->buildIndex(states, prevMove, count, depth, p0, p1, bucketBits, bloomBlocksPerPartition,
                                          fpsOut, fpsCap, startsOut, bloomOut, uniqueOut));
}

MBGPU_API const char* mbgpu_last_error(void) {
    return g_error.c_str();
}

MBGPU_API int mbvk_list_devices(char* out, const int outLen) {
    t_error.clear();
    if (!loadVulkan()) {
        return finish(false);
    }
    std::vector<Candidate> all = listDevices();
    std::vector<const Candidate*> autoPick;
    const std::string keep = t_error;
    if (!chooseDevices("auto", all, autoPick)) {
        autoPick.clear();
    }
    t_error = keep;
    std::string text;
    for (const Candidate& c : all) {
        bool picked = false;
        for (const Candidate* a : autoPick) {
            picked = picked || a->index == c.index;
        }
        char line[768];
        std::snprintf(line, sizeof(line), "#%d %s (%s%s, %s, subgroup %u, %.1f GB, %.1f GB free): %s%s\n", c.index,
                      c.name.c_str(), typeName(c.type), c.display ? ", drives a display" : "", c.driver.c_str(),
                      c.subgroup, c.heapBytes / 1e9, c.freeBytes / 1e9,
                      c.whyNot.empty() ? "usable" : ("cannot run: " + c.whyNot).c_str(), picked ? " [auto]" : "");
        text += line;
    }
    if (all.empty()) {
        text = "no Vulkan devices\n";
    }
    if (out != nullptr && outLen > 0) {
        std::snprintf(out, static_cast<size_t>(outLen), "%s", text.c_str());
    }
    return finish(true);
}

MBGPU_API void mbgpu_close(void) {
    g_m.gpus.clear();
    g_m = Module{};
}

}  // extern "C"
