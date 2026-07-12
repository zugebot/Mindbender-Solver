#include "utils/allocation_counter.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace alloc_counter {
static std::atomic_size_t g_new_calls{0};
static std::atomic_size_t g_delete_calls{0};
static std::atomic_size_t g_malloc_calls{0};
static std::atomic_size_t g_free_calls{0};
static std::atomic_size_t g_calloc_calls{0};
static std::atomic_size_t g_realloc_calls{0};

void printReport(const char* stage) {
    std::fprintf(stderr,
                 "\n=== Allocation counters (process total) ===\n"
                 "%s%s\n"
                 "new/delete tracked via global operators\n"
                 "new calls:        %zu\n"
                 "delete calls:     %zu\n"
                 "malloc calls:     %zu\n"
                 "free calls:       %zu\n"
                 "calloc calls:     %zu\n"
                 "realloc calls:    %zu\n",
                 stage ? "stage: " : "",
                 stage ? stage : "",
                 g_new_calls.load(std::memory_order_relaxed),
                 g_delete_calls.load(std::memory_order_relaxed),
                 g_malloc_calls.load(std::memory_order_relaxed),
                 g_free_calls.load(std::memory_order_relaxed),
                 g_calloc_calls.load(std::memory_order_relaxed),
                 g_realloc_calls.load(std::memory_order_relaxed));
}

struct RegisterAtExit {
    RegisterAtExit() { std::atexit([] { printReport("at exit"); }); }
};

static RegisterAtExit g_register_at_exit;
} // namespace alloc_counter

void* operator new(std::size_t size) {
    alloc_counter::g_new_calls.fetch_add(1, std::memory_order_relaxed);
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
    alloc_counter::g_delete_calls.fetch_add(1, std::memory_order_relaxed);
    std::free(ptr);
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete[](void* ptr) noexcept {
    ::operator delete(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    ::operator delete(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    ::operator delete[](ptr);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size);
    } catch (...) {
        return nullptr;
    }
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    ::operator delete(ptr);
}

#if defined(__GNUC__) || defined(__clang__)
extern "C" void* __real_malloc(std::size_t size);
extern "C" void __real_free(void* ptr);
extern "C" void* __real_calloc(std::size_t count, std::size_t size);
extern "C" void* __real_realloc(void* ptr, std::size_t size);

extern "C" void* __wrap_malloc(std::size_t size) {
    alloc_counter::g_malloc_calls.fetch_add(1, std::memory_order_relaxed);
    return __real_malloc(size);
}

extern "C" void __wrap_free(void* ptr) {
    alloc_counter::g_free_calls.fetch_add(1, std::memory_order_relaxed);
    __real_free(ptr);
}

extern "C" void* __wrap_calloc(std::size_t count, std::size_t size) {
    alloc_counter::g_calloc_calls.fetch_add(1, std::memory_order_relaxed);
    return __real_calloc(count, size);
}

extern "C" void* __wrap_realloc(void* ptr, std::size_t size) {
    alloc_counter::g_realloc_calls.fetch_add(1, std::memory_order_relaxed);
    return __real_realloc(ptr, size);
}
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
static int __cdecl allocHook(
        int allocType,
        void*,
        size_t,
        int,
        long,
        const unsigned char*,
        int
) {
    if (allocType == _HOOK_ALLOC || allocType == _HOOK_REALLOC) {
        alloc_counter::g_malloc_calls.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

struct DebugCrtHookInstall {
    DebugCrtHookInstall() { _CrtSetAllocHook(allocHook); }
};

static DebugCrtHookInstall g_debug_crt_hook_install;
#endif

