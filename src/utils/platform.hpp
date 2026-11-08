#pragma once
// src/utils/platform.hpp
//
// The few operating-system calls the solver makes, for Windows and Linux: memory figures,
// 64-bit file seeks, the process id, the exe's own path, shared libraries (the GPU modules)
// and page-aligned zeroed memory. Windows keeps the calls it always used.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#endif

namespace platform {

    struct MemoryInfo {
        uint64_t total = 0;       // physical RAM
        uint64_t available = 0;   // what new allocations can still get without swapping
    };

    inline MemoryInfo memory() {
        MemoryInfo m;
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        m.total = status.ullTotalPhys;
        m.available = status.ullAvailPhys;
#else
        std::ifstream in("/proc/meminfo");
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            std::string key;
            uint64_t kb = 0;
            fields >> key >> kb;
            if (key == "MemTotal:") m.total = kb * 1024;
            if (key == "MemAvailable:") m.available = kb * 1024;
        }
        // In a container (rented GPU machines) /proc/meminfo shows the whole host, but the
        // cgroup limit is what the process may use before it is killed: cap both figures by it.
        const auto readU64 = [](const char* path, uint64_t& v) {
            std::ifstream f(path);
            std::string s;
            if (!(f >> s) || s == "max") return false;
            try { v = std::stoull(s); } catch (...) { return false; }
            return true;
        };
        const auto statValue = [](const char* path, const std::string& name) {
            std::ifstream f(path);
            std::string key;
            uint64_t v = 0;
            while (f >> key >> v)
                if (key == name) return v;
            return uint64_t{0};
        };
        uint64_t limit = 0, used = 0, reclaimable = 0;
        if (readU64("/sys/fs/cgroup/memory.max", limit)) {   // cgroup v2
            readU64("/sys/fs/cgroup/memory.current", used);
            reclaimable = statValue("/sys/fs/cgroup/memory.stat", "inactive_file");
        } else if (readU64("/sys/fs/cgroup/memory/memory.limit_in_bytes", limit)) {   // v1
            readU64("/sys/fs/cgroup/memory/memory.usage_in_bytes", used);
            reclaimable = statValue("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file");
        }
        if (limit != 0 && limit < m.total) {
            const uint64_t inUse = used > reclaimable ? used - reclaimable : 0;
            m.total = limit;
            m.available = std::min(m.available, limit > inUse ? limit - inUse : 0);
        }
#endif
        return m;
    }

    // CPUs this process may use: the hardware threads, capped by a container's CPU quota
    // (cgroup cpu.max), so a rented 32-thread machine with a 15-CPU share runs 16 threads.
    inline unsigned cpuCount() {
        unsigned n = std::max(1U, std::thread::hardware_concurrency());
#ifndef _WIN32
        const auto cap = [&n](const std::string& quota, const uint64_t period) {
            if (quota == "max" || quota.empty() || quota[0] == '-' || period == 0) return;
            try {
                const uint64_t q = std::stoull(quota);
                n = std::min(n, static_cast<unsigned>(std::max<uint64_t>(1, (q + period - 1) / period)));
            } catch (...) {}
        };
        std::ifstream v2("/sys/fs/cgroup/cpu.max");   // cgroup v2: "QUOTA PERIOD" or "max PERIOD"
        std::string quota;
        uint64_t period = 0;
        if (v2 >> quota >> period) {
            cap(quota, period);
        } else {   // cgroup v1: cfs_quota_us (-1 = none) and cfs_period_us
            std::ifstream q1("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), p1("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
            if (q1 >> quota && p1 >> period) cap(quota, period);
        }
#endif
        return n;
    }

    inline int fseek64(std::FILE* f, const long long offset, const int whence) {
#ifdef _WIN32
        return _fseeki64(f, offset, whence);
#else
        return fseeko(f, static_cast<off_t>(offset), whence);
#endif
    }

    inline unsigned long processId() {
#ifdef _WIN32
        return GetCurrentProcessId();
#else
        return static_cast<unsigned long>(getpid());
#endif
    }

    // The running executable's full path.
    inline std::string exePath() {
#ifdef _WIN32
        char exe[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        return exe;
#else
        char exe[4096] = {};
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        return n > 0 ? std::string(exe, static_cast<std::size_t>(n)) : std::string();
#endif
    }

    // A shared library's file name: name.dll on Windows, libname.so on Linux.
    inline std::string libraryFile(const std::string& name) {
#ifdef _WIN32
        return name + ".dll";
#else
        return "lib" + name + ".so";
#endif
    }

    inline void* loadLibrary(const std::string& path) {
#ifdef _WIN32
        return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
        return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    }

    inline void* librarySymbol(void* lib, const char* name) {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
        return dlsym(lib, name);
#endif
    }

    inline void freeLibrary(void* lib) {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(lib));
#else
        dlclose(lib);
#endif
    }

}  // namespace platform
