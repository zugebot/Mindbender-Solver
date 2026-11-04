#pragma once
// src/utils/platform.hpp
//
// The few operating-system calls the solver makes, for Windows and Linux: memory figures,
// 64-bit file seeks, the process id, the exe's own path, shared libraries (the GPU modules)
// and page-aligned zeroed memory. Windows keeps the calls it always used.

#include <cstdint>
#include <cstdio>
#include <string>

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
#endif
        return m;
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
