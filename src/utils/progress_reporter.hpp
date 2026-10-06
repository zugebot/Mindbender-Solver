#pragma once
// utils/progress_reporter.hpp
//
// Machine-readable progress for long runs, read by python/scheduler. Worker
// threads only bump an atomic counter once per coarse unit of work (a seed, a
// partition); a background thread rewrites a small JSON file every few
// seconds. Nothing is printed and no lock is taken on the hot path, so the
// cost is one relaxed atomic add per unit (milliseconds of work each).
//
//   {"phase":"probe","detail":"pass 3/11","done":123,"total":456,
//    "phase_seconds":81.2,"elapsed":4021.5,"pid":1234}

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <process.h>
#define PROGRESS_GETPID _getpid
#else
#include <unistd.h>
#define PROGRESS_GETPID getpid
#endif

class ProgressReporter {
    using Clock = std::chrono::steady_clock;

    std::atomic<std::uint64_t> done_{0};
    std::atomic<std::uint64_t> total_{0};
    std::mutex mutex_;                 // guards phase_, detail_, phaseStart_, file I/O
    std::condition_variable wake_;
    std::thread thread_;
    std::string path_;
    std::string phase_ = "starting";
    std::string detail_;
    Clock::time_point start_ = Clock::now();
    Clock::time_point phaseStart_ = Clock::now();
    double intervalSeconds_ = 5.0;
    bool stop_ = false;
    bool enabled_ = false;

    static std::string escape(const std::string& s) {
        std::string out;
        for (const char c : s) {
            if (c == '"' || c == '\\') {
                out += '\\';
            }
            out += c;
        }
        return out;
    }

    // Caller holds mutex_. Writes a temp file, then renames it over the
    // target, so a reader never sees a half-written file.
    void writeLocked() {
        const auto now = Clock::now();
        const double phaseSeconds = std::chrono::duration<double>(now - phaseStart_).count();
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        const std::string tmp = path_ + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (f == nullptr) {
            return;
        }
        std::fprintf(f,
                     "{\"phase\":\"%s\",\"detail\":\"%s\",\"done\":%llu,\"total\":%llu,"
                     "\"phase_seconds\":%.3f,\"elapsed\":%.3f,\"pid\":%d}\n",
                     escape(phase_).c_str(), escape(detail_).c_str(),
                     static_cast<unsigned long long>(done_.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(total_.load(std::memory_order_relaxed)),
                     phaseSeconds, elapsed, static_cast<int>(PROGRESS_GETPID()));
        std::fclose(f);
        std::error_code ec;
        std::filesystem::rename(tmp, path_, ec);
    }

public:
    ProgressReporter() = default;
    ProgressReporter(const ProgressReporter&) = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;

    ~ProgressReporter() { stop(); }

    // No file path: every call below is a cheap no-op apart from the atomics.
    void start(const std::string& path, const double intervalSeconds) {
        if (path.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        path_ = path;
        intervalSeconds_ = intervalSeconds > 0.1 ? intervalSeconds : 0.1;
        enabled_ = true;
        start_ = phaseStart_ = Clock::now();
        writeLocked();
        thread_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mutex_);
            while (!stop_) {
                wake_.wait_for(lk, std::chrono::duration<double>(intervalSeconds_));
                if (!stop_) {
                    writeLocked();
                }
            }
        });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!enabled_ || stop_) {
                return;
            }
            stop_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // A new phase: resets the counter. `total` units are expected.
    void phase(const std::string& name, const std::uint64_t total, const std::string& detail = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        phase_ = name;
        detail_ = detail;
        done_.store(0, std::memory_order_relaxed);
        total_.store(total, std::memory_order_relaxed);
        phaseStart_ = Clock::now();
        if (enabled_) {
            writeLocked();
        }
    }

    // Same phase, new sub-step label (e.g. "pass 3/11"); the counter keeps going.
    void detail(const std::string& text) {
        std::lock_guard<std::mutex> lock(mutex_);
        detail_ = text;
        if (enabled_) {
            writeLocked();
        }
    }

    // Revise the expected total of the current phase (the counter keeps going).
    void total(const std::uint64_t n) {
        total_.store(n, std::memory_order_relaxed);
    }

    void add(const std::uint64_t n = 1) {
        done_.fetch_add(n, std::memory_order_relaxed);
    }
};

inline ProgressReporter g_progress;
