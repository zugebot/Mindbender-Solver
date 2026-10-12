#pragma once
// src/utils/parallel_for.hpp
//
// fn(thread, i) for every i in [0, count), on `threads` threads. Indices are
// handed out one at a time from an atomic counter, so uneven work balances
// itself; each call should be at least microseconds of work.

#include "src/utils/processor.hpp"

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

template<typename Fn>
void parallelFor(const u32 threads, const std::size_t count, Fn&& fn) {
    std::atomic<std::size_t> next = 0;
    std::vector<std::thread> pool;
    for (u32 t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (std::size_t i; (i = next.fetch_add(1)) < count;) {
                fn(t, i);
            }
        });
    }
    for (auto& th : pool) {
        th.join();
    }
}
