#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <array>

#include "timer.hpp"



/**
 * 36:\n
 * [3 12]: ~3.5-3.6sec\n
 * [4 10]: ~3.6-3.7sec\n
 * 60:\n
 * [4 15]: ~7.4-7.5sec\n
 * [5 12]: ~5.8-6.0sec\n
 * [6 10]: ~5.6-5.7sec\n
 *
 *
 * @tparam NUM_PASSES
 * @tparam NUM_BITS_PER_PASS
 * @tparam T
 * @tparam DEBUG
 * @param data_out
 * @param aux_buffer
 */
template<int NUM_PASSES, int NUM_BITS_PER_PASS, class T, bool DEBUG=false>
void radix_sort(std::vector<T>&data_out, std::vector<T>&aux_buffer) {
    using count_t = uint32_t;
    using vec1_count_t = std::vector<count_t>;
    using vec2_count_t = std::vector<vec1_count_t>;


    static constexpr int num_buckets = 1 << NUM_BITS_PER_PASS;

    const Timer total_time;
    float last_time = 0;


    if constexpr (DEBUG) {
        std::cout << total_time.getSeconds() - last_time << " Allocating Aux Phase\n";
        last_time = total_time.getSeconds();
    }


    count_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;


    const count_t chunk_size = (data_out.size() + num_threads - 1) / num_threads;

    // Precompute masks and shifts
    std::array<uint64_t, NUM_PASSES> masks{};
    std::array<count_t, NUM_PASSES> shifts{};
    for (count_t pass = 0; pass < NUM_PASSES; ++pass) {
        shifts[pass] = pass * NUM_BITS_PER_PASS;
        masks[pass] = (1ULL << NUM_BITS_PER_PASS) - 1;
    }

    if constexpr (DEBUG) {
        std::cout << total_time.getSeconds() - last_time << " Precomputing Masks and Shifts\n";
        last_time = total_time.getSeconds();
    }

    vec2_count_t thread_counts(num_threads, vec1_count_t(num_buckets, 0));
    vec2_count_t thread_offsets(num_threads, vec1_count_t(num_buckets));

    // These are global
    vec1_count_t count(num_buckets, 0);
    vec1_count_t bucket_offsets(num_buckets, 0);

    std::vector<std::thread> thread_pool(num_threads);

    if constexpr (DEBUG) {
        std::cout << std::fixed << std::setprecision(7);
        std::cout << total_time.getSeconds() << " Allocation Phase\n";
    }

    for (count_t pass = 0; pass < NUM_PASSES; ++pass) {
        if constexpr (DEBUG) {
            last_time = total_time.getSeconds();
            std::cout << "Doing Pass #" << pass + 1 << std::endl;
        }
        const uint64_t mask = masks[pass];
        const count_t shift = shifts[pass];

        // Counting Phase
        for (count_t t = 0; t < num_threads; ++t) {
            count_t start = t * chunk_size;
            count_t end = std::min(start + chunk_size, static_cast<count_t>(data_out.size()));
            thread_pool[t] = std::thread([&, t, start, end]() {
                auto& local_count = thread_counts[t];
                std::ranges::fill(local_count, 0);
                for (count_t i = start; i < end; ++i) {
                    c_u64 bucket = (data_out[i].hash >> shift) & mask;
                    ++local_count[bucket];
                }
            });
        }

        for (auto& thread : thread_pool) thread.join();
        if constexpr (DEBUG) {
            std::cout << total_time.getSeconds() - last_time << " Counting Phase\n";
            last_time = total_time.getSeconds();
        }

        // Accumulate Counts
        std::ranges::fill(count, 0);
        for (count_t b = 0; b < num_buckets; ++b) {
            for (count_t t = 0; t < num_threads; ++t) {
                count[b] += thread_counts[t][b];
            }
        }

        if constexpr (DEBUG) {
            std::cout << total_time.getSeconds() - last_time << " Accumulate Counts Phase\n";
            last_time = total_time.getSeconds();
        }

        // Compute Offsets
        bucket_offsets[0] = 0;
        for (count_t b = 1; b < num_buckets; ++b) {
            bucket_offsets[b] = bucket_offsets[b - 1] + count[b - 1];
        }

        if constexpr (DEBUG) {
            std::cout << total_time.getSeconds() - last_time << " Compute Offsets Phase\n";
            last_time = total_time.getSeconds();
        }

        // Compute Thread Offsets
        for (count_t b = 0; b < num_buckets; ++b) {
            count_t offset = bucket_offsets[b];
            for (count_t t = 0; t < num_threads; ++t) {
                thread_offsets[t][b] = offset;
                offset += thread_counts[t][b];
            }
        }

        if constexpr (DEBUG) {
            std::cout << total_time.getSeconds() - last_time << " Compute Thread Offsets Phase\n";
            last_time = total_time.getSeconds();
        }

        // Distribution Phase
        for (count_t t = 0; t < num_threads; ++t) {
            count_t start = t * chunk_size;
            count_t end = std::min(start + chunk_size, static_cast<count_t>(data_out.size()));
            thread_pool[t] = std::thread([&, t, start, end]() {
                auto& local_offset = thread_offsets[t];
                for (count_t i = start; i < end; ++i) {
                    c_u64 bucket = (data_out[i].hash >> shift) & mask;
                    aux_buffer[local_offset[bucket]++] = data_out[i];
                }
            });
        }
        for (auto& thread : thread_pool) thread.join();

        if constexpr (DEBUG) {
            std::cout << total_time.getSeconds() - last_time << " Distribution Phase\n";
            last_time = total_time.getSeconds();
        }

        // Swap Data and Aux
        data_out.swap(aux_buffer);
    }
}
