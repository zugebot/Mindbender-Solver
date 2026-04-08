#pragma once

#include <cstdint>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <array>
#include <algorithm>

#include "utils/hasGetHash.hpp"
#include "utils/jvec.hpp"
#include "utils/timer.hpp"


#define RADIX_IF_DEBUG_COUT(str) \
    if constexpr (DEBUG) { \
        std::cout << total_time.getSeconds() - last_time << (str); \
        last_time = total_time.getSeconds(); \
    }



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

#ifdef USE_CUDA
template<int NUM_PASSES, int NUM_BITS_PER_PASS, typename T, bool DEBUG=false>
#else
template<int NUM_PASSES, int NUM_BITS_PER_PASS, HasGetHash_v T, bool DEBUG=false>
#endif
void radix_sort(JVec<T>& data_out, JVec<T>& aux_buffer) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    using count_t = uint32_t;
    using vec1_count_t = std::vector<count_t>;
    using vec2_count_t = std::vector<vec1_count_t>;

    static constexpr int num_buckets = 1 << NUM_BITS_PER_PASS;

    const Timer total_time;
    double last_time = 0;

    RADIX_IF_DEBUG_COUT(" Allocating Aux Phase\n")

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


    RADIX_IF_DEBUG_COUT(" Precomputing Masks and Shifts\n")

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
                std::fill(local_count.begin(), local_count.end(), 0);
                for (count_t i = start; i < end; ++i) {
                    const u64 bucket = (data_out[i].getHash() >> shift) & mask;
                    ++local_count[bucket];
                }
            });
        }

        for (auto& thread : thread_pool) thread.join();

        RADIX_IF_DEBUG_COUT(" Counting Phase\n")

        // Accumulate Counts
        std::fill(count.begin(), count.end(), 0);
        for (count_t b = 0; b < num_buckets; ++b) {
            for (count_t t = 0; t < num_threads; ++t) {
                count[b] += thread_counts[t][b];
            }
        }

        RADIX_IF_DEBUG_COUT(" Accumulate Counts Phase\n")

        // Compute Offsets
        bucket_offsets[0] = 0;
        for (count_t b = 1; b < num_buckets; ++b) {
            bucket_offsets[b] = bucket_offsets[b - 1] + count[b - 1];
        }

        RADIX_IF_DEBUG_COUT(" Compute Offsets Phase\n")

        // Compute Thread Offsets
        for (count_t b = 0; b < num_buckets; ++b) {
            count_t offset = bucket_offsets[b];
            for (count_t t = 0; t < num_threads; ++t) {
                thread_offsets[t][b] = offset;
                offset += thread_counts[t][b];
            }
        }

        RADIX_IF_DEBUG_COUT(" Compute Thread Offsets Phase\n")

        // Distribution Phase
        for (count_t t = 0; t < num_threads; ++t) {
            count_t start = t * chunk_size;
            count_t end = std::min(start + chunk_size, static_cast<count_t>(data_out.size()));
            thread_pool[t] = std::thread([&, t, start, end]() {
                auto& local_offset = thread_offsets[t];
                for (count_t i = start; i < end; ++i) {
                    const u64 bucket = (data_out[i].getHash() >> shift) & mask;
                    aux_buffer[local_offset[bucket]++] = data_out[i];
                }
            });
        }
        for (auto& thread : thread_pool) thread.join();

        RADIX_IF_DEBUG_COUT(" Distribution Phase\n")

        // Swap Data and Aux
        data_out.swap(aux_buffer);
    }
}





template<int NUM_PASSES, int NUM_BITS_PER_PASS, typename T, bool DEBUG = false>
MU void radix_sort(JVec<T>& dataStates,
                   JVec<u64>& dataHashes,
                   JVec<T>& auxStates,
                   JVec<u64>& auxHashes) {
    static_assert(NUM_PASSES >= 1, "NUM_PASSES must be >= 1");
    static_assert(NUM_BITS_PER_PASS >= 1, "NUM_BITS_PER_PASS must be >= 1");
    static_assert(static_cast<u64>(NUM_PASSES - 1) * static_cast<u64>(NUM_BITS_PER_PASS) < 64ULL,
                  "last radix pass would start outside the 64-bit hash");
    static_assert(NUM_BITS_PER_PASS < 64, "NUM_BITS_PER_PASS must be < 64");
    
    if (dataStates.size() != dataHashes.size()) {
        throw std::runtime_error("radix_sort got mismatched state/hash lane sizes");
    }

    if (dataStates.size() <= 1) {
        return;
    }

    using count_t = std::size_t;
    using vec1_count_t = std::vector<count_t>;
    using vec2_count_t = std::vector<vec1_count_t>;

    static constexpr count_t MAX_BUCKETS = static_cast<count_t>(1ULL << NUM_BITS_PER_PASS);
    static constexpr u32 HASH_BITS = 64;

    const Timer total_time;
    double last_time = 0.0;

    auxStates.resize(dataStates.size());
    auxHashes.resize(dataHashes.size());

    RADIX_IF_DEBUG_COUT(" Allocating Aux Phase\n")

    count_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 8;
    }
    if (num_threads > dataStates.size()) {
        num_threads = dataStates.size();
    }
    if (num_threads == 0) {
        num_threads = 1;
    }

    const count_t chunk_size = (dataStates.size() + num_threads - 1) / num_threads;

    std::array<u64, NUM_PASSES> masks{};
    std::array<count_t, NUM_PASSES> shifts{};
    std::array<count_t, NUM_PASSES> activeBuckets{};
    std::array<count_t, NUM_PASSES> activeBits{};

    for (count_t pass = 0; pass < static_cast<count_t>(NUM_PASSES); ++pass) {
        const count_t shift = pass * static_cast<count_t>(NUM_BITS_PER_PASS);
        const count_t remaining_bits = (shift < HASH_BITS) ? (HASH_BITS - shift) : 0;
        const count_t bits_this_pass = std::min<count_t>(NUM_BITS_PER_PASS, remaining_bits);

        shifts[pass] = shift;
        activeBits[pass] = bits_this_pass;
        activeBuckets[pass] = static_cast<count_t>(1ULL << bits_this_pass);

        if (bits_this_pass == 64) {
            masks[pass] = ~0ULL;
        } else {
            masks[pass] = (1ULL << bits_this_pass) - 1ULL;
        }
    }

    RADIX_IF_DEBUG_COUT(" Precomputing Masks and Shifts\n")

    vec2_count_t thread_counts(num_threads, vec1_count_t(MAX_BUCKETS, 0));
    vec2_count_t thread_offsets(num_threads, vec1_count_t(MAX_BUCKETS, 0));

    vec1_count_t count(MAX_BUCKETS, 0);
    vec1_count_t bucket_offsets(MAX_BUCKETS, 0);

    std::vector<std::thread> thread_pool(num_threads);

    if constexpr (DEBUG) {
        std::cout << std::fixed << std::setprecision(7);
        std::cout << total_time.getSeconds() << " Allocation Phase\n";
    }

    for (count_t pass = 0; pass < static_cast<count_t>(NUM_PASSES); ++pass) {
        if constexpr (DEBUG) {
            last_time = total_time.getSeconds();
            std::cout << "Doing Pass #" << pass + 1
                      << " using " << activeBits[pass] << " bits"
                      << " and " << activeBuckets[pass] << " buckets"
                      << std::endl;
        }

        const u64 mask = masks[pass];
        const count_t shift = shifts[pass];
        const count_t num_buckets_this_pass = activeBuckets[pass];

        for (count_t t = 0; t < num_threads; ++t) {
            const count_t start = t * chunk_size;
            const count_t end = std::min(start + chunk_size, static_cast<count_t>(dataHashes.size()));

            thread_pool[t] = std::thread([&, t, start, end, mask, shift, num_buckets_this_pass]() {
                auto& local_count = thread_counts[t];
                std::fill(local_count.begin(), local_count.begin() + num_buckets_this_pass, 0);

                for (count_t i = start; i < end; ++i) {
                    const u64 bucket = (dataHashes[i] >> shift) & mask;
                    ++local_count[static_cast<count_t>(bucket)];
                }
            });
        }

        for (auto& thread : thread_pool) {
            thread.join();
        }

        RADIX_IF_DEBUG_COUT(" Counting Phase\n")

        std::fill(count.begin(), count.begin() + num_buckets_this_pass, 0);
        for (count_t b = 0; b < num_buckets_this_pass; ++b) {
            for (count_t t = 0; t < num_threads; ++t) {
                count[b] += thread_counts[t][b];
            }
        }

        RADIX_IF_DEBUG_COUT(" Accumulate Counts Phase\n")

        bucket_offsets[0] = 0;
        for (count_t b = 1; b < num_buckets_this_pass; ++b) {
            bucket_offsets[b] = bucket_offsets[b - 1] + count[b - 1];
        }

        RADIX_IF_DEBUG_COUT(" Compute Offsets Phase\n")

        for (count_t b = 0; b < num_buckets_this_pass; ++b) {
            count_t offset = bucket_offsets[b];
            for (count_t t = 0; t < num_threads; ++t) {
                thread_offsets[t][b] = offset;
                offset += thread_counts[t][b];
            }
        }

        RADIX_IF_DEBUG_COUT(" Compute Thread Offsets Phase\n")

        for (count_t t = 0; t < num_threads; ++t) {
            const count_t start = t * chunk_size;
            const count_t end = std::min(start + chunk_size, static_cast<count_t>(dataHashes.size()));

            thread_pool[t] = std::thread([&, t, start, end, mask, shift]() {
                auto& local_offset = thread_offsets[t];

                for (count_t i = start; i < end; ++i) {
                    const u64 bucket = (dataHashes[i] >> shift) & mask;
                    const count_t dst = local_offset[static_cast<count_t>(bucket)]++;

                    auxHashes[dst] = dataHashes[i];
                    auxStates[dst] = dataStates[i];
                }
            });
        }

        for (auto& thread : thread_pool) {
            thread.join();
        }

        RADIX_IF_DEBUG_COUT(" Distribution Phase\n")

        dataStates.swap(auxStates);
        dataHashes.swap(auxHashes);
    }
}