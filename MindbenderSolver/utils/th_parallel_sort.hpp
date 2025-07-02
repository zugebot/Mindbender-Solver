#pragma once

#include <algorithm>
#include <thread>
#include <vector>

#include "MindbenderSolver/utils/hasGetHash.hpp"
#include "MindbenderSolver/utils/jvec.hpp"



#ifdef USE_CUDA
template <int NUM_THREADS, typename T>
#else
template <int NUM_THREADS, HasGetHash_v T>
#endif
MU void parallel_sort(JVec<T>& data) {
 static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");


    size_t size = data.size();
    const size_t chunk_size = size / NUM_THREADS;

    std::vector<std::thread> threads;
    std::vector<JVec<T>> sorted_chunks(NUM_THREADS);

    // Step 1: Divide the data and sort each chunk in a separate thread
    for (int i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == NUM_THREADS - 1) ? size : (i + 1) * chunk_size;

        // Assign a portion of the array to each thread for sorting
        sorted_chunks[i] = JVec<T>(data.begin() + start, data.begin() + end);

        // Start thread to sort the chunk
        threads.emplace_back([&sorted_chunks, i]() {
            std::sort(sorted_chunks[i].begin(), sorted_chunks[i].end());
        });
    }

    // Step 2: Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Step 3: Merge all sorted chunks
    JVec<T> result;
    result.reserve(size);

    // Lambda to merge two sorted vectors
    auto merge_two_sorted = [](const std::vector<T>& left, const std::vector<T>& right) -> std::vector<T> {
        JVec<T> merged;
        merged.reserve(left.size() + right.size());

        auto it_left = left.begin();
        auto it_right = right.begin();

        while (it_left != left.end() && it_right != right.end()) {
            if (*it_left < *it_right) {
                merged.push_back(*it_left++);
            } else {
                merged.push_back(*it_right++);
            }
        }

        // Copy the remaining elements
        merged.insert(merged.end(), it_left, left.end());
        merged.insert(merged.end(), it_right, right.end());

        return merged;
    };

    // Merging in a pairwise fashion (multi-way merge)
    while (sorted_chunks.size() > 1) {
        std::vector<JVec<T>> new_sorted_chunks;

        for (size_t i = 0; i < sorted_chunks.size(); i += 2) {
            if (i + 1 < sorted_chunks.size()) {
                // Merge two adjacent chunks
                new_sorted_chunks.push_back(merge_two_sorted(sorted_chunks[i], sorted_chunks[i + 1]));
            } else {
                // If there is an odd chunk, just carry it forward
                new_sorted_chunks.push_back(sorted_chunks[i]);
            }
        }

        // Update the sorted_chunks vector with the newly merged chunks
        sorted_chunks = std::move(new_sorted_chunks);
    }

    // The final merged chunk is the fully sorted array
    data = std::move(sorted_chunks[0]);
}
