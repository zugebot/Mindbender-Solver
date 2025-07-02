#pragma once

#include "board.hpp"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "utils/jvec.hpp"
#include "utils/hasGetHash.hpp"

#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
void process_chunk(
        C JVec<T> &boards1,
        C JVec<T> &boards2,
        C size_t start1,
        C size_t end1,
        std::vector<std::pair<C T *, C T *>> &results) {
    auto it1 = boards1.begin() + static_cast<i64>(start1);
    C auto it1_end = boards1.begin() + static_cast<i64>(end1);

    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");


    // Get min and max hash values in this chunk
    C u64 min_hash = it1->getHash();
    C u64 max_hash = (it1_end - 1)->getHash();

    // Find corresponding range in boards2
    C auto boards2_start = std::lower_bound(
            boards2.begin(), boards2.end(), min_hash,
            [](C T &board, C u64 hash) { return board.getHash() < hash; });

    C auto boards2_end = std::upper_bound(
            boards2.begin(), boards2.end(), max_hash,
            [](C u64 hash, C T &board) { return hash < board.getHash(); });

    auto it2 = boards2_start;
    C auto it2_end = boards2_end;

    while (it1 != it1_end && it2 != it2_end) {
        if (it1->getHash() == it2->getHash()) {
            // Find ranges of matching hashes in boards1
            auto it1_range_end = it1;
            while (it1_range_end != it1_end && it1_range_end->getHash() == it1->getHash()) {
                ++it1_range_end;
            }
            // Find ranges of matching hashes in boards2
            auto it2_range_end = it2;
            while (it2_range_end != it2_end && it2_range_end->getHash() == it2->getHash()) {
                ++it2_range_end;
            }
            // Make pairs for all combinations of matching hashes
            for (auto it1_match = it1; it1_match != it1_range_end; ++it1_match) {
                for (auto it2_match = it2; it2_match != it2_range_end; ++it2_match) {
                    results.emplace_back(it1_match, it2_match);
                }
            }
            it1 = it1_range_end;
            it2 = it2_range_end;
        } else if (it1->getHash() < it2->getHash()) {
            ++it1;
        } else {
            ++it2;
        }
    }
}


#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
std::vector<std::pair<C T*, C T*>> intersection_threaded(
        C JVec<T>& boards1,
        C JVec<T>& boards2,
        size_t num_threads = std::thread::hardware_concurrency()) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    if (num_threads == 0) num_threads = 1;
    C size_t total_size = boards1.size();
    C size_t chunk_size = (total_size + num_threads - 1) / num_threads; // Round up
    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<C T*, C T*>>> partial_results(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start1 = i * chunk_size;
        size_t end1 = std::min(start1 + chunk_size, total_size);
        // Adjust start1
        if (start1 != 0) {
            C u64 current_hash = boards1[start1].getHash();
            while (start1 > 0 && boards1[start1 - 1].getHash() == current_hash) {
                --start1;
            }
        }
        // Adjust end1
        if (end1 < total_size) {
            C u64 current_hash = boards1[end1 - 1].getHash();
            while (end1 < total_size && boards1[end1].getHash() == current_hash) {
                ++end1;
            }
        }
        // Capture by value to avoid data races
        threads.emplace_back([&, start1, end1, i]() {
            process_chunk<T>(boards1, boards2, start1, end1, partial_results[i]);
        });
    }
    // Wait for all threads to complete
    for (auto& t : threads) { t.join(); }
    // Combine partial results
    std::vector<std::pair<C T*, C T*>> results;
    for (C auto& partial : partial_results) {
        results.insert(results.end(), partial.begin(), partial.end());
    }
    // removes board pairs that have the same hashes but different states
    std::vector<std::pair<C T *, C T *>> realResults;
    realResults.reserve(results.size());
    for (auto& [fst, snd] : results) {
        if (*fst == *snd) {
            realResults.emplace_back(fst, snd);
        }
    }

    return realResults;
}


#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
std::vector<std::pair<C T *, C T *>> intersection(JVec<T>& boards1,
                                              JVec<T>& boards2) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    std::vector<std::pair<C T *, C T *>> results;
    auto it1 = boards1.begin();
    auto it2 = boards2.begin();
    while (it1 != boards1.end() && it2 != boards2.end()) {
        if (it1->getHash() == it2->getHash()) {
            auto it1_end = it1;
            auto it2_end = it2;
            // find range of matching hashes in boards1
            while (it1_end != boards1.end() && it1_end->getHash() == it1->getHash()) {
                ++it1_end;
            }
            // find range of matching hashes in boards2
            while (it2_end != boards2.end() && it2_end->getHash() == it2->getHash()) {
                ++it2_end;
            }
            // make pairs for all combinations of matching hashes
            for (auto it1_match = it1; it1_match != it1_end; ++it1_match) {
                for (auto it2_match = it2; it2_match != it2_end; ++it2_match) {
                    results.emplace_back(it1_match, it2_match);
                    break;
                }
                break;
            }

            it1 = it1_end;
            it2 = it2_end;
        } else if (it1->getHash() < it2->getHash()) {
            ++it1;
        } else {
            ++it2;
        }
    }

    // removes board pairs that have the same hashes but different states
    std::vector<std::pair<C T *, C T *>> realResults;
    realResults.reserve(results.size());
    for (auto& [fst, snd] : results) {
        if (*fst == *snd) {
            realResults.emplace_back(fst, snd);
        }
    }
    return realResults;

}
