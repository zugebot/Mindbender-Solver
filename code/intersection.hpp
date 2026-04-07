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
        C JVec<T>& boards1,
        C JVec<T>& boards2,
        C size_t start1,
        C size_t end1,
        std::vector<std::pair<C T*, C T*>>& results) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
    
    C size_t size1 = boards1.size();
    C size_t size2 = boards2.size();

    if (size1 == 0 || size2 == 0) {
        return;
    }
    if (start1 >= end1) {
        return;
    }
    if (start1 >= size1) {
        return;
    }
    if (end1 > size1) {
        return;
    }

    C u64 min_hash = boards1[start1].getHash();
    C u64 max_hash = boards1[end1 - 1].getHash();

    auto it2_begin = std::lower_bound(
            boards2.begin(),
            boards2.end(),
            min_hash,
            [](C T& board, C u64 hash) {
                return board.getHash() < hash;
            });

    auto it2_end_it = std::upper_bound(
            boards2.begin(),
            boards2.end(),
            max_hash,
            [](C u64 hash, C T& board) {
                return hash < board.getHash();
            });

    C size_t start2 = static_cast<size_t>(it2_begin - boards2.begin());
    C size_t end2 = static_cast<size_t>(it2_end_it - boards2.begin());

    if (start2 >= end2) {
        return;
    }
    if (start2 >= size2 || end2 > size2) {
        return;
    }

    size_t i1 = start1;
    size_t i2 = start2;

    while (i1 < end1 && i2 < end2) {
        C u64 hash1 = boards1[i1].getHash();
        C u64 hash2 = boards2[i2].getHash();

        if (hash1 == hash2) {
            size_t i1_end = i1;
            while (i1_end < end1 && boards1[i1_end].getHash() == hash1) {
                ++i1_end;
            }

            size_t i2_end = i2;
            while (i2_end < end2 && boards2[i2_end].getHash() == hash2) {
                ++i2_end;
            }

            for (size_t a = i1; a < i1_end; ++a) {
                for (size_t b = i2; b < i2_end; ++b) {
                    results.emplace_back(&boards1[a], &boards2[b]);
                }
            }

            i1 = i1_end;
            i2 = i2_end;
        } else if (hash1 < hash2) {
            ++i1;
        } else {
            ++i2;
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
    
    C size_t total_size = boards1.size();
    if (total_size == 0 || boards2.empty()) {
        return {};
    }

    if (num_threads == 0) {
        num_threads = 1;
    }
    num_threads = std::min(num_threads, total_size);

    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(num_threads);

    {
        C size_t base_chunk = total_size / num_threads;
        C size_t remainder = total_size % num_threads;

        size_t start = 0;
        for (size_t i = 0; i < num_threads && start < total_size; ++i) {
            size_t chunk_len = base_chunk + (i < remainder ? 1 : 0);
            size_t end = start + chunk_len;

            if (end < total_size) {
                C u64 boundary_hash = boards1[end - 1].getHash();
                while (end < total_size && boards1[end].getHash() == boundary_hash) {
                    ++end;
                }
            }

            ranges.emplace_back(start, end);
            start = end;
        }
    }

    std::vector<std::thread> threads;
    threads.reserve(ranges.size());

    std::vector<std::vector<std::pair<C T*, C T*>>> partial_results(ranges.size());

    for (size_t i = 0; i < ranges.size(); ++i) {
        C auto [start1, end1] = ranges[i];

        threads.emplace_back([&, start1, end1, i]() {
            process_chunk<T>(boards1, boards2, start1, end1, partial_results[i]);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::vector<std::pair<C T*, C T*>> results;
    size_t total_pairs = 0;
    for (C auto& partial : partial_results) {
        total_pairs += partial.size();
    }
    results.reserve(total_pairs);

    for (auto& partial : partial_results) {
        results.insert(results.end(), partial.begin(), partial.end());
    }

    std::vector<std::pair<C T*, C T*>> realResults;
    realResults.reserve(results.size());

    for (auto& [fst, snd] : results) {
        if (*fst == *snd) {
            realResults.emplace_back(fst, snd);
        }
    }

    return realResults;
}


// TODO: garbage bad code
#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
std::vector<std::pair<C T *, C T *>> intersection(C JVec<T>& boards1, C JVec<T>& boards2) {
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


template<typename T>
std::vector<std::pair<const T*, const T*>> intersection_all_pairs(
        const JVec<T>& boards1,
        const JVec<T>& boards2) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
    
    std::vector<std::pair<const T*, const T*>> results;

    auto it1 = boards1.begin();
    auto it2 = boards2.begin();

    while (it1 != boards1.end() && it2 != boards2.end()) {
        if (it1->getHash() < it2->getHash()) {
            ++it1;
            continue;
        }
        if (it2->getHash() < it1->getHash()) {
            ++it2;
            continue;
        }

        auto it1_end = it1;
        auto it2_end = it2;

        while (it1_end != boards1.end() && it1_end->getHash() == it1->getHash()) {
            ++it1_end;
        }
        while (it2_end != boards2.end() && it2_end->getHash() == it2->getHash()) {
            ++it2_end;
        }

        for (auto a = it1; a != it1_end; ++a) {
            for (auto b = it2; b != it2_end; ++b) {
                results.emplace_back(&*a, &*b);
            }
        }

        it1 = it1_end;
        it2 = it2_end;
    }

    return results;
}








#pragma once

#include "board.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "utils/jvec.hpp"

template<typename T>
MU static void append_equal_state_pairs_in_bucket(
        C JVec<T>& boards1,
        C JVec<T>& boards2,
        C size_t begin1,
        C size_t end1,
        C size_t begin2,
        C size_t end2,
        std::vector<std::pair<C T*, C T*>>& results) {
    size_t i1 = begin1;
    size_t i2 = begin2;

    while (i1 < end1 && i2 < end2) {
        if (boards1[i1] < boards2[i2]) {
            ++i1;
            continue;
        }
        if (boards2[i2] < boards1[i1]) {
            ++i2;
            continue;
        }

        size_t i1_end = i1 + 1;
        while (i1_end < end1 && boards1[i1_end] == boards1[i1]) {
            ++i1_end;
        }

        size_t i2_end = i2 + 1;
        while (i2_end < end2 && boards2[i2_end] == boards2[i2]) {
            ++i2_end;
        }

        for (size_t a = i1; a < i1_end; ++a) {
            for (size_t b = i2; b < i2_end; ++b) {
                results.emplace_back(&boards1[a], &boards2[b]);
            }
        }

        i1 = i1_end;
        i2 = i2_end;
    }
}

template<typename T>
MU static void process_chunk(
        C JVec<T>& boards1,
        C JVec<u64>& hashes1,
        C JVec<T>& boards2,
        C JVec<u64>& hashes2,
        C size_t start1,
        C size_t end1,
        std::vector<std::pair<C T*, C T*>>& results) {
    C size_t size1 = boards1.size();
    C size_t size2 = boards2.size();

    if (size1 != hashes1.size()) {
        throw std::runtime_error("process_chunk: boards1/hashes1 size mismatch");
    }
    if (size2 != hashes2.size()) {
        throw std::runtime_error("process_chunk: boards2/hashes2 size mismatch");
    }

    if (size1 == 0 || size2 == 0) {
        return;
    }
    if (start1 >= end1) {
        return;
    }
    if (start1 >= size1) {
        return;
    }
    if (end1 > size1) {
        return;
    }

    C u64 min_hash = hashes1[start1];
    C u64 max_hash = hashes1[end1 - 1];

    auto it2_begin = std::lower_bound(hashes2.begin(), hashes2.end(), min_hash);
    auto it2_end_it = std::upper_bound(hashes2.begin(), hashes2.end(), max_hash);

    C size_t start2 = static_cast<size_t>(it2_begin - hashes2.begin());
    C size_t end2 = static_cast<size_t>(it2_end_it - hashes2.begin());

    if (start2 >= end2) {
        return;
    }
    if (start2 >= size2 || end2 > size2) {
        return;
    }

    size_t i1 = start1;
    size_t i2 = start2;

    while (i1 < end1 && i2 < end2) {
        C u64 hash1 = hashes1[i1];
        C u64 hash2 = hashes2[i2];

        if (hash1 < hash2) {
            ++i1;
            continue;
        }
        if (hash2 < hash1) {
            ++i2;
            continue;
        }

        size_t i1_end = i1 + 1;
        while (i1_end < end1 && hashes1[i1_end] == hash1) {
            ++i1_end;
        }

        size_t i2_end = i2 + 1;
        while (i2_end < end2 && hashes2[i2_end] == hash2) {
            ++i2_end;
        }

        append_equal_state_pairs_in_bucket(
                boards1, boards2,
                i1, i1_end,
                i2, i2_end,
                results
        );

        i1 = i1_end;
        i2 = i2_end;
    }
}

template<typename T>
MU std::vector<std::pair<C T*, C T*>> intersection_threaded(
        C JVec<T>& boards1,
        C JVec<u64>& hashes1,
        C JVec<T>& boards2,
        C JVec<u64>& hashes2,
        size_t num_threads = std::thread::hardware_concurrency()) {
    if (boards1.size() != hashes1.size()) {
        throw std::runtime_error("intersection_threaded: boards1/hashes1 size mismatch");
    }
    if (boards2.size() != hashes2.size()) {
        throw std::runtime_error("intersection_threaded: boards2/hashes2 size mismatch");
    }

    C size_t total_size = boards1.size();
    if (total_size == 0 || boards2.empty()) {
        return {};
    }

    if (num_threads == 0) {
        num_threads = 1;
    }
    num_threads = std::min(num_threads, total_size);

    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(num_threads);

    {
        C size_t base_chunk = total_size / num_threads;
        C size_t remainder = total_size % num_threads;

        size_t start = 0;
        for (size_t i = 0; i < num_threads && start < total_size; ++i) {
            size_t chunk_len = base_chunk + (i < remainder ? 1 : 0);
            size_t end = start + chunk_len;

            if (end < total_size) {
                C u64 boundary_hash = hashes1[end - 1];
                while (end < total_size && hashes1[end] == boundary_hash) {
                    ++end;
                }
            }

            ranges.emplace_back(start, end);
            start = end;
        }
    }

    std::vector<std::thread> threads;
    threads.reserve(ranges.size());

    std::vector<std::vector<std::pair<C T*, C T*>>> partial_results(ranges.size());

    for (size_t i = 0; i < ranges.size(); ++i) {
        C auto [start1, end1] = ranges[i];

        threads.emplace_back([&, start1, end1, i]() {
            process_chunk(
                    boards1, hashes1,
                    boards2, hashes2,
                    start1, end1,
                    partial_results[i]
            );
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::vector<std::pair<C T*, C T*>> results;
    size_t total_pairs = 0;
    for (C auto& partial : partial_results) {
        total_pairs += partial.size();
    }
    results.reserve(total_pairs);

    for (auto& partial : partial_results) {
        results.insert(results.end(), partial.begin(), partial.end());
    }

    return results;
}

template<typename T>
MU std::vector<std::pair<C T*, C T*>> intersection(
        C JVec<T>& boards1,
        C JVec<u64>& hashes1,
        C JVec<T>& boards2,
        C JVec<u64>& hashes2) {
    if (boards1.size() != hashes1.size()) {
        throw std::runtime_error("intersection: boards1/hashes1 size mismatch");
    }
    if (boards2.size() != hashes2.size()) {
        throw std::runtime_error("intersection: boards2/hashes2 size mismatch");
    }

    std::vector<std::pair<C T*, C T*>> results;
    if (boards1.empty() || boards2.empty()) {
        return results;
    }

    process_chunk(
            boards1, hashes1,
            boards2, hashes2,
            0, boards1.size(),
            results
    );

    return results;
}

template<typename T>
MU std::vector<std::pair<C T*, C T*>> intersection_all_pairs(
        C JVec<T>& boards1,
        C JVec<u64>& hashes1,
        C JVec<T>& boards2,
        C JVec<u64>& hashes2) {
    return intersection(boards1, hashes1, boards2, hashes2);
}











