#pragma once

#include "src/chuzzles/board.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/utils/hasGetHash.hpp"
#include "src/utils/jvec.hpp"

#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
void process_chunk(
        const JVec<T>& boards1,
        const JVec<T>& boards2,
        const size_t start1,
        const size_t end1,
        std::vector<std::pair<const T*, const T*>>& results) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
    
    const size_t size1 = boards1.size();
    const size_t size2 = boards2.size();

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
        throw std::runtime_error("intersection chunk exceeds left lane size");
    }

    const u64 min_hash = boards1[start1].getHash();
    const u64 max_hash = boards1[end1 - 1].getHash();

    auto it2_begin = std::lower_bound(
            boards2.begin(),
            boards2.end(),
            min_hash,
            [](const T& board, const u64 hash) {
                return board.getHash() < hash;
            });

    auto it2_end_it = std::upper_bound(
            boards2.begin(),
            boards2.end(),
            max_hash,
            [](const u64 hash, const T& board) {
                return hash < board.getHash();
            });

    const size_t start2 = static_cast<size_t>(it2_begin - boards2.begin());
    const size_t end2 = static_cast<size_t>(it2_end_it - boards2.begin());

    if (start2 >= end2) {
        return;
    }
    if (start2 >= size2 || end2 > size2) {
        return;
    }

    size_t i1 = start1;
    size_t i2 = start2;

    while (i1 < end1 && i2 < end2) {
        const u64 hash1 = boards1[i1].getHash();
        const u64 hash2 = boards2[i2].getHash();

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
std::vector<std::pair<const T*, const T*>> intersection_threaded(
        const JVec<T>& boards1,
        const JVec<T>& boards2,
        size_t num_threads = std::thread::hardware_concurrency()) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
    
    const size_t total_size = boards1.size();
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
        size_t start = 0;
        for (size_t i = 0; i < num_threads && start < total_size; ++i) {
            const size_t threadsLeft = num_threads - i;
            const size_t statesLeft = total_size - start;
            const size_t chunkLen = (statesLeft + threadsLeft - 1) / threadsLeft;
            size_t end = std::min(total_size, start + chunkLen);

            if (end < total_size) {
                const u64 boundaryHash = boards1[end - 1].getHash();
                while (end < total_size && boards1[end].getHash() == boundaryHash) {
                    ++end;
                }
            }

            ranges.emplace_back(start, end);
            start = end;
        }
    }

    if (ranges.empty() || ranges.front().first != 0 ||
        ranges.back().second != total_size) {
        throw std::runtime_error("intersection partition did not cover the full left lane");
    }

    std::vector<std::thread> threads;
    threads.reserve(ranges.size());

    std::vector<std::vector<std::pair<const T*, const T*>>> partial_results(ranges.size());

    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [start1, end1] = ranges[i];

        threads.emplace_back([&, start1, end1, i]() {
            process_chunk<T>(boards1, boards2, start1, end1, partial_results[i]);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::vector<std::pair<const T*, const T*>> results;
    size_t total_pairs = 0;
    for (const auto& partial : partial_results) {
        total_pairs += partial.size();
    }
    results.reserve(total_pairs);

    for (auto& partial : partial_results) {
        results.insert(results.end(), partial.begin(), partial.end());
    }

    std::vector<std::pair<const T*, const T*>> realResults;
    realResults.reserve(results.size());

    for (auto& [fst, snd] : results) {
        if (*fst == *snd) {
            realResults.emplace_back(fst, snd);
        }
    }

    return realResults;
}


// Exact single-threaded intersection.  Preserve every matching path pair in
// equal-hash runs and validate the full state before returning it.
#ifdef USE_CUDA
template<typename T>
#else
template<HasGetHash_v T>
#endif
std::vector<std::pair<const T*, const T*>> intersection(
        const JVec<T>& boards1,
        const JVec<T>& boards2) {
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    std::vector<std::pair<const T*, const T*>> results;
    std::size_t i1 = 0;
    std::size_t i2 = 0;

    while (i1 < boards1.size() && i2 < boards2.size()) {
        const u64 hash1 = boards1[i1].getHash();
        const u64 hash2 = boards2[i2].getHash();

        if (hash1 < hash2) {
            ++i1;
            continue;
        }
        if (hash2 < hash1) {
            ++i2;
            continue;
        }

        std::size_t end1 = i1 + 1;
        std::size_t end2 = i2 + 1;
        while (end1 < boards1.size() && boards1[end1].getHash() == hash1) {
            ++end1;
        }
        while (end2 < boards2.size() && boards2[end2].getHash() == hash2) {
            ++end2;
        }

        for (std::size_t a = i1; a < end1; ++a) {
            for (std::size_t b = i2; b < end2; ++b) {
                if (boards1[a] == boards2[b]) {
                    results.emplace_back(&boards1[a], &boards2[b]);
                }
            }
        }

        i1 = end1;
        i2 = end2;
    }

    return results;
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

#include "src/chuzzles/board.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/utils/jvec.hpp"

template<typename T>
MU static void append_equal_state_pairs_in_bucket(
        const JVec<T>& boards1,
        const JVec<T>& boards2,
        const size_t begin1,
        const size_t end1,
        const size_t begin2,
        const size_t end2,
        std::vector<std::pair<const T*, const T*>>& results) {
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
        const JVec<T>& boards1,
        const JVec<u64>& hashes1,
        const JVec<T>& boards2,
        const JVec<u64>& hashes2,
        const size_t start1,
        const size_t end1,
        std::vector<std::pair<const T*, const T*>>& results) {
    const size_t size1 = boards1.size();
    const size_t size2 = boards2.size();

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
        throw std::runtime_error("intersection chunk exceeds left lane size");
    }

    const u64 min_hash = hashes1[start1];
    const u64 max_hash = hashes1[end1 - 1];

    auto it2_begin = std::lower_bound(hashes2.begin(), hashes2.end(), min_hash);
    auto it2_end_it = std::upper_bound(hashes2.begin(), hashes2.end(), max_hash);

    const size_t start2 = static_cast<size_t>(it2_begin - hashes2.begin());
    const size_t end2 = static_cast<size_t>(it2_end_it - hashes2.begin());

    if (start2 >= end2) {
        return;
    }
    if (start2 >= size2 || end2 > size2) {
        return;
    }

    size_t i1 = start1;
    size_t i2 = start2;

    while (i1 < end1 && i2 < end2) {
        const u64 hash1 = hashes1[i1];
        const u64 hash2 = hashes2[i2];

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
MU std::vector<std::pair<const T*, const T*>> intersection_threaded(
        const JVec<T>& boards1,
        const JVec<u64>& hashes1,
        const JVec<T>& boards2,
        const JVec<u64>& hashes2,
        size_t num_threads = std::thread::hardware_concurrency()) {
    if (boards1.size() != hashes1.size()) {
        throw std::runtime_error("intersection_threaded: boards1/hashes1 size mismatch");
    }
    if (boards2.size() != hashes2.size()) {
        throw std::runtime_error("intersection_threaded: boards2/hashes2 size mismatch");
    }

    const size_t total_size = boards1.size();
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
        size_t start = 0;
        for (size_t i = 0; i < num_threads && start < total_size; ++i) {
            const size_t threadsLeft = num_threads - i;
            const size_t statesLeft = total_size - start;
            const size_t chunkLen = (statesLeft + threadsLeft - 1) / threadsLeft;
            size_t end = std::min(total_size, start + chunkLen);

            if (end < total_size) {
                const u64 boundaryHash = hashes1[end - 1];
                while (end < total_size && hashes1[end] == boundaryHash) {
                    ++end;
                }
            }

            ranges.emplace_back(start, end);
            start = end;
        }
    }

    if (ranges.empty() || ranges.front().first != 0 ||
        ranges.back().second != total_size) {
        throw std::runtime_error("intersection partition did not cover the full left lane");
    }

    std::vector<std::thread> threads;
    threads.reserve(ranges.size());

    std::vector<std::vector<std::pair<const T*, const T*>>> partial_results(ranges.size());

    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [start1, end1] = ranges[i];

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

    std::vector<std::pair<const T*, const T*>> results;
    size_t total_pairs = 0;
    for (const auto& partial : partial_results) {
        total_pairs += partial.size();
    }
    results.reserve(total_pairs);

    for (auto& partial : partial_results) {
        results.insert(results.end(), partial.begin(), partial.end());
    }

    return results;
}

template<typename T>
MU std::vector<std::pair<const T*, const T*>> intersection(
        const JVec<T>& boards1,
        const JVec<u64>& hashes1,
        const JVec<T>& boards2,
        const JVec<u64>& hashes2) {
    if (boards1.size() != hashes1.size()) {
        throw std::runtime_error("intersection: boards1/hashes1 size mismatch");
    }
    if (boards2.size() != hashes2.size()) {
        throw std::runtime_error("intersection: boards2/hashes2 size mismatch");
    }

    std::vector<std::pair<const T*, const T*>> results;
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
MU std::vector<std::pair<const T*, const T*>> intersection_all_pairs(
        const JVec<T>& boards1,
        const JVec<u64>& hashes1,
        const JVec<T>& boards2,
        const JVec<u64>& hashes2) {
    return intersection(boards1, hashes1, boards2, hashes2);
}











