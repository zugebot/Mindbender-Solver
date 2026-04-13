#pragma once
// code/solver/sorter.hpp

#ifdef BOOST_FOUND
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#else
#include "utils/th_parallel_sort.hpp"
#include "utils/th_radix_sort.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "utils/jvec.hpp"

template<typename T>
class BoardSorter {
    std::vector<JVec<T>> auxBoards_{};
    std::vector<JVec<u64>> auxHashes_{};
    std::vector<std::vector<std::size_t>> auxOrder_{};

    enum DEPTH { D2 = 2, D3 = 3, D4 = 4, D5 = 5 };
    enum COLORS { C2 = 2, C3 = 3 };

    MU static void normalizeEqualHashRuns(JVec<T>& boards,
                                          const JVec<u64>& hashes) {
        if (boards.size() <= 1) {
            return;
        }

        std::size_t begin = 0;
        while (begin < boards.size()) {
            std::size_t end = begin + 1;
            const u64 hash = hashes[begin];

            while (end < boards.size() && hashes[end] == hash) {
                ++end;
            }

            if (end - begin > 1) {
                std::sort(boards.begin() + begin, boards.begin() + end);
            }

            begin = end;
        }
    }

    MU void sortByHashThenStateWithAux(JVec<T>& boards,
                                       JVec<u64>& hashes,
                                       const u32 depth) {
        ensureAux(depth, boards.size());

        auto& order = auxOrder_[depth];
        std::iota(order.begin(), order.end(), static_cast<std::size_t>(0));

        std::sort(order.begin(), order.end(), [&](const std::size_t lhs, const std::size_t rhs) {
            if (hashes[lhs] < hashes[rhs]) {
                return true;
            }
            if (hashes[rhs] < hashes[lhs]) {
                return false;
            }
            return boards[lhs] < boards[rhs];
        });

        for (std::size_t i = 0; i < order.size(); ++i) {
            auxBoards_[depth][i] = boards[order[i]];
            auxHashes_[depth][i] = hashes[order[i]];
        }

        boards.swap(auxBoards_[depth]);
        hashes.swap(auxHashes_[depth]);

        auxBoards_[depth].clear();
        auxHashes_[depth].clear();
    }

public:
    MU void ensureDepthSlots(const u32 maxDepth) {
        const std::size_t needed = static_cast<std::size_t>(maxDepth) + 1;
        if (auxBoards_.size() >= needed) {
            return;
        }

        // Reserve outer slot vectors once so increasing depth does not repeatedly reallocate metadata lanes.
        if (auxBoards_.capacity() < needed) {
            auxBoards_.reserve(needed);
            auxHashes_.reserve(needed);
            auxOrder_.reserve(needed);
        }

        auxBoards_.resize(needed);
        auxHashes_.resize(needed);
        auxOrder_.resize(needed);
    }

    MU void resize(const u32 depth, const size_t size) {
        ensureDepthSlots(depth);

        auxBoards_[depth].resize(size);
        auxHashes_[depth].resize(size);
        auxOrder_[depth].resize(size);
    }

    MU void ensureAux(const u32 depth, const u64 size) {
        ensureDepthSlots(depth);

        if (auxBoards_[depth].capacity() < size) {
            auxBoards_[depth].reserve(size);
        }
        if (auxHashes_[depth].capacity() < size) {
            auxHashes_[depth].reserve(size);
        }
        if (auxOrder_[depth].capacity() < size) {
            auxOrder_[depth].reserve(size);
        }

        resize(depth, size);
    }

    MU void sortBoards(JVec<T>& boards,
                       JVec<u64>& hashes,
                       const u32 depth,
                       const u32 colorCount) {
        if (boards.size() != hashes.size()) {
            throw std::runtime_error("BoardSorter::sortBoards got mismatched board/hash lane sizes");
        }

        if (boards.size() <= 1) {
            return;
        }

        switch (depth) {
            case (DEPTH::D2): {
                sortByHashThenStateWithAux(boards, hashes, depth);
                break;
            }
            case (DEPTH::D3): {
                sortByHashThenStateWithAux(boards, hashes, depth);
                // parallel_sort<2>(boards, hashes);
                break;
            }
            case (DEPTH::D4): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                    default: {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                }
                break;
            }
            case (DEPTH::D5): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                    default: {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, hashes, auxBoards_[depth], auxHashes_[depth]);
                        normalizeEqualHashRuns(boards, hashes);
                        break;
                    }
                }
                break;
            }
            default: {
                sortByHashThenStateWithAux(boards, hashes, depth);
                break;
            }
        }
    }
};