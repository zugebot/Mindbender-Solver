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
#include <stdexcept>
#include <vector>

#include "utils/jvec.hpp"

template<typename T>
class BoardSorter {
    std::vector<JVec<T>> auxBoards_{};
    std::vector<JVec<u64>> auxHashes_{};

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

        std::vector<std::size_t> order(boards.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }

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
    MU void resize(const u32 depth, const size_t size) {
        if (auxBoards_.size() < depth + 1) {
            auxBoards_.resize(depth + 1);
            auxHashes_.resize(depth + 1);
        }

        auxBoards_[depth].resize(size);
        auxHashes_[depth].resize(size);
    }

    MU void ensureAux(const u32 depth, const u64 size) {
        if (auxBoards_.size() < depth + 1) {
            auxBoards_.resize(depth + 1);
            auxHashes_.resize(depth + 1);
        }

        if (auxBoards_[depth].capacity() < size) {
            auxBoards_[depth].reserve(size);
        }
        if (auxHashes_[depth].capacity() < size) {
            auxHashes_[depth].reserve(size);
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