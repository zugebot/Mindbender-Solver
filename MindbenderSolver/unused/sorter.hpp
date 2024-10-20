#pragma once

#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <vector>

#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/utils/jvec.hpp"
#include "th_parallel_sort.hpp"
#include "th_radix_sort.hpp"


template<typename T>
class BoardSorter {
    std::vector<JVec<T>> aux_buffer{};

    enum DEPTH { D2 = 2, D3 = 3, D4 = 4, D5 = 5 };
    enum COLORS { C2 = 2, C3 = 3 };
public:

    MU void resize(const u32 depth, const size_t size) {
        if (aux_buffer.size() < depth) {
            aux_buffer.resize(depth + 1);
        }
        aux_buffer[depth].resize(size);
    }

    MU void ensureAux(const u32 depth, const u64 size) {
        if (aux_buffer.size() < depth + 1) {
            aux_buffer.resize(depth + 1);
        }

        if (aux_buffer[depth].capacity() < size) {
            aux_buffer[depth].reserve(size);
        }
        resize(depth, size);
    }

    MU void sortBoards(JVec<T>& boards, c_u32 depth, c_u32 colorCount) {
        // boost::sort::block_indirect_sort(boards.begin(), boards.end());
        // return;

        switch (depth) {
            case (DEPTH::D2): {
                std::sort(boards.begin(), boards.end());
                break;
            }
            case (DEPTH::D3): {
                std::sort(boards.begin(), boards.end());
                // parallel_sort<2>(boards);
                break;
            }
            case (DEPTH::D4): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(depth, boards.size());
                        radix_sort<3, 12>(boards, aux_buffer[depth]);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(depth, boards.size());
                        radix_sort<5, 12>(boards, aux_buffer[depth]);
                        break;
                    }
                    default: {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, aux_buffer[depth]);
                        break;
                    }
                }
                break;
            }
            case (DEPTH::D5): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(depth, boards.size());
                        radix_sort<3, 12>(boards, aux_buffer[depth]);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 10>(boards, aux_buffer[depth]);
                        break;
                    }
                    default: {
                        ensureAux(depth, boards.size());
                        radix_sort<6, 11>(boards, aux_buffer[depth]);
                        break;
                    }
                }
                break;
            }
            default: {
                std::sort(boards.begin(), boards.end());
                break;

            }
        }

    }

};


