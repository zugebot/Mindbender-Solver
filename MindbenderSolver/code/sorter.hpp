#pragma once

#include <vector>

#include "board.hpp"
#include "MindbenderSolver/utils/th_parallel_sort.hpp"
#include "MindbenderSolver/utils/th_radix_sort.hpp"


class BoardSorter {
    std::vector<Board> aux;

    enum DEPTH { D2 = 2, D3 = 3, D4 = 4, D5 = 5 };
    enum COLORS { C2 = 2, C3 = 3 };
public:
    MU void resize(const size_t size) {
        aux.resize(size);
    }

    MU void ensureAux(const size_t size) {
        if (aux.capacity() < size) {
            aux.reserve(size);
        }
        resize(size);
    }

    MU void sortBoards(std::vector<Board>& boards, c_u32 depth, c_u32 colorCount) {
        switch (depth) {
            case (DEPTH::D2): {
                std::sort(boards.begin(), boards.end());
                break;
            }
            case (DEPTH::D3): {
                parallel_sort<2>(boards);
                break;
            }
            case (DEPTH::D4): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(boards.size());
                        radix_sort<3, 12>(boards, aux);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(boards.size());
                        radix_sort<5, 12>(boards, aux);
                        break;
                    }
                    default: {
                        ensureAux(boards.size());
                        radix_sort<6, 11>(boards, aux);
                        break;
                    }
                }
                break;
            }
            case (DEPTH::D5): {
                switch (colorCount) {
                    case (COLORS::C2): {
                        ensureAux(boards.size());
                        radix_sort<3, 12>(boards, aux);
                        break;
                    }
                    case (COLORS::C3): {
                        ensureAux(boards.size());
                        radix_sort<6, 10>(boards, aux);
                        break;
                    }
                    default: {
                        ensureAux(boards.size());
                        radix_sort<6, 11>(boards, aux);
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


