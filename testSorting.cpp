#include <algorithm>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>


#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/sorter.hpp"
#include "MindbenderSolver/utils/th_parallel_sort.hpp"
#include "MindbenderSolver/utils/th_radix_sort.hpp"
#include "MindbenderSolver/utils/timer.hpp"


// 2 colors: 3, 12
// 3 colors: 6, 10


int main() {
    const size_t size = 173325000;
    std::vector<Board> data(size);
    std::vector<Board> aux(size);
    BoardSorter sorter;
    sorter.resize(data.size());

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 0x0FFF'FFFF'FFFF'FFFF);
    for (auto& obj : data) { obj.hash = dist(gen); }
    std::cout << "Starting sort now!" << std::endl;
    const Timer timer;


    sorter.sortBoards(data, 5, 3);


    std::cout << "Sort Time: " << timer.getSeconds() << std::endl;
    if (std::is_sorted(data.begin(), data.end())) {
        std::cout << "Sorting successful!" << std::endl;
    } else {
        std::cout << "Sorting failed!" << std::endl;
    }




    return 0;
}




