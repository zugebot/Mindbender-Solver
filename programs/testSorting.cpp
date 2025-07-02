#include <algorithm>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>


#include "code/board.hpp"
#include "unused/sorter.hpp"
#include "unused/th_parallel_sort.hpp"
#include "utils/timer.hpp"

#include <boost/sort/sort.hpp>


// 2 colors: 3, 12
// 3 colors: 6, 10


int main() {
    const size_t size = 1'000'000;

    JVec<Board> data1(size); data1.resize(size);
    JVec<Board> data2(size); data2.resize(size);

    JVec<Board> aux(size); aux.resize(size);

    BoardSorter<Board> sorter; sorter.resize(5, data1.size());

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 0x0FFF'FFFF'FFFF'FFFF);

    for (u64 index = 0; index < size; index++) {
        uint64_t value = dist(gen);
        data1[index].hashMem.setHash(value);
        data2[index].hashMem.setHash(value);
    }



    std::cout << "Starting Sort1 now! (threaded radix)" << std::endl;
    const Timer timer1;

    sorter.sortBoards(data1, 5, 3);

    std::cout << "Sort Time: " << timer1.getSeconds() << std::endl;
    if (std::is_sorted(data1.begin(), data1.end())) {
        std::cout << "Sorting successful!" << std::endl;
    } else {
        std::cout << "Sorting failed!" << std::endl;
    }



    std::cout << "\nStarting Sort2 now! (boost::block_indirect)" << std::endl;
    const Timer timer2;

    boost::sort::block_indirect_sort(data2.begin(), data2.end());

    std::cout << "Sort Time: " << timer2.getSeconds() << std::endl;
    if (std::is_sorted(data2.begin(), data2.end())) {
        std::cout << "Sorting successful!" << std::endl;
    } else {
        std::cout << "Sorting failed!" << std::endl;
    }


    return 0;
}




