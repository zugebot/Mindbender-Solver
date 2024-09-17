#include "MindbenderSolver/solver/board.hpp"
#include "MindbenderSolver/solver/levels.hpp"
#include "MindbenderSolver/solver/memory.hpp"
#include "MindbenderSolver/solver/permutations.hpp"
#include "MindbenderSolver/solver/rotations.hpp"
#include "MindbenderSolver/utils/timer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

// uint64_t ROW = 0;
// auto *uncoveredRows = reinterpret_cast<uint8_t *>(&ROW);
// uint64_t COL = 0;
// auto *uncoveredCols = reinterpret_cast<uint8_t *>(&COL);
// uncoveredRows[5] = 1;


int main() {
    auto first  = Levels::b8_3;
    auto second = Levels::s8_3;
    static constexpr int DEPTH_1 = 5;
    static constexpr int DEPTH_2 = 5;
    static constexpr bool backwards = false;


    Board board1, board2;
    if (!backwards) {
        board1.setState(first);
        board2.setState(second);
    } else {
        board1.setState(first);
        board2.setState(second);
        std::cout << "backwards!" << std::endl;
    }

    auto BOARD_SORTER = [](const Board &a, const Board &b) { return a.hash < b.hash; };



    const Timer timer1;
    std::cout << "Creating the initial-state boards...\n";
    std::vector<Board> boards1 = (makePermutationListFuncs[DEPTH_1])(board1);
    std::cout << "Size: " << boards1.size() << std::endl;
    std::cout << "Time: " << timer1.getSeconds() << "\n";
    std::cout << "Sorting..." << std::endl;
    Timer timerSort1;
    std::sort(boards1.begin(), boards1.end(), BOARD_SORTER);
    std::cout << "Sort Time: " << timerSort1.getSeconds() << "\n";
    std::cout << "\n";



    std::cout << "Creating the solution-state boards...\n";
    const Timer timer2;
    std::vector<Board> boards2 = (makePermutationListFuncs[DEPTH_2])(board2);
    std::cout << "Size: " << boards2.size() << std::endl;
    std::cout << "Time: " << timer2.getSeconds() << "\n";
    std::cout << "Sorting..." << std::endl;
    Timer timerSort2;
    std::sort(boards2.begin(), boards2.end(), BOARD_SORTER);
    std::cout << "Sort Time: " << timerSort2.getSeconds() << "\n";
    std::cout << "\n";

    std::vector<std::pair<Board *, Board *>> results;
    auto it1 = boards1.begin();
    auto it2 = boards2.begin();

    while (it1 != boards1.end() && it2 != boards2.end()) {
        if (it1->hash == it2->hash) {
            auto it1_end = it1;
            auto it2_end = it2;
            // find range of matching hashes in boards1
            while (it1_end != boards1.end() && it1_end->hash == it1->hash) {
                ++it1_end;
            }
            // find range of matching hashes in boards2
            while (it2_end != boards2.end() && it2_end->hash == it2->hash) {
                ++it2_end;
            }
            // make pairs for all combinations of matching hashes
            for (auto it1_match = it1; it1_match != it1_end; ++it1_match) {
                for (auto it2_match = it2; it2_match != it2_end; ++it2_match) {
                    results.emplace_back(&*it1_match, &*it2_match);
                }
            }
            it1 = it1_end;
            it2 = it2_end;
        } else if (it1->hash < it2->hash) {
            ++it1;
        } else {
            ++it2;
        }
    }


    /*
    for (auto pair: results) {
        std::string moveset;
        if (backwards) {
            moveset = pair.second->assembleMoveString(pair.first);
        } else {
            moveset = pair.first->assembleMoveString(pair.second);
        }
        std::cout << moveset << std::endl;
    }
     */
    std::cout << "\nSolutions: " << results.size() << std::endl;

    std::ofstream outfile("movesets.txt");
    for (auto pair: results) {
        std::string moveset;
        if (backwards) {
            moveset = pair.second->assembleMoveString(pair.first);
        } else {
            moveset = pair.first->assembleMoveString(pair.second);
        }
        outfile << moveset << std::endl;
    }
    outfile.close();




    return 0;
}
