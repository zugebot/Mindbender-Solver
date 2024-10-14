#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/memory.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/rotations.hpp"
#include "MindbenderSolver/utils/timer.hpp"

#include <filesystem>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>

#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>

namespace fs = std::filesystem;


std::vector<std::string> getTxtFiles(const std::string& directory) {
    std::vector<std::string> txtFiles;

    // Iterate through all files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            txtFiles.push_back(entry.path().filename().string());
        }
    }

    return txtFiles;
}


std::vector<std::string> readFileLines(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}


std::string extractSegment(const std::string& input) {
    size_t pos = input.find('_');
    if (pos != std::string::npos) {
        return input.substr(0, pos);
    }
    return input; // If no underscore found, return the whole string.
}



int main() {
    // std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    std::string folder = "levels";
    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\)" + folder + "\\";




    auto files = getTxtFiles(outDirectory);
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        std::string levelName = extractSegment(file);
        if (levelName.size() >= 5) {
            std::cout << "[" << levelName << "] (invalid) skipping" << "...\n";
            continue;
        }

        BoardPair const* pair = BoardLookup::getBoardPair(levelName);
        if (pair == nullptr) {
            std::cout << "[" << levelName << "] (nullptr) skipping" << "...\n";
            continue;
        }


        Board startingBoard = pair->getInitialState();
        Board realSolutionBoard = pair->getSolutionState();

        std::cout << realSolutionBoard.toString(startingBoard) << std::endl;



        auto solutions = readFileLines(outDirectory + file);

        size_t realSolutionCount = 0;
        size_t totalSolutionCount = solutions.size();
        if (!startingBoard.getFatBool()) {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseNormMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (c_u8 action_number : action_numbers) {
                    allActionsList[action_number](toCheckBoard);
                }
                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            std::cout << "Norm [" << levelName << "]: ";
            std::cout << realSolutionCount << "/" << totalSolutionCount << "\n";

        } else {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseFatMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (c_u8 action_number : action_numbers) {
                    allActionsList[action_number](toCheckBoard);
                }

                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            std::cout << "Fat. [" << levelName << "]: ";
            std::cout << realSolutionCount << "/" << totalSolutionCount << "\n";
        }


    }
    return 0;
}