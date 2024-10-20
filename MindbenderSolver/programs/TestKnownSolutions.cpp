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
#include <regex>

namespace fs = std::filesystem;


// Function to split the filename and extract X, Y, N, M
std::tuple<int, int, int, int> parseFileName(const std::string& filename) {
    std::regex pattern(R"((\d+)\-(\d+)\_c(\d+)\_(\d+)\.txt)");
    std::smatch match;

    if (std::regex_match(filename, match, pattern)) {
        int X = std::stoi(match[1].str());
        int Y = std::stoi(match[2].str());
        int N = std::stoi(match[3].str());
        int M = std::stoi(match[4].str());

        return std::make_tuple(X, Y, N, M);
    }

    // If the pattern doesn't match, return default values (could handle error)
    return std::make_tuple(0, 0, 0, 0);
}

// Comparator for sorting the files
bool compareFiles(const std::string& file1, const std::string& file2) {
    auto [X1, Y1, N1, M1] = parseFileName(file1);
    auto [X2, Y2, N2, M2] = parseFileName(file2);

    if (X1 != X2) return X1 < X2;
    if (Y1 != Y2) return Y1 < Y2;
    if (N1 != N2) return N1 < N2;
    return M1 < M2;
}


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


int main() {
    // std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    std::string folder = "levels";
    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\MindbenderSolver\)" + folder + "\\";


    auto files = getTxtFiles(outDirectory);
    std::sort(files.begin(), files.end(), compareFiles);
    for (const auto& file : files) {
        auto [X, Y, M, N] = parseFileName(file);

        std::string levelName = std::to_string(X) + "-" + std::to_string(Y);
        if (levelName.size() >= 5) {
            std::cout << "[" << std::setw(4) << levelName
                      << ", " << std::setw(3) << "c" + std::to_string(M)
                      << "] (invalid) skipping" << "...\n";
            continue;
        }

        BoardPair const* pair = BoardLookup::getBoardPair(levelName);
        if (pair == nullptr) {
            std::cout << "[" << std::setw(4) << levelName
                      << ", " << std::setw(3) << "c" + std::to_string(M)
                      << "] (nullptr) skipping" << "...\n";
            continue;
        }


        Board startingBoard = pair->getInitialState();
        Board realSolutionBoard = pair->getSolutionState();

        // std::cout << realSolutionBoard.toString(startingBoard) << std::endl;



        auto solutions = readFileLines(outDirectory + file);

        size_t realSolutionCount = 0;
        size_t totalSolutionCount = solutions.size();
        if (!startingBoard.getFatBool()) {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseNormMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (c_u8 action_number : action_numbers) {
                    allActStructList[action_number].action(toCheckBoard);
                }
                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            std::cout << "--- [" << std::setw(4) << levelName << " / "
                      << std::setw(3) << "c" + std::to_string(M) << "]: ";
            std::cout << realSolutionCount << "/" << totalSolutionCount << "\n";

        } else {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseFatMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (c_u8 action_number : action_numbers) {
                    allActStructList[action_number].action(toCheckBoard);
                }

                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            std::cout << "FAT [" << std::setw(4) << levelName << " / "
                      << std::setw(3) << "c" + std::to_string(M) << "]: ";
            std::cout << realSolutionCount << "/" << totalSolutionCount << "\n";
        }


    }
    return 0;
}
