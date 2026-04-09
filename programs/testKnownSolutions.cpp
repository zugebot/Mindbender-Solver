#include "code/include.hpp"

#include <cmath>
#include <fstream>
#include <set>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <regex>


// Function to split the filename and extract X, Y, N, M
std::tuple<int, int, int, int> parseFileName(const fs::path&filepath) {
    std::regex pattern(R"((\d+)\-(\d+)\_c(\d+)\_(\d+)\.txt)");
    std::smatch match;
    
    const std::string filename = filepath.filename().string();
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
bool compareFiles(const fs::path& file1, const fs::path& file2) {
    auto [X1, Y1, N1, M1] = parseFileName(file1.filename().string());
    auto [X2, Y2, N2, M2] = parseFileName(file2.filename().string());

    if (X1 != X2) return X1 < X2;
    if (Y1 != Y2) return Y1 < Y2;
    if (N1 != N2) return N1 < N2;
    return M1 < M2;
}


std::vector<fs::path> getTxtFiles(const fs::path& directory) {
    std::vector<fs::path> txtFiles;

    // Iterate through all files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            txtFiles.push_back(entry.path());
        }
    }

    return txtFiles;
}


std::vector<std::string> readFileLines(const fs::path&filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    std::string line;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}


int main() {
    
    fs::path outDirectory = fs::path(R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\levels)");
    
    auto files = getTxtFiles(outDirectory);
    std::sort(files.begin(), files.end(), compareFiles);
    for (const auto& file : files) {
        auto [X, Y, M, N] = parseFileName(file);

        if (X == 6 && Y < 5) { continue; }
        std::string levelName = std::to_string(X) + "-" + std::to_string(Y);
        if (levelName.size() >= 5) {
            tcout << "[" << std::setw(4) << levelName
                      << ", " << std::setw(3) << "c" + std::to_string(M)
                      << "] (invalid) skipping" << "...\n";
            continue;
        }

        BoardPair const* pair = BoardLookup::getBoardPair(levelName);
        if (pair == nullptr) {
            tcout << "[" << std::setw(4) << levelName
                      << ", " << std::setw(3) << "c" + std::to_string(M)
                      << "] (nullptr) skipping" << "...\n";
            continue;
        }


        Board startingBoard = pair->getStartState();
        Board realSolutionBoard = pair->getEndState();

        // tcout << realSolutionBoard.toString(startingBoard) << std::endl;



        auto solutions = readFileLines(file);

        size_t realSolutionCount = 0;
        size_t totalSolutionCount = solutions.size();
        if (!startingBoard.getFatBool()) {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseNormMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (const u8 action_number : action_numbers) {
                    allActStructList[action_number].action(toCheckBoard);
                }
                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            tcout << "--- [" << std::setw(4) << levelName << " / "
                      << std::setw(3) << "c" + std::to_string(M) << "]: ";
            tcout << realSolutionCount << "/" << totalSolutionCount << "\n";

        } else {
            for (const auto& solution : solutions) {
                std::vector<u8> action_numbers = Memory::parseFatMoveString(solution);
                Board toCheckBoard = startingBoard;
                for (const u8 action_number : action_numbers) {
                    allActStructList[action_number].action(toCheckBoard);
                }

                if (toCheckBoard == realSolutionBoard) {
                    realSolutionCount += 1;
                }
            }

            tcout << "FAT [" << std::setw(4) << levelName << " / "
                      << std::setw(3) << "c" + std::to_string(M) << "]: ";
            tcout << realSolutionCount << "/" << totalSolutionCount << "\n";
        }


    }
    return 0;
}
