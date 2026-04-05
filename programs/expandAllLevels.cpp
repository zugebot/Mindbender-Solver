#include "code/include.hpp"
#include "code/memory_perm_gen.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// Hardcoded directories
// -----------------------------------------------------------------------------
static const std::vector<fs::path> INPUT_DIRS = {
        R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\levels)",
        R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\levels_all)"
};

static const fs::path OUTPUT_DIR =
        R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\levels_exp)";

static constexpr int MAX_INPUT_FILE_M = 10000;

// -----------------------------------------------------------------------------
// Filename parsing
// Matches: X-Y_cN_M.txt
// -----------------------------------------------------------------------------
static std::tuple<int, int, int, int> parseFileName(const fs::path& filepath) {
    static const std::regex pattern(R"((\d+)\-(\d+)\_c(\d+)\_(\d+)\.txt)");

    std::smatch match;
    const std::string filename = filepath.filename().string();

    if (std::regex_match(filename, match, pattern)) {
        const int x = std::stoi(match[1].str());
        const int y = std::stoi(match[2].str());
        const int c = std::stoi(match[3].str());
        const int m = std::stoi(match[4].str());
        return {x, y, c, m};
    }

    return {0, 0, 0, 0};
}

static bool compareFiles(const fs::path& file1, const fs::path& file2) {
    const auto [x1, y1, c1, m1] = parseFileName(file1);
    const auto [x2, y2, c2, m2] = parseFileName(file2);

    if (x1 != x2) return x1 < x2;
    if (y1 != y2) return y1 < y2;
    if (c1 != c2) return c1 < c2;
    return m1 < m2;
}

// -----------------------------------------------------------------------------
// Group key: same puzzle species means same X-Y and same c
// -----------------------------------------------------------------------------
struct PuzzleKey {
    int x = 0;
    int y = 0;
    int c = 0;

    bool operator<(const PuzzleKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return c < other.c;
    }
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static std::string makeLevelName(const int x, const int y) {
    return std::to_string(x) + "-" + std::to_string(y);
}

static std::string makeOutputFileName(const int x,
                                      const int y,
                                      const int c,
                                      const std::size_t outputCount) {
    return std::to_string(x) + "-" +
           std::to_string(y) + "_c" +
           std::to_string(c) + "_" +
           std::to_string(outputCount) + ".txt";
}

static std::string trimCopy(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static std::vector<std::string> tokenizeMoves(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);

    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

static std::string joinTokens(const std::vector<std::string>& tokens) {
    std::string out;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            out += ' ';
        }
        out += tokens[i];
    }
    return out;
}

static std::vector<std::string> readFileLines(const fs::path& filepath) {
    std::vector<std::string> lines;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open input file");
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trimCopy(line);
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    return lines;
}

static void writeLines(const fs::path& filepath, const std::set<std::string>& lines) {
    std::ofstream file(filepath, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open output file");
    }

    for (const std::string& line : lines) {
        file << line << '\n';
    }
}

static void appendTxtFilesFromDirectory(const fs::path& directory,
                                        std::vector<fs::path>& outFiles) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            outFiles.push_back(entry.path());
        }
    }
}

static void removeExistingSpeciesOutputs(const fs::path& outputDir,
                                         const int x,
                                         const int y,
                                         const int c) {
    if (!fs::exists(outputDir) || !fs::is_directory(outputDir)) {
        return;
    }

    const std::regex pattern(
            "^" + std::to_string(x) + "\\-" +
            std::to_string(y) + "_c" +
            std::to_string(c) + "_\\d+\\.txt$"
    );

    for (const auto& entry : fs::directory_iterator(outputDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, pattern)) {
            std::error_code ec;
            fs::remove(entry.path(), ec);
        }
    }
}

// -----------------------------------------------------------------------------
// Move helpers
// -----------------------------------------------------------------------------
static bool applyMovesAndCheckGoal(Board board,
                                   const Board& goal,
                                   const std::vector<u8>& moves) {
    for (const u8 move : moves) {
        allActStructList[move].action(board);
    }
    return board == goal;
}

static std::string movesToAsmString(const std::vector<u8>& moves) {
    return moveVectorToDirectString(moves);
}

static bool tryParseValidatedLine(const std::string& rawLine,
                                  const bool isFatPuzzle,
                                  const std::size_t expectedMoveCount,
                                  std::vector<u8>& outMoves,
                                  std::string& outCanonicalLine) {
    const std::vector<std::string> inputTokens = tokenizeMoves(rawLine);
    if (inputTokens.empty()) {
        return false;
    }

    if (inputTokens.size() != expectedMoveCount) {
        return false;
    }

    try {
        outMoves = isFatPuzzle
                           ? Memory::parseFatMoveString(rawLine)
                           : Memory::parseNormMoveString(rawLine);
    } catch (...) {
        return false;
    }

    if (outMoves.size() != expectedMoveCount) {
        return false;
    }

    outCanonicalLine = movesToAsmString(outMoves);
    const std::vector<std::string> canonicalTokens = tokenizeMoves(outCanonicalLine);

    if (canonicalTokens.size() != expectedMoveCount) {
        return false;
    }

    if (joinTokens(inputTokens) != joinTokens(canonicalTokens)) {
        return false;
    }

    return true;
}

static bool tryProcessNormalLine(const Board& start,
                                 const Board& goal,
                                 const std::size_t expectedMoveCount,
                                 const std::string& solution,
                                 std::set<std::string>& expandedSolutions) {
    std::vector<u8> baseMoves;
    std::string canonicalBaseLine;

    if (!tryParseValidatedLine(solution, false, expectedMoveCount, baseMoves, canonicalBaseLine)) {
        return false;
    }

    if (!applyMovesAndCheckGoal(start, goal, baseMoves)) {
        return false;
    }

    std::vector<std::vector<u8>> perms;
    try {
        perms = createMemoryPermutationsChecked(start, goal, baseMoves);
    } catch (...) {
        return false;
    }

    expandedSolutions.insert(canonicalBaseLine);

    for (const std::vector<u8>& perm : perms) {
        if (perm.size() != expectedMoveCount) {
            continue;
        }

        if (!applyMovesAndCheckGoal(start, goal, perm)) {
            continue;
        }

        expandedSolutions.insert(movesToAsmString(perm));
    }

    return true;
}

static bool tryProcessFatLine(const Board& start,
                              const Board& goal,
                              const std::size_t expectedMoveCount,
                              const std::string& solution,
                              std::set<std::string>& fatSolutions) {
    std::vector<u8> baseMoves;
    std::string canonicalLine;

    if (!tryParseValidatedLine(solution, true, expectedMoveCount, baseMoves, canonicalLine)) {
        return false;
    }

    if (!applyMovesAndCheckGoal(start, goal, baseMoves)) {
        return false;
    }

    fatSolutions.insert(canonicalLine);
    return true;
}

// -----------------------------------------------------------------------------
// Group processing
// -----------------------------------------------------------------------------
struct GroupResult {
    bool isFatPuzzle = false;
    std::size_t sourceFileCount = 0;
    std::size_t rawLineCount = 0;
    std::size_t validLineCount = 0;
    std::size_t skippedLineCount = 0;
    std::size_t outputCount = 0;
};

static GroupResult processGroup(const PuzzleKey& key,
                                std::vector<fs::path> files,
                                const fs::path& outputDir) {
    if (key.x == 0 && key.y == 0 && key.c == 0) {
        throw std::runtime_error("invalid group key");
    }

    std::sort(files.begin(), files.end(), compareFiles);

    const std::string levelName = makeLevelName(key.x, key.y);
    const BoardPair* pair = BoardLookup::getBoardPair(levelName);
    if (pair == nullptr) {
        throw std::runtime_error("level lookup failed");
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();
    const bool isFatPuzzle = start.getFatBool() || goal.getFatBool();

    std::set<std::string> outputSolutions;
    std::size_t rawLineCount = 0;
    std::size_t validLineCount = 0;
    std::size_t skippedLineCount = 0;

    for (const fs::path& file : files) {
        tcout << "Reading file: " << file.string() << '\n' << std::flush;
        
        const std::vector<std::string> lines = readFileLines(file);

        for (const std::string& line : lines) {
            ++rawLineCount;

            const bool ok = isFatPuzzle
                                    ? tryProcessFatLine(start, goal, static_cast<std::size_t>(key.c), line, outputSolutions)
                                    : tryProcessNormalLine(start, goal, static_cast<std::size_t>(key.c), line, outputSolutions);

            if (ok) {
                ++validLineCount;
            } else {
                ++skippedLineCount;
            }
        }
    }

    fs::create_directories(outputDir);
    removeExistingSpeciesOutputs(outputDir, key.x, key.y, key.c);

    const std::size_t outputCount = outputSolutions.size();
    const fs::path outputFile = outputDir / makeOutputFileName(key.x, key.y, key.c, outputCount);

    writeLines(outputFile, outputSolutions);

    return {
            isFatPuzzle,
            files.size(),
            rawLineCount,
            validLineCount,
            skippedLineCount,
            outputCount
    };
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    tcout << "Input dirs:\n";
    for (const fs::path& dir : INPUT_DIRS) {
        tcout << "  " << dir.string() << '\n';
    }
    tcout << "Output dir:\n";
    tcout << "  " << OUTPUT_DIR.string() << "\n\n";

    std::vector<fs::path> allFiles;
    for (const fs::path& dir : INPUT_DIRS) {
        appendTxtFilesFromDirectory(dir, allFiles);
    }

    std::sort(allFiles.begin(), allFiles.end(), compareFiles);

    if (allFiles.empty()) {
        tcout << "No .txt files found.\n";
        return 0;
    }

    std::map<PuzzleKey, std::vector<fs::path>> groups;
    std::size_t skippedFiles = 0;

    for (const fs::path& file : allFiles) {
        const auto [x, y, c, m] = parseFileName(file);

        if (x == 0 && y == 0 && c == 0 && m == 0) {
            tcout << "Skipping file: " << file.filename().string() << " | Reason: invalid filename format\n";
            ++skippedFiles;
            continue;
        }

        if (m > MAX_INPUT_FILE_M) {
            tcout << "Skipping file: " << file.filename().string() << " | Reason: m > " << MAX_INPUT_FILE_M << '\n';
            ++skippedFiles;
            continue;
        }

        groups[{x, y, c}].push_back(file);
    }

    if (groups.empty()) {
        tcout << "No valid grouped files found.\n";
        return 0;
    }

    std::size_t successGroups = 0;
    std::size_t skippedGroups = 0;
    std::size_t fatGroups = 0;

    for (const auto& [key, files] : groups) {
        try {
            const std::string levelName = makeLevelName(key.x, key.y);
            tcout << "\nProcessing group: " << levelName << "_c" << key.c << '\n' << std::flush;
            const GroupResult result = processGroup(key, files, OUTPUT_DIR);

            if (result.isFatPuzzle) {
                ++fatGroups;
                tcout << "FAT ["
                          << std::setw(4) << levelName
                          << ", " << std::setw(3) << ("c" + std::to_string(key.c))
                          << "] files: " << std::setw(2) << result.sourceFileCount
                          << " | raw: " << std::setw(5) << result.rawLineCount
                          << " | valid: " << std::setw(5) << result.validLineCount
                          << " | skipped lines: " << std::setw(5) << result.skippedLineCount
                          << " | out: " << result.outputCount
                          << '\n';
            } else {
                tcout << "["
                          << std::setw(4) << levelName
                          << ", " << std::setw(3) << ("c" + std::to_string(key.c))
                          << "] files: " << std::setw(2) << result.sourceFileCount
                          << " | raw: " << std::setw(5) << result.rawLineCount
                          << " | valid: " << std::setw(5) << result.validLineCount
                          << " | skipped lines: " << std::setw(5) << result.skippedLineCount
                          << " | out: " << result.outputCount
                          << '\n';
            }

            ++successGroups;
        } catch (const std::exception& e) {
            tcout << "Skipping group: "
                      << key.x << "-" << key.y << "_c" << key.c
                      << " | Reason: " << e.what() << '\n';
            ++skippedGroups;
        }
    }

    tcout << "\nDone.\n";
    tcout << "Succeeded groups: " << successGroups << '\n';
    tcout << "Fat groups      : " << fatGroups << '\n';
    tcout << "Skipped groups  : " << skippedGroups << '\n';
    tcout << "Skipped files   : " << skippedFiles << '\n';

    return 0;
}