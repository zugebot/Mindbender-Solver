// src/programs/solvers/findSolutions.cpp
#include "src/chuzzles/include.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "include/nlohmann/json.hpp"
#include "src/solver/solver_frontier.hpp"
#include "src/utils/get_free_memory.hpp"


struct FindSolutionConfig {
    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    BoardSolverFrontier::SearchDirection searchDirection = BoardSolverFrontier::SearchDirection::Auto;
    bool ignoredExperimentalSettings = false;
    bool debug = true;
    std::string puzzle = "14-2";
    int estimatedDepth = 10;
    int threads = 12;
    std::vector<std::string> unsolvedPuzzles;
};

static std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

static BoardSolverFrontier::SearchDirection parseSearchDirection(const nlohmann::json& value) {
    using SearchDirection = BoardSolverFrontier::SearchDirection;

    if (value.is_number_integer()) {
        switch (value.get<int>()) {
            case 0: return SearchDirection::Auto;
            case 1: return SearchDirection::Forward;
            case 2: return SearchDirection::Reverse;
            default: return SearchDirection::Auto;
        }
    }

    if (value.is_string()) {
        const std::string s = toLowerCopy(value.get<std::string>());
        if (s == "auto") {
            return SearchDirection::Auto;
        }
        if (s == "forward") {
            return SearchDirection::Forward;
        }
        if (s == "reverse") {
            return SearchDirection::Reverse;
        }
    }

    return SearchDirection::Auto;
}

static FindSolutionConfig 
loadFindSolutionConfig() {
    FindSolutionConfig cfg;
    namespace fs = std::filesystem;

    const fs::path sourceDir = fs::path(__FILE__).parent_path();
    const fs::path candidatePaths[] = {
            sourceDir / "config.json",
            fs::current_path() / "config.json"
    };

    fs::path configPath;
    for (const fs::path& candidate: candidatePaths) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            configPath = candidate;
            break;
        }
    }

    if (configPath.empty()) {
        return cfg;
    }

    try {
        std::ifstream in(configPath);
        if (!in.is_open()) {
            return cfg;
        }

        nlohmann::json j;
        in >> j;

        if (const auto it = j.find("outDirectory"); it != j.end() && it->is_string()) {
            cfg.outDirectory = it->get<std::string>();
        }

        if (const auto it = j.find("searchDirection"); it != j.end()) {
            cfg.searchDirection = parseSearchDirection(*it);
        }

        static constexpr const char* REMOVED_EXPERIMENTAL_KEYS[] = {
                "rightFrontierIndex",
                "rightFrontierCachelineLoadFactor",
                "leftGeneration",
                "rightGeneration",
                "probeStream"
        };
        for (const char* key : REMOVED_EXPERIMENTAL_KEYS) {
            if (j.find(key) != j.end()) {
                cfg.ignoredExperimentalSettings = true;
            }
        }

        if (const auto it = j.find("debug"); it != j.end() && it->is_boolean()) {
            cfg.debug = it->get<bool>();
        }

        if (const auto it = j.find("puzzle"); it != j.end() && it->is_string()) {
            cfg.puzzle = it->get<std::string>();
        }

        if (const auto it = j.find("estimatedDepth"); it != j.end() && it->is_number_integer()) {
            cfg.estimatedDepth = it->get<int>();
        }

        if (const auto it = j.find("threads"); it != j.end() && it->is_number_integer()) {
            cfg.threads = it->get<int>();
        }
        if (const auto it = j.find("unsolvedPuzzles"); it != j.end() && it->is_array()) {
            cfg.unsolvedPuzzles.clear();
            cfg.unsolvedPuzzles.reserve(it->size());
            for (const auto& v : *it) {
                if (v.is_string()) {
                    cfg.unsolvedPuzzles.push_back(v.get<std::string>());
                }
            }
        }
        
    } catch (const std::exception& e) {
        tcout << "Failed to read config.json, using defaults: " << e.what() << '\n';
    }

    return cfg;
}

template<bool DEBUG>
static int runSolverForEstimatedDepth(
        BoardSolverFrontier& solver,
        int estimatedDepth,
        int threads,
        BoardSolverFrontier::SearchDirection searchDirection
) {

    switch (estimatedDepth) {
        case 3:
            solver.findSolutionsFrontierThreaded<1, 1, 1, DEBUG>(threads, searchDirection);
            return 0;

        case 4:
            solver.findSolutionsFrontierThreaded<1, 1, 2, DEBUG>(threads, searchDirection);
            return 0;

        case 5:
            solver.findSolutionsFrontierThreaded<1, 1, 3, DEBUG>(threads, searchDirection);
            return 0;

        case 6:
            solver.findSolutionsFrontierThreaded<1, 1, 4, DEBUG>(threads, searchDirection);
            return 0;

        case 7:
            solver.findSolutionsFrontierThreaded<1, 1, 5, DEBUG>(threads, searchDirection);
            return 0;

        case 8:
            solver.findSolutionsFrontierThreaded<1, 3, 4, DEBUG>(threads, searchDirection);
            return 0;

        case 9:
            solver.findSolutionsFrontierThreaded<1, 4, 4, DEBUG>(threads, searchDirection);
            return 0;

        case 10:
            solver.findSolutionsFrontierThreaded<1, 4, 5, DEBUG>(threads, searchDirection);
            return 0;

        case 11:
            solver.findSolutionsFrontierThreaded<1, 5, 5, DEBUG>(threads, searchDirection);
            return 0;

        case 12:
            // <2,5,5> [0:19:39.047]
            // <3,4,5> [0:21:31.331]
            // <4,3,5> [also too long]
            // <5,2,5> [way too long..., like 3 days]
            solver.findSolutionsFrontierThreaded<2, 5, 5, DEBUG>(threads, searchDirection);
            return 0;
            
        case 13:
            solver.findSolutionsFrontierThreaded<3, 5, 5, DEBUG>(threads, searchDirection);
            return 0;

        default:
            tcout << "Unsupported estimated depth: " << estimatedDepth << '\n';
            return -1;
    }
}

int runSingleFindSolution(const FindSolutionConfig& config) {
    const std::string outDirectory = config.outDirectory;
    const auto SEARCH_DIRECTION = config.searchDirection;
    const std::string puzzle = config.puzzle;
    const int estimatedDepth = config.estimatedDepth;
    const int threads = config.threads;
    
    const auto pair = BoardLookup::getBoardPair(puzzle);
    if (pair == nullptr) {
        tcout << "Unknown puzzle: " << puzzle << '\n';
        return -1;
    }

    tcout << pair->toStringNoColors() << "\n";

    BoardSolverFrontier solver(pair);
    solver.setWriteDirectory(outDirectory);
    tcout << "solver mode: exact-state exhaustive\n";
    tcout << "right frontier validation: full board state\n";
    tcout << "solution recovery: global exact-depth recovery\n";
    if (config.ignoredExperimentalSettings) {
        tcout << "warning: removed experimental config keys were ignored; "
                 "the exact-state exhaustive pipeline is always used\n";
    }

    if (config.debug) {
        return runSolverForEstimatedDepth<true>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
    }

    return runSolverForEstimatedDepth<false>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
}


int runMultiFindSolution(const FindSolutionConfig& config) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    
    const std::string outDirectory = config.outDirectory;
    const auto SEARCH_DIRECTION = config.searchDirection;
    int estimatedDepth = config.estimatedDepth;
    int threads = config.threads;
    
    int status = 0;
    for (auto puzzle : config.unsolvedPuzzles) {
        
        
        tcout << "Solving puzzle: " << puzzle << "\n";
        
        const auto pair = BoardLookup::getBoardPair(puzzle);
        if (pair == nullptr) {
            tcout << "Unknown puzzle: " << puzzle << '\n';
            --status;
            continue;
        }

        tcout << pair->toString() << "\n";
        
        BoardSolverFrontier solver(pair);
        solver.setWriteDirectory(outDirectory);

        if (config.debug) {
            status += runSolverForEstimatedDepth<true>(
                    solver, estimatedDepth, threads, SEARCH_DIRECTION);
        } else {
            status += runSolverForEstimatedDepth<false>(
                    solver, estimatedDepth, threads, SEARCH_DIRECTION);
        }
    }
    return status;
}



int main() {
    static constexpr u32 GB_NEEDED = 12;
    if (!hasAtLeastGBMemoryTotal(GB_NEEDED)) {
        tcout << "Program requires more RAM, exiting...\n";
        const u64 mem = getTotalSystemMemory();
        tcout << bytesFormatted(mem) << "/" << GB_NEEDED << ".000GB\n";
        return -1;
    }

    const FindSolutionConfig config = loadFindSolutionConfig();
    return runSingleFindSolution(config);
}
