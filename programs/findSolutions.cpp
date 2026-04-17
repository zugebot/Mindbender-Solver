// programs/findSolutions.cpp
#include "code/include.hpp"
#include "code/solver/memory_perm_gen.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "code/solver/solver_direct.hpp"
#include "code/solver/solver_frontier.hpp"
#include "include/nlohmann/json.hpp"
#include "utils/get_free_memory.hpp"






struct FindSolutionConfig {
    std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    BoardSolverFrontier::SearchDirection searchDirection = BoardSolverFrontier::SearchDirection::Auto;
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

static FindSolutionConfig loadFindSolutionConfig() {
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
    std::string puzzle = config.puzzle;
    int estimatedDepth = config.estimatedDepth;
    int threads = config.threads;
    
    const auto pair = BoardLookup::getBoardPair(puzzle);

    tcout << pair->toString() << std::endl;

    BoardSolverFrontier solver(pair);
    solver.setWriteDirectory(outDirectory);

    if (config.debug) {
        return runSolverForEstimatedDepth<true>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
    }

    return runSolverForEstimatedDepth<false>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
}


int runMultiFindSolution(const FindSolutionConfig& config) {
    const std::string outDirectory = config.outDirectory;
    const auto SEARCH_DIRECTION = config.searchDirection;
    int estimatedDepth = config.estimatedDepth;
    int threads = config.threads;
    
    int status = 0;
    for (auto puzzle : config.unsolvedPuzzles) {
        
        
        tcout << "Solving puzzle: " << puzzle << std::endl;
        
        const auto pair = BoardLookup::getBoardPair(puzzle);
        if (pair->getStartState().getFatBool()) {
            continue;
        }
    
        tcout << pair->toString() << std::endl;
        
        BoardSolverFrontier solver(pair);
        solver.setWriteDirectory(outDirectory);

        if (config.debug) {
            status += runSolverForEstimatedDepth<true>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
        }
        status += runSolverForEstimatedDepth<false>(solver, estimatedDepth, threads, SEARCH_DIRECTION);
    }
    return status;
}



int main() {
    static constexpr u32 GB_NEEDED = 12;
    if (!hasAtLeastGBMemoryTotal(GB_NEEDED)) {
        tcout << "Program requires more RAM, exiting...\n";
        const u64 mem = getTotalSystemMemory();
        tcout << bytesFormatted(mem) << "/" << GB_NEEDED << ".000GB" << std::endl;
        return -1;
    }

    const FindSolutionConfig config = loadFindSolutionConfig();

    return runSingleFindSolution(config);
    // return runMultiFindSolution(config);
  
    
    
    
    
    
    
    /*
    namespace std {
        template <>
        struct hash<Board> {
            std::size_t operator()(const Board& b) const {
                return b.getHash();
            }
        };
    }
    void func(Board& board, Action action, std::vector<u8>& indexes) {
        u8* list = fatActionsIndexes[board.getFatXY()];
        indexes.push_back(std::find(list, list + 48, getIndexFromAction(action)) - list);
        action(board);
    }
    std::vector<HashMem> boards_out(48);
    const Board board = pair->getInitialState();
    const auto hasher = HashMem::getHashFunc(board);
    make_fat_perm_list<1>(board, boards_out, hasher);

    tcout << board.toString() << std::endl;

    int index = 10;
    int funcIndex = fatActionsIndexes[board.getFatXY()][index];

    Board temp = board;

    auto function = allActionsList[funcIndex];


    tcout << "is fat: " << temp.getFatBool() << "\n";
    function(temp);
    tcout << "is fat: " << temp.getFatBool() << "\n";

    temp.getMemory().setNextNMove<1>(funcIndex);
    // applyMoves(temp, boards_out[index]);
    tcout << temp.toString() << std::endl;

    tcout << boards_out[index].getMemory().asmFatStringForwards(board.getFatXY());


    return 0;
    */
    /*
    c_Board board = pair->getInitialState();
    c_auto hasher = board.getHashFunc();

    static constexpr u64 depth = 2;
    c_u64 sizeOut = 170000000; // 2948970, 170000000

    std::vector<Board> boards_new;
    boards_new.reserve(sizeOut);
    std::vector<Board> boards_old(sizeOut);


    Timer timer_new;
    // make_fat_perm_list<depth>(board, boards_new, hasher);
    make_fat_perm_list<depth>(board, boards_new, hasher);
    tcout << "New: " << timer_new.getSeconds() << " size: " << boards_new.size() << std::endl;


    Timer timer_old;
    Perms::getDepthFunc(board, boards_old, depth, true);
    tcout << "Old: " << timer_old.getSeconds() << " size: " << boards_old.size() << std::endl;


    MU volatile int x = 0;
    */
    /*
    BoardSolver solver(pair);
    solver.setWriteDirectory(outDirectory);
    solver.setDepthParams(6, 10, 11);
    solver.preAllocateMemory(6);

    tcout << pair->toString() << std::endl;
    solver.findSolutions<true>();
    return 0;
    */
    /*
    tcout << board1.toString() << std::endl;

    bool intersection[5];

    intersection[0] = doActISColMatch(board1, 4, 3, 1, 1);
    intersection[1] = doActISColMatch(board1, 4, 3, 1, 2);
    intersection[2] = doActISColMatch(board1, 4, 3, 1, 3);
    intersection[3] = doActISColMatch(board1, 4, 3, 1, 4);
    intersection[4] = doActISColMatch(board1, 4, 3, 1, 5);

    for (int i = 0; i < 5; i++) {
        std::string valStr = intersection[i] ? "true" : "false";
        tcout << "3 Colors at [" << i << "]: " << valStr << "\n";
    }

    tcout << "\n";

    auto intersection2 = doActISColMatchBatched(board1, 4, 3, 1);

    for (int i = 0; i < 5; i++) {
        u8 mask = 1 << i;
        bool val = intersection2 & mask;
        std::string valStr = val ? "true" : "false";
        tcout << "3 Colors at [" << i << "]: " << valStr << std::endl;
    }




    return 0;
    */
}
