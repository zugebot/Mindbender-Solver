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

int main() {
    static constexpr u32 GB_NEEDED = 12;
    if (!hasAtLeastGBMemoryTotal(GB_NEEDED)) {
        tcout << "Program requires more RAM, exiting...\n";
        const u64 mem = getTotalSystemMemory();
        tcout << bytesFormatted(mem) << "/" << GB_NEEDED << ".000GB" << std::endl;
        return -1;
    }

    const FindSolutionConfig config = loadFindSolutionConfig();

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


    /*
    Board board = pair->getSolutionState();
    std::vector<std::vector<HashMem>> boards_vec(6);
    Perms::reserveForDepth(board, boards_vec[0], 0);
    Perms::reserveForDepth(board, boards_vec[0], 1);
    Perms::reserveForDepth(board, boards_vec[0], 2);
    Perms::reserveForDepth(board, boards_vec[1], 3);
    Perms::reserveForDepth(board, boards_vec[2], 4);
    Perms::reserveForDepth(board, boards_vec[3], 5);

    std::vector<Sizes> sizes(25);
    int index = 0;
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            board.setFatXY(x, y);
            for (int i = 0; i < 6; i++) {
                Perms::getDepthFunc(board, boards_vec[i], i, true);
                tcout << x << " " << y << " Size " << i << ": " << boards_vec[i].size() << std::endl;
                sizes[index].sizes[i] = boards_vec[i].size();
                boards_vec[i].resize(0);
            }
            index++;
        }
    }

    for (int x = 0; x < 6; x++) {
        u64 lowest = 0;
        for (auto size : sizes) {
            if (size.sizes[x] > lowest) {
                lowest = size.sizes[x];
            }
        }
        tcout << "Size " << x << ": " << lowest << std::endl;

    }

    return 0;
    */
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
    /*
    static constexpr int DEPTH_TEST = 5;

    Timer timer1;
    auto boards1 = make2PermutationListFuncs[DEPTH_TEST](board1, 2);
    auto time1 = timer1.getSeconds();


    std::map<u64, Board> boardMap1;
    std::map<u64, u64> board1_B1;
    std::map<u64, u64> board1_B2;
    for (auto& board : boards1) {
        // boardMap1[board.hash] = board;
        board1_B1[board.b1] = 1;
        board1_B2[board.b2] = 1;
    }


    Timer timer2;
    auto boards2 = makePermutationListFuncs[DEPTH_TEST](board1, 2);
    auto time2 = timer2.getSeconds();


    std::map<u64, Board> boardMap2;
    std::map<u64, u64> board2_B1;
    std::map<u64, u64> board2_B2;
    for (auto& board : boards2) {
        // boardMap2[board.hash] = board;
        board2_B1[board.b1] = 1;
        board2_B2[board.b2] = 1;
    }




    tcout << "Size New: " << boards1.size() << "\n";
    tcout << "Size Old: " << boards2.size() << "\n";
    tcout << "\n";
    // tcout << "Uniq New: " << boardMap1.size() << "\n";
    // tcout << "Uniq Old: " << boardMap2.size() << "\n";
    // tcout << "\n";
    tcout << "__b1 New: " << board1_B1.size() << "\n";
    tcout << "__b2 New: " << board1_B2.size() << "\n";
    tcout << "\n";
    tcout << "__b1 Old: " << board2_B1.size() << "\n";
    tcout << "__b2 Old: " << board2_B2.size() << "\n";
    tcout << "\n";
    tcout << "Time New: " << time1 << "\n";
    tcout << "Time Old: " << time2 << "\n";
    tcout << std::flush;






    return 0;
    */
    /*
    vecBoard_t boards1;
    Permutations::getDepthFunc(board1, boards1, 4);

    vecBoard_t boards2;
    Permutations::allocateForDepth(boards2, 5);

    Timer timer;
    Permutations::getDepthPlus1Func(boards1, boards2, false);
    auto end = timer.getSeconds();

    tcout << "Time: " << end << "\n";
    tcout << "siz4: " << boards1.size() << "\n";
    tcout << "siz5: " << boards2.size() << "\n";

    return 0;
     */
}
