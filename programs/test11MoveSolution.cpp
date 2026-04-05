#include "code/include.hpp"


#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>

#include "include/ghc/fs_std.hpp"


namespace std {
    template <>
    struct hash<Board> {
        std::size_t operator()(const Board& b) const {
            return b.getHash();
        }
    };
}


std::vector<std::string> getFilesInDir(const std::string& path) {
    std::vector<std::string> file_list;
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (fs::is_regular_file(entry.status())) {
                file_list.push_back(entry.path().string());  // Collect file paths
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
        return file_list;
    }
    return file_list;
}


static constexpr u64 BUFFER_SIZE = 33'554'432;


template<eSequenceDir SECT_DIR>
std::vector<Board> create5Depth(Board board) {
    std::vector<Board> boards_initial_side_5;
    Perms<Board>::reserveForDepth<SECT_DIR>(board, boards_initial_side_5, 5);
    Perms<Board>::getDepthFunc<SECT_DIR>(board, boards_initial_side_5, 5, true);
    std::vector<Board> aux_buffer(boards_initial_side_5.size());
    radix_sort<5, 12>(boards_initial_side_5, aux_buffer);
    return boards_initial_side_5;
}


void create6Depth(const Board& board, std::vector<Board>& boards_buffer, const std::string& path) {
    auto boards5 = create5Depth(board);

    Timer timer;
    Perms<Board>::getDepthPlus1BufferedFunc(path, boards5, boards_buffer, 5);
    tcout << timer.getSeconds() << std::endl;
}




int main() {

    const std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    const auto pair = BoardLookup::getBoardPair("7-1");

    tcout << pair->toString() << std::endl;
    Board board = pair->getStartState();
    Board solve = pair->getEndState();


    std::vector<Board> boards_buffer(BUFFER_SIZE);
    C u32 board_depth = 6;
    C u32 solve_depth = 5;

    const std::string path = "E:\\" + pair->getName() + "_b\\";
    if (fs::create_directory(path)) {
        tcout << "Created directory: " << path << std::endl;
        create6Depth(board, boards_buffer, path);
    } else {
        tcout << "Failed to create directory or it already exists." << std::endl;
    }
    auto boards5 = create5Depth(solve);


    std::set<std::string> resultSet;
    std::vector<Board> boards6(BUFFER_SIZE);

    auto files = getFilesInDir(path);

    int index = 0;
    Timer timer;
    for (const auto& file_path : files) {

        tcout << "Opening: \"" << file_path << "\"";
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        tcout << " | Reading";
        boards6.resize(fileSize / 32);
        file.read(reinterpret_cast<char *>(boards6.data()), fileSize);
        file.close();

        tcout << " | Sorting";
        radix_sort<5, 12>(boards6, boards_buffer);

        tcout << " | Intersect";
        auto results = intersection(boards6, boards5);

        for (const auto [fst, snd]: results) {
            std::string moveset = fst->hashMem.getMemory().asmString(&snd->hashMem.getMemory());
            resultSet.insert(moveset);
        }
        tcout << " | Results: " << results.size() << "\n";

        index++;
    }


    if (!resultSet.empty()) {
        const std::string filename = pair->getName()
                                     + "_c" + std::to_string(11)
                                     + "_" + std::to_string(resultSet.size())
                                     + "_" + std::to_string(board_depth)
                                     + "_" + std::to_string(solve_depth)
                                     + ".txt";
        tcout << "Saving results to '" << filename << "'.\n";
        std::ofstream outfile(outDirectory + "\\levels\\" + filename);
        for (const auto& str: resultSet) {
            outfile << str << std::endl;
        }
        outfile.close();
    } else {
        tcout << "No solutions found...\n";
    }
    tcout << "Total Time: " << timer.getSeconds() << std::endl;







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
