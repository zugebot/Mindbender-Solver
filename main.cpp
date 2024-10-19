#include "MindbenderSolver/include.hpp"

#include <iostream>


// TODO: For Fats: make it choose all rows below last_row if its from the right

struct Sizes {
    std::vector<u64> sizes = std::vector<u64>(6);
};


int main() {
    const std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver\MindbenderSolver)";
    const auto pair = BoardLookup::getBoardPair("5-3");

    std::cout << pair->toString() << std::endl;

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
                std::cout << x << " " << y << " Size " << i << ": " << boards_vec[i].size() << std::endl;
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
        std::cout << "Size " << x << ": " << lowest << std::endl;

    }

    return 0;
    */

    BoardSolver solver(pair);
    solver.setWriteDirectory(outDirectory);
    solver.setDepthParams(5, 10, 10);

    const Timer allocateTimer;
    solver.preAllocateMemory(5);
    std::cout << "Alloc Time: " << allocateTimer.getSeconds() << std::endl;

    solver.findSolutions<false>();

    std::cout << "Calls: " << MAKE_FAT_PERM_LIST_HELPER_CALLS << std::endl;
    std::cout << "Checks: " << MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS << std::endl;
    std::cout << "Similar: " << MAKE_FAT_PERM_LIST_HELPER_FOUND_SIMILAR << std::endl;

    return 0;

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

    std::cout << board.toString() << std::endl;

    int index = 10;
    int funcIndex = fatActionsIndexes[board.getFatXY()][index];

    Board temp = board;

    auto function = allActionsList[funcIndex];


    std::cout << "is fat: " << temp.getFatBool() << "\n";
    function(temp);
    std::cout << "is fat: " << temp.getFatBool() << "\n";

    temp.getMemory().setNextNMove<1>(funcIndex);
    // applyMoves(temp, boards_out[index]);
    std::cout << temp.toString() << std::endl;

    std::cout << boards_out[index].getMemory().asmFatStringForwards(board.getFatXY());


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
    std::cout << "New: " << timer_new.getSeconds() << " size: " << boards_new.size() << std::endl;


    Timer timer_old;
    Perms::getDepthFunc(board, boards_old, depth, true);
    std::cout << "Old: " << timer_old.getSeconds() << " size: " << boards_old.size() << std::endl;


    MU volatile int x = 0;
    */
    /*
    BoardSolver solver(pair);
    solver.setWriteDirectory(outDirectory);
    solver.setDepthParams(6, 10, 11);
    solver.preAllocateMemory(6);

    std::cout << pair->toString() << std::endl;
    solver.findSolutions<true>();
    return 0;
    */
    /*
    std::cout << board1.toString() << std::endl;

    bool intersection[5];

    intersection[0] = doActISColMatch(board1, 4, 3, 1, 1);
    intersection[1] = doActISColMatch(board1, 4, 3, 1, 2);
    intersection[2] = doActISColMatch(board1, 4, 3, 1, 3);
    intersection[3] = doActISColMatch(board1, 4, 3, 1, 4);
    intersection[4] = doActISColMatch(board1, 4, 3, 1, 5);

    for (int i = 0; i < 5; i++) {
        std::string valStr = intersection[i] ? "true" : "false";
        std::cout << "3 Colors at [" << i << "]: " << valStr << "\n";
    }

    std::cout << "\n";

    auto intersection2 = doActISColMatchBatched(board1, 4, 3, 1);

    for (int i = 0; i < 5; i++) {
        u8 mask = 1 << i;
        bool val = intersection2 & mask;
        std::string valStr = val ? "true" : "false";
        std::cout << "3 Colors at [" << i << "]: " << valStr << std::endl;
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




    std::cout << "Size New: " << boards1.size() << "\n";
    std::cout << "Size Old: " << boards2.size() << "\n";
    std::cout << "\n";
    // std::cout << "Uniq New: " << boardMap1.size() << "\n";
    // std::cout << "Uniq Old: " << boardMap2.size() << "\n";
    // std::cout << "\n";
    std::cout << "__b1 New: " << board1_B1.size() << "\n";
    std::cout << "__b2 New: " << board1_B2.size() << "\n";
    std::cout << "\n";
    std::cout << "__b1 Old: " << board2_B1.size() << "\n";
    std::cout << "__b2 Old: " << board2_B2.size() << "\n";
    std::cout << "\n";
    std::cout << "Time New: " << time1 << "\n";
    std::cout << "Time Old: " << time2 << "\n";
    std::cout << std::flush;






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

    std::cout << "Time: " << end << "\n";
    std::cout << "siz4: " << boards1.size() << "\n";
    std::cout << "siz5: " << boards2.size() << "\n";

    return 0;
     */
}
