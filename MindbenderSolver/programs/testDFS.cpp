#include "MindbenderSolver/include.hpp"

#include <cmath>
#include <set>


template<int DEPTH>
struct RefState {
    B1B2 start;
    B1B2 end;
    int count = 0;

    RefState() = default;
};





static constexpr int DEPTH = 9;
static int DEPTHS_COUNT[DEPTH + 1] = {};
static auto THE_STATE = RefState<DEPTH>();
static int states_traversed = 0;

template<int CUR_DEPTH, int MAX_DEPTH, bool ROW_TRUE>
void recursive_helper(C B1B2 theBoard, C int theNext) {
    ++DEPTHS_COUNT[CUR_DEPTH];
    ++states_traversed;

    static constexpr int DIFF_DEPTH = MAX_DEPTH - CUR_DEPTH;

    if constexpr (CUR_DEPTH == MAX_DEPTH) {
        // std::cout << theBoard.getScore3(THE_STATE.end) << ", ";
        // should be a mix of 0's and 1's.
        if EXPECT_FALSE(theBoard == THE_STATE.end) {
            THE_STATE.count++;
        }

    } else {

        // set boundaries
        int startRow, startCol;
        if constexpr (ROW_TRUE) {
            startRow = theNext;
            startCol = 30;
        } else {
            startRow = 0;
            startCol = theNext;
        }

        B1B2 nextBoard;

        // Loop for normal row and modified column
        for (int actIndex = startRow; actIndex < 30; ++actIndex) {
            nextBoard = theBoard;
            C auto func = allActStructList[actIndex];

            func.action(nextBoard);
            if (nextBoard == theBoard) { continue; }

            if constexpr (DIFF_DEPTH > 0 && DIFF_DEPTH < 6) {
                if (nextBoard.getScore3Till<DIFF_DEPTH>(THE_STATE.end)) { continue; } }

            recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, true>(
                    nextBoard, func.index + func.tillNext);
        }

        for (int actIndex = startCol; actIndex < 60; ++actIndex) {
            nextBoard = theBoard;
            C auto func = allActStructList[actIndex];

            func.action(nextBoard);
            if (nextBoard == theBoard) { continue; }


            if constexpr (DIFF_DEPTH > 0 && DIFF_DEPTH < 6) {
                if (nextBoard.getScore3Till<DIFF_DEPTH>(THE_STATE.end)) { continue; } }


            recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, false>(
                    nextBoard, func.index + func.tillNext);
        }
    }
}


template<int MAX_DEPTH>
double recursive() {
    C Timer timer;
    recursive_helper<0, MAX_DEPTH, true>(THE_STATE.start, 0);
    return timer.getSeconds();
}


int main() {
    C Board board = BoardLookup::getBoardPair("13-1")->getStartState();
    Board solve = board;

    R_4_1(solve); // 0
    C_5_5(solve); // 1
    R_2_2(solve); // 2
    R_1_5(solve); // 3
    C_3_4(solve); // 4
    R_2_2(solve); // 5
    R_4_4(solve); // 6
    C_3_4(solve); // 7
    R_1_1(solve); // 8

    THE_STATE.start = static_cast<B1B2>(board);
    THE_STATE.end = static_cast<B1B2>(solve);


    JVec<Board> boards;
    Perms<Board>::reserveForDepth(board, boards, 5);
    Perms<Board>::toDepthFromLeft::funcPtrs[5](board, boards, board.getHashFunc());
    std::cout << "[Arr] Length: " << boards.size() << std::endl;
    std::set<Board> boardSet;
    for (int i = 0; i < boards.size(); i++) {
        Board bi = boards[i];
        boardSet.insert(bi);
    }
    std::cout << "[Set] Length: " << boardSet.size() << std::endl;




    C double time = recursive<DEPTH>();

    std::cout << "Time: " << time << std::endl;
    std::cout << "Depth: " << DEPTH << std::endl;
    std::cout << "Solves: " << THE_STATE.count << std::endl;
    std::cout << "Traversed: " << states_traversed << std::endl;
    std::cout << "Total States: " << pow(60, DEPTH) << std::endl;
    std::cout << "GetScore3: " << GET_SCORE_3_CALLS << std::endl;

    std::cout << "Depths: [";
    for (int i = 0; i < DEPTH + 1; ++i) {
        std::cout << DEPTHS_COUNT[i];
        if (i != DEPTH) {  std::cout << ", "; }
    }
    std::cout << "]\n";

    return 0;
}