#include "MindbenderSolver/include.hpp"



template<int DEPTH>
struct RefState {
    B1B2 start;
    B1B2 end;
    int count = 0;

    RefState() = default;
};



static constexpr int DEPTH = 7;
static RefState<DEPTH> THE_STATE = RefState<DEPTH>();
static int states_traversed = 0;

template<int CUR_DEPTH, int MAX_DEPTH, bool ROW_TRUE>
void recursive_helper(B1B2 theBoard, int theNext) {
    ++states_traversed;

    if constexpr (CUR_DEPTH == MAX_DEPTH) {
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
            auto func = allActStructList[actIndex];

            func.action(nextBoard);
            if (nextBoard == theBoard) { continue; }



            if constexpr (CUR_DEPTH + 5 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<5>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 4 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<4>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 3 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<3>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 2 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<2>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<1>(THE_STATE.end)) { continue; }
            }




            recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, true>(
                    nextBoard, func.index + func.tillNext);
        }

        for (int actIndex = startCol; actIndex < 60; ++actIndex) {
            nextBoard = theBoard;
            auto func = allActStructList[actIndex];

            func.action(nextBoard);
            if (nextBoard == theBoard) { continue; }


            if constexpr (CUR_DEPTH + 5 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<5>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 4 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<4>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 3 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<3>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 2 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<2>(THE_STATE.end)) { continue; }
            }
            if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                if (nextBoard.getScore3Till<1>(THE_STATE.end)) { continue; }
            }





            recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, false>(
                    nextBoard, func.index + func.tillNext);
        }





    }
}


template<int MAX_DEPTH>
double recursive() {
    Timer timer;
    recursive_helper<0, MAX_DEPTH, true>(THE_STATE.start, 0);
    return timer.getSeconds();
}


int main() {
    // C BoardPair* pair = BoardLookup::getBoardPair("4-3");
    // C Board board = pair->getStartState();
    // C Board solve = pair->getEndState();

    C Board board = BoardLookup::getBoardPair("13-1")->getStartState();
    Board solve = board;
    // std::cout << "0: " << board.getScore3(solve) << "\n";
    R_4_1(solve); // std::cout << "1: " << board.getScore3(solve) << "\n";
    C_5_5(solve); // std::cout << "2: " << board.getScore3(solve) << "\n";
    R_2_2(solve); // std::cout << "3: " << board.getScore3(solve) << "\n";
    R_1_5(solve); // std::cout << "4: " << board.getScore3(solve) << "\n";
    C_3_4(solve); // std::cout << "5: " << board.getScore3(solve) << "\n";
    R_2_2(solve); // std::cout << "6: " << board.getScore3(solve) << "\n";
    R_4_4(solve); // std::cout << "7: " << board.getScore3(solve) << "\n";
    // std::cout << std::flush;

    THE_STATE.start = static_cast<B1B2>(board);
    THE_STATE.end = static_cast<B1B2>(solve);


    JVec<Board> boards;
    Perms<Board>::reserveForDepth(board, boards, 2);
    Perms<Board>::toDepthFromLeft::funcPtrs[2](board, boards, board.getHashFunc());
    std::cout << "Length: " << boards.size() << std::endl;


    double time = recursive<DEPTH>();

    std::cout << "Time: " << time << std::endl;
    std::cout << "Solves: " << THE_STATE.count << std::endl;
    std::cout << "Traversed: " << states_traversed << std::endl;
    std::cout << "GetScore3: " << GET_SCORE_3_CALLS << std::endl;

    return 0;
}