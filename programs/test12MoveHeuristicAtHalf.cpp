#include "code/include.hpp"

#include <cmath>
#include <set>


// 9-2 at depth=5 should be 168896630
template<u32 CLASS_MAX_DEPTH>
class BoardDFS {
public:
    union {
        B1B2 myStart;
        B1B2 myArray[CLASS_MAX_DEPTH + 1];
    };
    B1B2 myEnd;

    // statistics
    u64 myCount;
    u64 myStatesTraversed;
    u64 myDepthsCount[CLASS_MAX_DEPTH + 1] = {};


    BoardDFS(const Board& board, const Board& solve) :
    myStart(static_cast<B1B2>(board)),
    myEnd(static_cast<B1B2>(solve)),
    myCount(0),
    myStatesTraversed(0) {}


    double recursive() {
        const Timer timer;
        recursive_helper<0, true>(0);
        return timer.getSeconds();
    }


private:
    template<u32 CUR_DEPTH, bool ROW_TRUE>
    void recursive_helper(const int theNext) {
        ++myDepthsCount[CUR_DEPTH];
        ++myStatesTraversed;

        const B1B2 * stack_prev = &myArray[CUR_DEPTH];
        B1B2 * stack_next = &myArray[CUR_DEPTH + 1];
        // static constexpr int DIFF_DEPTH = MAX_DEPTH - CUR_DEPTH;

        if constexpr (CUR_DEPTH == CLASS_MAX_DEPTH) {
            /* const i32 heur = theBoard.getScore3(THE_STATE.end);
            if (heur < lowestHueuristic) {
                lowestHueuristic = heur;
                worstBoard = theBoard; } */
            if EXPECT_FALSE(*stack_prev == myEnd) { ++myCount; }

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

            for (int actIndex = startRow; actIndex < 30; ++actIndex) {
                *stack_next = *stack_prev;
                // allActStructList[actIndex]. action(*stack_next);
                applyMove(*stack_next, actIndex);
                if (*stack_next == *stack_prev) { continue; }
                /* if constexpr (DIFF_DEPTH > 0 && DIFF_DEPTH < 6) {
                    if (nextBoard.getScore3Till<DIFF_DEPTH>(THE_STATE.end)) { continue; } } */
                recursive_helper<CUR_DEPTH + 1, true>(
                    actIndex + allActStructList[actIndex].tillNext);
            }

            for (int actIndex = startCol; actIndex < 60; ++actIndex) {
                *stack_next = *stack_prev;
                // allActStructList[actIndex]. action(*stack_next);
                applyMove(*stack_next, actIndex);
                if (*stack_next == *stack_prev) { continue; }
                /* if constexpr (DIFF_DEPTH > 0 && DIFF_DEPTH < 6) {
                    if (nextBoard.getScore3Till<DIFF_DEPTH>(THE_STATE.end)) { continue; } } */
                recursive_helper<CUR_DEPTH + 1, false>(
                    actIndex + allActStructList[actIndex].tillNext);
            }
        }
    }



};



int main() {
    const Board board = BoardLookup::getBoardPair("9-2")->getStartState();
    const Board solve = BoardLookup::getBoardPair("9-2")->getEndState();

    tcout << BoardLookup::getBoardPair("9-2")->toString() << std::endl;

    static constexpr u32 DEPTH = 5;
    BoardDFS<DEPTH> dfs(board, solve);
    const double time = dfs.recursive();

    tcout << "Time: " << time << std::endl;
    tcout << "Depth: " << DEPTH << std::endl;
    tcout << "Solves: " << dfs.myCount << std::endl;
    tcout << "Traversed: " << dfs.myStatesTraversed << std::endl;
    tcout << "Total States: " << pow(60, DEPTH) << std::endl;
    tcout << "GetScore3: " << GET_SCORE_3_CALLS << std::endl;



    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_pool(num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_pool[t] = std::thread([&, t]() {

        });
    }
    for (auto& thread : thread_pool) thread.join();














    tcout << "Depths: [";
    for (int i = 0; i < DEPTH + 1; ++i) {
        tcout << dfs.myDepthsCount[i];
        if (i != DEPTH) {  tcout << ", "; }
    }
    tcout << "]\n";

    return 0;
}