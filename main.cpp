#include "MindbenderSolver/include.hpp"


/*
MUND static u64 getRow(const Board* board, c_u64 y);
MUND static u64 getCol(const Board* board, c_u32 x);
MUND static u64 constructMapCenter(c_u64 row, c_u32 x);
MUND static u64 getScore1ShiftComp(c_u64 sect, c_u64 mapCent);
static void shiftLeft(u64& sect, c_u32 index);
MUND u64 getRowColIntersections(u32 x, u32 y) const;
 */


/*
Metrics I need to time:
- whether a lookup table for {y: 0 <= y <= 5} is faster for:
    - if y % 3
    - if y * 3
    - any other expressions of y
*/


u32 getIndex(u32 x, u32 y) {
    x -= 1;
    y -= 1;
    return 5 * ((x - y) % 5) + x;
}


int main() {
    const std::string outDirectory = R"(C:\Users\jerrin\CLionProjects\Mindbender-Solver)";
    const auto pair = BoardLookup::getBoardPair("6-5");
    // const Board board = pair->getInitialState();
    Board board;
    c_u8 state[36] = {
        1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    };
    board.setState(state);
    std::cout << board.toString() << std::endl;

    MU c_u64 inters_a = board.getRowColIntersections(0, 1);

    volatile int end = 0;




    /*
    // initialize solver
    BoardSolver solver(pair);
    solver.setWriteDirectory(outDirectory);
    solver.setDepthParams(5, 10, 10);
    solver.preAllocateMemory();

    std::cout << pair->toString() << std::endl;



    solver.findSolutions<true>();
    return 0;
    */

}
