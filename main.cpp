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
    const auto pair = BoardLookup::getBoardPair("4-4");
    // const Board board = pair->getInitialState();

    // std::cout << "Score: " << pair->getInitialState().getScore1(pair->getSolutionState()) << std::endl;

    Board board;
    c_u8 state[36] = {
        1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 3,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 4,
    };
    board.setState(state);
    // std::cout << board.toString() << std::endl;

    MU c_u64 inters_a = board.getRowColIntersections(5, 0);

    /*
    volatile int v1 = 10;
    volatile int v2 = 0'010;


    volatile int end = 0;

    Board bbb;
    bbb.setFat(0, 0);

    for (int x = 0; x <= 4; x++) {
        for (int amnt = 1; amnt <= 5; amnt++) {
            if (x + amnt == 5) {
                std::cout<<"#########\n";
                continue;
            }
            bbb.setFatX(x);
            bbb.addFatX(amnt);
            c_int moved = bbb.getFatX();

            c_int mod = (x + amnt) % 6;

            std::cout<<"("<<x<<", "<<amnt<<"): "<<mod<<", got "<<moved<<"\n";
        }
        std::cout<<"\n";
    }
    std::cout<<std::flush;


    return 0;
*/

    // initialize solver
    BoardSolver solver(pair);
    solver.setWriteDirectory(outDirectory);
    solver.setDepthParams(4, 7, 7);
    solver.preAllocateMemory();

    std::cout << pair->toString() << std::endl;



    solver.findSolutions<true, false>();

    // int ret;
    // std::cin >> ret;

    return 0;


}
