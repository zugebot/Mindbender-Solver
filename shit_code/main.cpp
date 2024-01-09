// Jerrin Shirks

#include "Board/Board.h"
#include "Board/BoardOperations.h"
#include "Board/Levels.h"
#include "support/colors.h"
#include "support/time.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

typedef unsigned char byte;


class Main {
public:
    static Levels levels;

    static const std::string RESET;
    static const std::string GREEN;
    static const std::string MAGENTA;
    static const std::string CYAN;
    static const std::string RED;

    static const int MAXIMUM_DEPTH;
    static const int MAXIMUM_SIZE;
    static const int DEBUG;
};

Levels Main::levels;

const std::string Main::RESET = Colors::Reset;
const std::string Main::GREEN = Colors::Green;
const std::string Main::MAGENTA = Colors::Magenta;
const std::string Main::CYAN = Colors::Cyan;
const std::string Main::RED = Colors::Red;

const int Main::DEBUG = true;
const int Main::MAXIMUM_DEPTH = 20;
const int Main::MAXIMUM_SIZE = 300000;

int main() {

    // starting details
    std::string level_name = "4-3";

    Board board = Main::levels.b4_3;
    Board solve = Main::levels.s4_3;

    BoardArray boards = BoardArray(0);
    BoardArray solves = BoardArray(0);
    board.print("Boards");
    solve.print("Solve");
    // starting states
    int depth = 0;
    BoardOperations::myMaximumSize = Main::MAXIMUM_SIZE;
    BoardOperations::myDebug = Main::DEBUG;
    clock_t start = clock();
    std::vector<std::array<Board, 2>> pairs;

    // extends each one, alternating
    while (depth < Main::MAXIMUM_DEPTH) {
        depth++;

        std::string side = (depth % 2) ? "solves" : "boards";

        if (depth > 2) {
            std::cout << Main::MAGENTA << "\n[" << depth << "]"
                 << Main::RESET << " Attempting search [" << side << "]" << std::endl;
        }

        if (depth == 1) { // if it is the first move
            std::cout << Main::MAGENTA << "[1]" << Main::RESET << " Generating boards." << std::endl;
            boards = board.possibleBoards();

        } else if (depth == 2) { // if it is the second move
            std::cout << Main::MAGENTA << "[2]" << Main::RESET << " Generating Solves." << std::endl;
            solves = solve.possibleBoards();

        } else if ((depth % 2) == 0) {
            boards = BoardOperations::extend(boards, solve, start);

        } else {
            solves = BoardOperations::extend(solves, board, start);
        }

        // finds matches, or "pairs"
        pairs = BoardOperations::find_intersection(boards, solves);

        // nothing found
        if (pairs.empty()) {
            std::cout << Main::GREEN << "   [--------]"
                 << Main::RED << " no solutions found" << Main::RESET << std::endl;
            continue;

            // if solutions found
        }

        std::cout << "\n" << Colors::CyanBold << "**End Details**" << Main::RESET << std::endl;
        std::cout << "Puzzle Title  : " << Main::CYAN << level_name << Main::RESET << std::endl;
        std::cout << "Moves To Solve: " << Main::CYAN << depth << Main::RESET << std::endl;
        std::cout << "Solution Count: " << Main::CYAN<< pairs.size() << Main::RESET << std::endl;
        std::cout << "Solution Time : " << Main::CYAN << Time::time(start) << "s" << Main::RESET << std::endl;

        std::string path = R"(C:\Users\jerrin\CLionProjects\Chuzzle Solver C++\levels\)";
        std::string filename = path + level_name + "_c" + std::to_string(depth) + ".txt";
        std::ofstream out(filename);

        // writes all the solutions to a file
        for (size_t i = 0; i < pairs.size(); i++) {
            out << BoardOperations::getMovesetString(pairs[i]);
            if (i != pairs.size() - 1) {
                out << std::endl;
            }
        }
        out.flush();
        out.close();
        return 0;

    } // end of while loop

} // end of main