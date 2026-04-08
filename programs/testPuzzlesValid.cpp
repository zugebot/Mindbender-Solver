#include "code/include.hpp"

#include <cstdlib>

bool is_number(const std::string& s) {
    auto it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}


void clear_cmd() {
    system("cls");
}



int main() {

    const std::string PREV_SYM = ";";
    const std::string NEXT_SYM = "'";

    std::vector<std::string> puzzleStrs;
    for (int world = 1; world <= 20; world++) {
        for (int level = 1; level <= 5; level++) {
            puzzleStrs.push_back(std::to_string(world) + "-" + std::to_string(level));
        }
    }

    int last_index = 0;
    const BoardPair* boardPair = nullptr;
    while (true) {
        std::string input;
        tcout << "puzzle: ";
        std::cin >> input;

        if (input == "exit") {
            break;
        }

        if (input == PREV_SYM || input == NEXT_SYM) {
            if (input == PREV_SYM) { --last_index; }
            if (input == NEXT_SYM) { ++last_index; }
            if (last_index < 0) { last_index = 99; }
            if (last_index > 99) { last_index = 0; }
            boardPair = BoardLookup::getBoardPair(puzzleStrs[last_index]);
        } else if (is_number(input)) {
            const int num = atoi(input.c_str());
            if (num >= 0 && num <= 99) {
                boardPair = BoardLookup::getBoardPair(puzzleStrs[num]);
            }
            last_index = num;
        } else {
            boardPair = BoardLookup::getBoardPair(input);
            for (int i = 0; i < 100; i++) {
                if (puzzleStrs[i] == input) {
                    last_index = i;
                    break;
                }
            }
        }



        if (boardPair == nullptr) {
            continue;
        }

        clear_cmd();
        tcout << "\n";
        tcout << "Name  : " << boardPair->getName() << "\n";
        tcout << "Colors: " << boardPair->getStartState().getColorCount() << "\n";
        tcout << "Fat   : ";
        if (boardPair->getStartState().getFatBool()) {
            tcout << "True\n";
        } else {
            tcout << "False\n";
        }
        tcout << "\n";
        tcout << boardPair->toStringReversed() << "\n";
        tcout << std::flush;
    }

    return 0;
}