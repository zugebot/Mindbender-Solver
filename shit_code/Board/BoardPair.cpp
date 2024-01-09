#include "BoardPair.h"

std::vector<Board>::iterator BoardPair::begin() {
    return values.begin();
}

std::vector<Board>::iterator BoardPair::end() {
    return values.end();
}


void BoardPair::setKey(Board board) {
    key = board;
}

void BoardPair::add(Board board) {
    values.push_back(board);
}
