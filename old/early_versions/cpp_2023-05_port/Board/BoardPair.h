# pragma once

#include <vector>

#include "board.h"
#include "boardArray.h"


class BoardPair {
private:
    Board key;
    std::vector<Board> values;

public:
    BoardPair() = default;
    std::vector<Board>::iterator begin();
    std::vector<Board>::iterator end();

    void setKey(Board board);
    void add(Board board);



};
