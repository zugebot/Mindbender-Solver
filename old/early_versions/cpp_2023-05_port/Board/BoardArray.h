#pragma once

#include <stdexcept>
#include <iterator>

// class Board;
#include "Board.h"

class BoardArray {
private:
    int _size;

public:
    Board* _array;

    explicit BoardArray(int size);
    ~BoardArray();
    [[nodiscard]] int size() const;
    Board* begin();
    Board* end();
    [[nodiscard]] const Board* begin() const;
    [[nodiscard]] const Board* end() const;
    void resize(int newSize);
    template<typename It>
    BoardArray(It begin, It end) {
        _size = std::distance(begin, end);
        _array = new Board[_size]();
        std::copy(begin, end, _array);
    }
};