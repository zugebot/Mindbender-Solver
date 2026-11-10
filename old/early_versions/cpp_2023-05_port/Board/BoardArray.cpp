#include "BoardArray.h"
#include "Board.h"

BoardArray::BoardArray(int size) : _size(size), _array(new Board[size]) {}

BoardArray::~BoardArray() = default;

int BoardArray::size() const {
    return _size;
}

Board* BoardArray::begin() {
    return _array;
}

Board* BoardArray::end() {
    return _array + _size;
}

const Board* BoardArray::begin() const {
    return _array;
}

const Board* BoardArray::end() const {
    return _array + _size;
}

void BoardArray::resize(int newSize) {
    if (newSize < 0) {
        throw std::invalid_argument("New size must be non-negative");
    }

    // If the size hasn't changed, there's nothing to do
    if (newSize == _size) {
        return;
    }

    auto* newArray = new Board[newSize];

    if (newSize < _size) {
        // If the new size is smaller, copy only the elements that fit
        std::copy(_array, _array + newSize, newArray);
    } else {
        // If the new size is larger, copy all existing elements and initialize the rest
        std::copy(_array, _array + _size, newArray);
        std::fill(newArray + _size, newArray + newSize, Board());
    }

    // Deallocate the old array and update the pointer and size
    delete[] _array;
    _array = newArray;
    _size = newSize;
}
