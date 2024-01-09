#pragma once

#include <cstdint>



struct Board {

    /// uses the lower 54 bits
    uint64_t b1 = 0;
    /// uses the lower 54 bits
    uint64_t b2 = 0;


    void setBoard(const uint8_t values[36]);
    void printState() const;
};

