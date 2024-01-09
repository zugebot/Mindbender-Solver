#pragma once

#include "MindbenderSolver/utils/processor.hpp"


#include <string>


struct Board {

    /**
     * uses the lower 54 bits
     * holds upper 3x6 cell grid
     */
    uint64_t b1 = 0;

    /**
     * uses the lower 54 bits
     * holds lower 3x6 cell grid
     */
    uint64_t b2 = 0;

    void setState(const uint8_t values[36]);



    ND uint32_t getScore1() const;
    ND uint32_t getScore2() const;




    ND uint64_t hash() const;
    ND std::string toString() const;
};

