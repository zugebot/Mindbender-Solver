#pragma once

#include <string>
#include "MindbenderSolver/utils/processor.hpp"
#include "memory.hpp"




class Board {
public:

    /**
     *  3 bits: fat x position
     *  3 bits: unused
     *  1 bit : 1 if has a fat, otherwise 0
     *  3 bits: (# of colors - 1)
     * 54 bits: holds upper 3x6 cell grid
     */
    u64 b1 = 0;
    /**
     *  3 bits: fat y position
     *  7 bits: unused
     * 54 bits: holds lower 3x6 cell grid
     */
    u64 b2 = 0;

    u64 hash = 0;

    Memory mem;

    explicit Board() = default;
    explicit Board(const u8 values[36]);
    explicit Board(const u8 values[36], u8 x, u8 y);

    MU void setState(const u8 values[36]);

    MU void setFat(u8 x, u8 y);
    MU void setFatX(u8 x);
    MU void setFatY(u8 y);
    MU void addFatX(u8 x);
    MU void addFatY(u8 y);
    MUND u8 getFatX() const;
    MUND u8 getFatY() const;
    MUND u8 getFatXY() const;
    MUND bool hasFat() const;



    MUND u32 getColorCount() const;

    MU void precomputeHash(u32 colorCount);

    MUND u64 getScore1(const Board& other) const;
    MUND u64 getScore3(const Board& other) const;

    MUND std::string toString() const;
    MUND std::string toString(const Board& other) const;


    bool operator==(const Board& other) const {
        static constexpr u64 MASK = 0x3FFFFFFFFFFFFF;
        return (b1) == (other.b1)
               && (b2) == (other.b2);
    }

    bool operator<(const Board& other) const {
        return this->hash < other.hash;
    }


    bool operator>(const Board& other) const {
        return this->hash > other.hash;
    }
};
