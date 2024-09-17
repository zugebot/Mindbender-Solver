#include "board.hpp"

#include <iostream>
#include <string>


void Board::setState(const uint8_t values[36]) {
    b1 = 0;
    for (int i = 0; i < 18; i++) {
        b1 = b1 << 3 | (values[i] & 0b111);
    }
    b2 = 0;
    for (int i = 18; i < 36; i++) {
        b2 = b2 << 3 | (values[i] & 0b111);
    }
}


/**
 * returns the ..100.000.100.000... of board1 compared to board2
 * ..001.. if cells are similar in value
 * ..000.. if cells differ in value
 * @param sect1 b1/b2 of 1st board
 * @param sect2 b1/b2 of 2nd board
 * @return
 */
inline uint64_t getSimilar(const uint64_t& sect1, const uint64_t& sect2) {
    static constexpr uint64_t M0 = 0x0009249249249249;
    const uint64_t s = sect1 ^ sect2;
    return ~(s | s >> 1 | s >> 2) & M0;
}


uint64_t score1Helper(const uint64_t& sect) {
    static constexpr uint64_t M3 = 0x0000E070381C0E07;
    static constexpr uint64_t M4 = 0x000000007800000F;
    static constexpr uint64_t M5 = 0x0000000007FFFFFF;

    const uint64_t p1 = sect + (sect >> 3) + (sect >> 6) & M3;
    const uint64_t p2 = p1 + (p1 >> 9) + (p1 >> 18) & M4;
    return (p2 & M5) + (p2 >> 27);
}

uint64_t Board::getScore1(const Board &other) const {
    return score1Helper(getSimilar(b1, other.b1))
         + score1Helper(getSimilar(b2, other.b2));
}

std::string Board::assembleMoveString(Board* other) const {
    std::string start = assembleMoveStringForwards();
    std::string end = other->assembleMoveStringBackwards();
    return start + " " + end;
}


std::string Board::assembleMoveStringForwards() const {
    std::string moves;
    const int count = mem.getMoveCount();
    for (int i = 0; i < count; i++) {
        uint8_t move = mem.getMove(i);
        uint8_t rowCol = (move % 30) / 5;
        uint8_t amount = 1 + move % 5;
        char letter = (char)('C' + (15 * (move < 30)));
        moves += letter + std::to_string(rowCol)
                 + std::to_string(amount);
        if (i != count - 1) {
            moves += " ";
        }
    }
    return moves;
}


std::string Board::assembleMoveStringBackwards() const {
    std::string moves;
    const int count = mem.getMoveCount();
    for (int i = count - 1; i >= 0; i--) {
        uint8_t move = mem.getMove(i);
        uint8_t rowCol = (move % 30) / 5;
        uint8_t amount = 6 - (1 + move % 5);
        char letter = (char)('C' + (15 * (move < 30)));
        moves += letter + std::to_string(rowCol)
                 + std::to_string(amount);
        if (i != 0) {
            moves += " ";
        }
    }
    return moves;
}



uint64_t Board::getScore3(const Board& other) const {


    uint64_t ROW = 0;
    auto *uncoveredRows = reinterpret_cast<uint8_t *>(&ROW);
    uint64_t COL = 0;
    auto *uncoveredCols = reinterpret_cast<uint8_t *>(&COL);

    /*

    // Find all differing cells and update the counts in rows and cols
    // for C++, I can instantly sum the rows, but IDK about the columns
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (this.myBoard[i][j] != theOther.myBoard[i][j]) {
                uncoveredRows[i]++;
                uncoveredCols[j]++;
                differingCells++;
            }
        }
    }


    // While there are still uncovered differing cells
    // worst case this should only occur 6 times?
    int lanes = 0;
    int maxCover = 0;
    bool isRow = false;
    int index = -1;
    while (differingCells > 0) {
        maxCover = 0;
        isRow = false;
        index = -1;
        // Find the row or column that covers the most uncovered differing cells
        // in C++, can probably reinterpret the bytes to see if either can be skipped,
        // base which level of checking I am doing off of getScore1?
        // can be recoded to find the index and value of the max in both?
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (uncoveredRows[i] > maxCover) {
                maxCover = uncoveredRows[i];
                isRow = true;
                index = i;
            }
            if (uncoveredCols[i] > maxCover) {
                maxCover = uncoveredCols[i];
                isRow = false;
                index = i;
            }
        }

        if (index == -1) {
            break;
        }

        // Cover the chosen row or column and update the counts in
        // uncoveredRows and uncoveredColumns
        // I could cache the results of getScore1 for this lol
        if (isRow) {
            differingCells -= uncoveredRows[index];
            uncoveredRows[index] = 0;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (this.myBoard[index][j] != theOther.myBoard[index][j] && uncoveredCols[j] > 0) {
                    uncoveredCols[j]--;
                }
            }
        } else {
            differingCells -= uncoveredCols[index];
            uncoveredCols[index] = 0;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (this.myBoard[j][index] != theOther.myBoard[j][index] && uncoveredRows[j] > 0) {
                    uncoveredRows[j]--;
                }
            }
        }

        lanes++;
    }
    return lanes;
    */
    return 0;

}



void Board::precompute_hash() {
    constexpr uint64_t prime = 31;
    static constexpr uint64_t MASK = 0x3FFFFFFFFFFFFF;
    hash = 17;
    hash = hash * prime + ((b1 & MASK) ^ (b1 & MASK) >> 32);
    hash = hash * prime + ((b2 & MASK) ^ (b2 & MASK) >> 32);
}


std::string Board::toString() const {
    std::string str;

    auto appendBoardToString = [&str](const uint64_t board) {
        for (int y = 0; y < 54; y += 18) {
            for (int x = 0; x < 18; x += 3) {
                str.append(std::to_string(board >> (51 - x - y) & 0b111));
                str.append(" ");
            }
            str.append("\n");
        }
    };

    appendBoardToString(b1);
    appendBoardToString(b2);
    return str;
}

