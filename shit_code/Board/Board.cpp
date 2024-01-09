#include "Board.h"
#include "BoardArray.h"

typedef unsigned char byte;


std::random_device Board::RANDOM;
int** Board::MOVE_TABLE_SIZE = Board::generateMoveTableSize();
byte** Board::MODULO_TABLE = Board::generateModuloTable();
byte**** Board::MOVE_TABLE = Board::generateMoveTable();

const std::string Board::colors = "ROYGCBPW-";

Board::Board() = default;

Board::Board(const std::string& board_str) {
    load(board_str);
}

byte** Board::generateModuloTable() {
    byte i, j;
    byte** table = new byte*[BOARD_SIZE]();
    for (i = 0; i < BOARD_SIZE; i++) {
        table[i] = new byte[BOARD_SIZE]();
        for (j = 0; j < BOARD_SIZE; j++) {
            table[i][j] = (i + j) % BOARD_SIZE;
        }
    }
    return table;
}

byte Board::getModulo(byte x, byte y) {
    return MODULO_TABLE[x][y];
}

int** Board::generateMoveTableSize() {
    int** table = new int*[2]();
    for (int i = 0; i < 2; i++) {
        table[i] = new int[6]();
    }
    return table;
}


byte**** Board::generateMoveTable() {
    byte direction1, highsect, count, direction2, section, amount;
    int size;

    byte**** table = new byte***[2]();
    for (int i = 0; i < 2; i++) {
        table[i] = new byte**[6]();
    }

    for (direction1 = 0; direction1 < 2; direction1++) {
        table[direction1] = new byte**[6]();
        for (highsect = 0; highsect < 6; highsect++) {
            count = 0;
            size = 30 + 5 * (6 - highsect);
            MOVE_TABLE_SIZE[direction1][highsect] = size;
            table[direction1][highsect] = new byte*[size]();

            for (direction2 = 0; direction2 < 2; direction2++) {

                if (direction1 == direction2) {
                    for (section = highsect; section < 6; section++) {
                        for (amount = 1; amount < 6; amount++) {
                            if (count >= size) {
                                std::cerr << "Array index out of bound: count = " << count << ", size = " << size << std::endl;
                                return nullptr;
                            }
                            table[direction1][highsect][count] = new byte[3]{direction2, section, amount};
                            count++;
                        }
                    }
                } else {
                    for (section = 0; section < 6; section++) {
                        for (amount = 1; amount < 6; amount++) {
                            if (count >= size) {
                                std::cerr << "Array index out of bound: count = " << count << ", size = " << size << std::endl;
                                return nullptr;
                            }
                            table[direction1][highsect][count] = new byte[3]{direction2, section, amount};
                            count++;
                        }
                    }
                }
            }
        }
    }
    return table;
}

void Board::printMoveTable() {
    if (Board::MOVE_TABLE == nullptr) {
        std::cerr << "Table is null" << std::endl;
        return;
    }

    for (int direction1 = 0; direction1 < 2; ++direction1) {
        if (Board::MOVE_TABLE[direction1] == nullptr) {
            std::cerr << "table[" << direction1 << "] is null" << std::endl;
            continue;
        }

        for (int highsect = 0; highsect < 6; ++highsect) {
            if (Board::MOVE_TABLE[direction1][highsect] == nullptr) {
                std::cerr << "table[" << direction1 << "][" << highsect << "] is null" << std::endl;
                continue;
            }

            int size = Board::MOVE_TABLE_SIZE[direction1][highsect];
            for (int count = 0; count < size; ++count) {
                if (Board::MOVE_TABLE[direction1][highsect][count] == nullptr) {
                    std::cerr << "table[" << direction1 << "][" << highsect << "][" << count << "] is null" << std::endl;
                    continue;
                }

                std::cout << "table[" << direction1 << "][" << highsect << "][" << count << "] = {";
                for (int i = 0; i < 3; ++i) {
                    std::cout << static_cast<int>(Board::MOVE_TABLE[direction1][highsect][count][i]);
                    if (i < 2) std::cout << ", ";
                }
                std::cout << "}\n";
            }
        }
    }
}


bool Board::operator==(const Board& other) const {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] != other.board[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void Board::load(const std::string& board_str) {
    if (board_str.empty()) {return;}

    byte x = 0; byte y = 0; byte value;
    for (char c : board_str) {
        if (c == ' ' || c == '\n') {
            continue;
        }
        value = -1;
        for (byte i = 0; i < 9; i++) {
            if (c == colors[i]) {value = i;}
        }
        board[x][y] = value;
        y++;
        if (y == 6) {
            y = 0;
            x++;
        }
    }

}

byte Board::getDirection(byte index) const {
    return moves[index] & 3;
}

byte Board::getDirection() const {
    return getDirection(moveNum);
}

byte Board::getAmount(byte index) const {
    return (moves[index] >> 5) & 7;
}

byte Board::getAmount() const {
    return getAmount(moveNum);
}

byte Board::getSection(byte index) const {
    return (moves[index] >> 2) & 7;
}

byte Board::getSection() const {
    return getSection(moveNum);
}

void Board::setDirection(byte val) {
    moves[moveNum] = (moves[moveNum] & 252) | val;
}

void Board::setMove(byte section, byte amount, byte dir) {
    highSect = section;
    moves[moveNum] = amount << 5 | section << 2 | dir;
    moveNum++;
    setDirection(dir);

}

void Board::setMoveROR(byte section, byte amount) {
    highSect = section;
    moves[moveNum] = amount << 5 | section << 2 | Bytes::b0;
    moveNum++;
    setDirection(Bytes::b0);
}

void Board::setMoveCOD(byte section, byte amount) {
    highSect = section;
    moves[moveNum] = amount << 5 | section << 2 | Bytes::b1;
    moveNum++;
    setDirection(Bytes::b1);
}

/**
 *
 * @param temp byte[6] - for usage of rotating sections.
 */
void Board::undoMove(byte* temp) {
    moves[moveNum] = 0;
    moveNum--;
    doMove(temp,
           moves[moveNum] & 3,
           (moves[moveNum] >> 2) & 7,
           BOARD_SIZE - ((moves[moveNum] >> 5) & 7)
    );
    moves[moveNum] = 0;
    moveNum--;
}

void Board::doMove(byte* temp, byte* move) {
    if (move[0] == 0) {
        ROR(temp, move[1], move[2]);
    } else {
        COD(temp, move[1], move[2]);
    }
}

void Board::doMove(byte* temp, byte direction, byte section, byte amount) {
    if (direction == 0) {
        ROR(temp, section, amount);
    } else {
        COD(temp, section, amount);
    }
}

BoardArray Board::possibleBoards() {
    auto boards = BoardArray(60);
    byte* temp = new byte[BOARD_SIZE] {0, 0, 0, 0, 0, 0};
    for (int index = 0; index < MOVE_TABLE_SIZE[0][0]; index++) {
        copy(boards, index);
        boards._array[index].doMove(temp, MOVE_TABLE[0][0][index]);
    }
    /*
    for (int i = 0; i < 60; i++) {
        boards._array[i].print(std::to_string(i) + "/60");
        std::cout << boards._array[i].hashCode() << std::endl;
    }
     */

    delete[] temp;
    return boards;
}

void Board::print(const std::string& header) const {
    std::cout << header << std::endl;
    print();
}

void Board::print() const {
    byte x, y;
    std::ostringstream string;
    for (y = 0; y < BOARD_SIZE; y++) {
        std::ostringstream row;
        for (x = 0; x < BOARD_SIZE; x++) {
            std::string slot = std::to_string(board[y][x]);
            slot = std::string(1, slot[0]); // assuming slot should be 1 character long
            row << Colors::getColor(board[y][x]) << slot << " ";
        }
        string << row.str() << "\n";
    }
    string << Colors::Reset;
    std::cout << string.str() << std::endl;
}

std::string Board::getDirectionString(byte index) const {
    switch (getDirection(index)) {
        case 1: return "R";
        case 2: return "C";
        default: return "#";
    }
}

std::string Board::getMoves() const {
    std::string moveset;
    for (int i = 0; i < moveNum; i++) {
        moveset += getDirectionString(i);
        moveset += std::to_string(getAmount(i));
        moveset += std::to_string(getAmount(i));
        moveset += " ";
    }
    return moveset;
}

std::string Board::getMovesReversed() const {
    byte val;
    std::string moveset;
    for (byte i = moveNum; i > 0; i--) {
        val = Bytes::MINUS_ONE_TABLE[i];
        moveset += getDirectionString(val);
        moveset += std::to_string(getSection(val));
        moveset += std::to_string(BOARD_SIZE - getAmount(val));
        if (i != 1) {
            moveset += " ";
        }
    }
    return moveset;
}

byte** Board::getPossibleMoves() const {
    return MOVE_TABLE[getDirection()][highSect];
}

int Board::getPossibleMovesArraySize() const {
    return MOVE_TABLE_SIZE[getDirection()][highSect];
}

void Board::ROR(byte* temp, byte row, byte amount) {
    // write row to temp with offset
    temp[MODULO_TABLE[amount][0]] = board[row][0];
    temp[MODULO_TABLE[amount][1]] = board[row][1];
    temp[MODULO_TABLE[amount][2]] = board[row][2];
    temp[MODULO_TABLE[amount][3]] = board[row][3];
    temp[MODULO_TABLE[amount][4]] = board[row][4];
    temp[MODULO_TABLE[amount][5]] = board[row][5];
    // rewrite temp back to row
    std::memcpy(board[row], temp, BOARD_SIZE);
    // update moveset
    setMoveROR(row, amount);
}

void Board::COD(byte* temp, byte col, byte amount) {
    // store the row in temp variable
    temp[MODULO_TABLE[0][amount]] = board[0][col];
    temp[MODULO_TABLE[1][amount]] = board[1][col];
    temp[MODULO_TABLE[2][amount]] = board[2][col];
    temp[MODULO_TABLE[3][amount]] = board[3][col];
    temp[MODULO_TABLE[4][amount]] = board[4][col];
    temp[MODULO_TABLE[5][amount]] = board[5][col];
    // rewrite the row back with offset
    board[0][col] = temp[0];
    board[1][col] = temp[1];
    board[2][col] = temp[2];
    board[3][col] = temp[3];
    board[4][col] = temp[4];
    board[5][col] = temp[5];
    // update moveset
    setMoveCOD(col, amount);
}

Board* Board::randomMoves(byte moveCount) {
    std::uniform_int_distribution<int> dist_0_5(0, 5);
    std::uniform_int_distribution<int> dist_1_5(1, 5);
    std::uniform_int_distribution<int> dist_0_2(0, 2);
    byte* temp = new byte[]{0, 0, 0, 0, 0, 0};
    for (byte i = 0; i < moveCount; i++) {
        doMove(temp,
               dist_0_5(RANDOM),
               dist_1_5(RANDOM),
               dist_0_2(RANDOM));
    }
    return this;
}

short Board::getScore1(Board other) const {
    return (short) (
            ((board[0][0] == other.board[0][0]) ? 1 : 0) +
            ((board[0][1] == other.board[0][1]) ? 1 : 0) +
            ((board[0][2] == other.board[0][2]) ? 1 : 0) +
            ((board[0][3] == other.board[0][3]) ? 1 : 0) +
            ((board[0][4] == other.board[0][4]) ? 1 : 0) +
            ((board[0][5] == other.board[0][5]) ? 1 : 0) +
            ((board[1][0] == other.board[1][0]) ? 1 : 0) +
            ((board[1][1] == other.board[1][1]) ? 1 : 0) +
            ((board[1][2] == other.board[1][2]) ? 1 : 0) +
            ((board[1][3] == other.board[1][3]) ? 1 : 0) +
            ((board[1][4] == other.board[1][4]) ? 1 : 0) +
            ((board[1][5] == other.board[1][5]) ? 1 : 0) +
            ((board[2][0] == other.board[2][0]) ? 1 : 0) +
            ((board[2][1] == other.board[2][1]) ? 1 : 0) +
            ((board[2][2] == other.board[2][2]) ? 1 : 0) +
            ((board[2][3] == other.board[2][3]) ? 1 : 0) +
            ((board[2][4] == other.board[2][4]) ? 1 : 0) +
            ((board[2][5] == other.board[2][5]) ? 1 : 0) +
            ((board[3][0] == other.board[3][0]) ? 1 : 0) +
            ((board[3][1] == other.board[3][1]) ? 1 : 0) +
            ((board[3][2] == other.board[3][2]) ? 1 : 0) +
            ((board[3][3] == other.board[3][3]) ? 1 : 0) +
            ((board[3][4] == other.board[3][4]) ? 1 : 0) +
            ((board[3][5] == other.board[3][5]) ? 1 : 0) +
            ((board[4][0] == other.board[4][0]) ? 1 : 0) +
            ((board[4][1] == other.board[4][1]) ? 1 : 0) +
            ((board[4][2] == other.board[4][2]) ? 1 : 0) +
            ((board[4][3] == other.board[4][3]) ? 1 : 0) +
            ((board[4][4] == other.board[4][4]) ? 1 : 0) +
            ((board[4][5] == other.board[4][5]) ? 1 : 0) +
            ((board[5][0] == other.board[5][0]) ? 1 : 0) +
            ((board[5][1] == other.board[5][1]) ? 1 : 0) +
            ((board[5][2] == other.board[5][2]) ? 1 : 0) +
            ((board[5][3] == other.board[5][3]) ? 1 : 0) +
            ((board[5][4] == other.board[5][4]) ? 1 : 0) +
            ((board[5][5] == other.board[5][5]) ? 1 : 0));
}

short Board::getScore2(short scoreTotal, byte scoreMax, byte i, byte sect,
                byte offset, byte currentScore, Board other) const {
    scoreTotal = 0;
    // Calculate score both horizontally and vertically
    for (sect = 0; sect < BOARD_SIZE; sect++) {
        scoreMax = 0;
        // Calculate offsets
        for (offset = 0; offset < BOARD_SIZE; offset++) {
            currentScore = 0;
            for (i = 0; i < BOARD_SIZE; i++) {
                if (board[sect][getModulo(i, offset)] == other.board[sect][i]) {
                    currentScore++;
                }
                if (board[getModulo(i, offset)][sect] == other.board[i][sect]) {
                    currentScore++;
                }
            }
            // Update the maximum score if necessary
            if (currentScore > scoreMax) {
                scoreMax = currentScore;
            }
        }
        scoreTotal += scoreMax;
    }

    return (short)(scoreTotal * MODIFIER_SCORE_2);
}

/*
bool isGoalState(Board solve) {
    return equals(solve);
}*/

std::string Board::getScoreString(Board solve) const {
    byte b0 = 0; byte b1 = 0; byte b2 = 0;
    byte b3 = 0; byte b4 = 0; byte b5 = 0;
    int score1 = this->getScore1(solve);
    int score2 = this->getScore2(b0, b1, b2, b3, b4, b5, solve);
    return
        std::to_string(score2 / MODIFIER_SCORE_2) + " " +
        std::to_string(score1);
}

void Board::updateHeuristic(byte scoreTotal, byte scoreMax, byte i, byte sect,
                     byte offset, byte currentScore, Board solve) {
    score = (short) (
            getScore1(solve) +
            getScore2(scoreTotal, scoreMax, i, sect,
                      offset, currentScore, solve)
            );
}

short Board::getHeuristic() const {
    return score;
}

void Board::copy(BoardArray& boardArray, int index) {
    // copy the board array
    std::memcpy(boardArray._array[index].board, board, BOARD_SIZE * BOARD_SIZE);
    // copy single byte objects
    boardArray._array[index].moveNum = moveNum;
    boardArray._array[index].highSect = highSect;
    // copy moveset
    std::memcpy(boardArray._array[index].moves, moves, MOVE_COUNT);
}

/**
 * Copies current board object to board at the pointer.
 * @param newBoard
 * @return
 */
void Board::copyInto(Board newBoard) const {
    // copy the board array
    std::memcpy(newBoard.board, board, BOARD_SIZE * BOARD_SIZE);
    // copy single byte objects
    newBoard.moveNum = moveNum;
    newBoard.highSect = highSect;
    // copy moveset
    std::memcpy(newBoard.moves, moves, MOVE_COUNT);
}


int Board::hashCode() const {
    return (((((((((((((((((((((((((((((((((((17
                                              * 31 + board[0][0]) * 31 + board[0][1]) * 31 + board[0][2]) * 31 + board[0][3]) * 31 + board[0][4]) * 31 + board[0][5])
                                        * 31 + board[1][0]) * 31 + board[1][1]) * 31 + board[1][2]) * 31 + board[1][3]) * 31 + board[1][4]) * 31 + board[1][5])
                                  * 31 + board[2][0]) * 31 + board[2][1]) * 31 + board[2][2]) * 31 + board[2][3]) * 31 + board[2][4]) * 31 + board[2][5])
                            * 31 + board[3][0]) * 31 + board[3][1]) * 31 + board[3][2]) * 31 + board[3][3]) * 31 + board[3][4]) * 31 + board[3][5])
                      * 31 + board[4][0]) * 31 + board[4][1]) * 31 + board[4][2]) * 31 + board[4][3]) * 31 + board[4][4]) * 31 + board[4][5])
                * 31 + board[5][0]) * 31 + board[5][1]) * 31 + board[5][2]) * 31 + board[5][3]) * 31 + board[5][4]) * 31 + board[5][5];
}


/*
void Board::copy(Board board, Board* boardArray, int index) {
    // copy the board array
    std::memcpy(boardArray[index].board, board.board, BOARD_SIZE * BOARD_SIZE);
    // copy single byte objects
    boardArray[index].moveNum = board.moveNum;
    boardArray[index].highSect = board.highSect;
    // copy moveset
    std::memcpy(boardArray[index].moves, board.moves, MOVE_COUNT);
}*/