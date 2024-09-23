#pragma once

#include <utility>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>

class BoardArray;
#include "../support/bytes.h"
#include "../support/colors.h"


typedef unsigned char byte;
#define ND [[nodiscard]]

class Board {
private:
    static std::random_device RANDOM;
    static byte** MODULO_TABLE;
    static int** MOVE_TABLE_SIZE;
    static byte**** MOVE_TABLE;

    static byte** generateModuloTable();
    static byte getModulo(byte x, byte y);
    static int** generateMoveTableSize();
    static byte**** generateMoveTable();

public:
    static const int BOARD_SIZE = 6;
    static const int MOVE_COUNT = 11;
    static const int MODIFIER_SCORE_1 = 1;
    static const int MODIFIER_SCORE_2 = 36;
    static const std::string colors;

    byte board[BOARD_SIZE][BOARD_SIZE]{};
    byte moves[MOVE_COUNT]{};
    byte highSect = 0;
    byte moveNum = 0;
    byte score = 0;

    static void printMoveTable();

    Board();
    explicit Board(const std::string& board_str);
    bool operator==(const Board& other) const;
    void load(const std::string& board_str);
    ND byte getDirection(byte index) const;
    ND byte getDirection() const;
    ND byte getAmount(byte index) const;
    ND byte getAmount() const;
    ND byte getSection(byte index) const;
    ND byte getSection() const;
    void setDirection(byte val);
    void setMove(byte section, byte amount, byte dir);
    void setMoveROR(byte section, byte amount);
    void setMoveCOD(byte section, byte amount);
    void undoMove(byte* temp);
    void doMove(byte* temp, byte* move);
    void doMove(byte* temp, byte direction, byte section, byte amount);
    BoardArray possibleBoards();
    void print(const std::string& header) const;
    void print() const;
    ND std::string getDirectionString(byte index) const;
    ND std::string getMoves() const;
    ND std::string getMovesReversed() const;
    ND byte** getPossibleMoves() const;
    ND int getPossibleMovesArraySize() const;
    void ROR(byte* temp, byte row, byte amount);
    void COD(byte* temp, byte col, byte amount);
    Board* randomMoves(byte moveCount);
    ND short getScore1(Board other) const;
    ND short getScore2(short scoreTotal, byte scoreMax, byte i, byte sect,
                    byte offset, byte currentScore, Board other) const;
    ND std::string getScoreString(Board solve) const;
    void updateHeuristic(byte scoreTotal, byte scoreMax, byte i, byte sect,
                         byte offset, byte currentScore, Board solve);
    ND short getHeuristic() const;

    void copy(BoardArray& boardArray, int index);
    void copyInto(Board newBoard) const;
    ND int hashCode() const;

    // static void copy(Board board, Board* boardArray, int index);
};














