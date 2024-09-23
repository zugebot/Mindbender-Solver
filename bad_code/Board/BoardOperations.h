# pragma once

#include <utility>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <chrono>
#include <functional> // for std::hash
#include <iostream>
#include <unordered_set>

#include "../support/bytes.h"
#include "../support/colors.h"
#include "../support/time.h"

#include "Board.h"
#include "BoardArray.h"
#include "BoardHash.h"


typedef unsigned char byte;
class BoardOperations {
public:
    static const int BOARD_SIZE = Board::BOARD_SIZE;

    static int myMaximumSize;
    static bool myDebug;
    static byte b0;
    static byte b1;
    static byte b2;
    static byte b3;
    static byte b4;
    static byte b5;

    static BoardArray extend(BoardArray& board, Board& solve, double time);
    static BoardArray generateUnique(BoardArray& boardsInput, int multiple);
    static std::vector<std::array<Board, 2>> find_intersection(const BoardArray& boards, const BoardArray& solves);
    static BoardArray resizeArray(BoardArray& boards, int count);
    static BoardArray removeDuplicates(const BoardArray& boards);
    static void sort(BoardArray& boards, const Board& solve);
    static std::string getMovesetString(std::array<Board, 2> pair);
    static void printDebug(double time, const std::string& message_str);
};



























