#pragma once

#include "Board.h"

struct BoardHash {
    size_t operator()(const Board& board) const {
        return (((((((((((((((((((((((((((((((((((17
                                                  * 31 + board.board[0][0]) * 31 + board.board[0][1]) * 31 + board.board[0][2]) * 31 + board.board[0][3]) * 31 + board.board[0][4]) * 31 + board.board[0][5])
                                            * 31 + board.board[1][0]) * 31 + board.board[1][1]) * 31 + board.board[1][2]) * 31 + board.board[1][3]) * 31 + board.board[1][4]) * 31 + board.board[1][5])
                                      * 31 + board.board[2][0]) * 31 + board.board[2][1]) * 31 + board.board[2][2]) * 31 + board.board[2][3]) * 31 + board.board[2][4]) * 31 + board.board[2][5])
                                * 31 + board.board[3][0]) * 31 + board.board[3][1]) * 31 + board.board[3][2]) * 31 + board.board[3][3]) * 31 + board.board[3][4]) * 31 + board.board[3][5])
                          * 31 + board.board[4][0]) * 31 + board.board[4][1]) * 31 + board.board[4][2]) * 31 + board.board[4][3]) * 31 + board.board[4][4]) * 31 + board.board[4][5])
                    * 31 + board.board[5][0]) * 31 + board.board[5][1]) * 31 + board.board[5][2]) * 31 + board.board[5][3]) * 31 + board.board[5][4]) * 31 + board.board[5][5];
    }

};