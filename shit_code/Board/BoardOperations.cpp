#include "BoardOperations.h"


int BoardOperations::myMaximumSize = 0;
bool BoardOperations::myDebug = true;
byte BoardOperations::b0 = 0;
byte BoardOperations::b1 = 0;
byte BoardOperations::b2 = 0;
byte BoardOperations::b3 = 0;
byte BoardOperations::b4 = 0;
byte BoardOperations::b5 = 0;


BoardArray BoardOperations::extend(BoardArray& board, Board& solve, double time) {
    double start;

    start = Time::getTime();
    BoardArray boards = BoardOperations::generateUnique(board, 55);
    BoardOperations::printDebug(start, "Generating...     [" + std::to_string(boards.size()) + "]");

    auto size = boards.size();
    boards = BoardOperations::removeDuplicates(boards);
    BoardOperations::printDebug(start, "Removing Dupes... [" + std::to_string(size) + "->" +
    std::to_string(boards.size()) + "]");

    if (boards.size() > myMaximumSize) {

        start = Time::getTime();
        BoardOperations::sort(boards, solve);
        BoardOperations::printDebug(start, "Sorting...        []");

        start = Time::getTime();
        boards = BoardOperations::resizeArray(boards, myMaximumSize);
        BoardOperations::printDebug(start, "Resizing...");

    }

    std::cout << "\033[32m   [--------] \033[37m" <<
              "Best Candidate: " << boards._array[0].getScoreString(solve) <<
              "\033[0m" << std::endl;
    return boards;
}



BoardArray BoardOperations::generateUnique(BoardArray& boardsInput, int multiple) {
    int count = 0;
    byte temp[BOARD_SIZE];
    BoardArray boards = BoardArray(boardsInput.size() * multiple);
    std::unordered_set<Board, BoardHash> boardSet;
    for (auto board : boardsInput) {
        board.copy(boards, count);
        for (int i = 0; i < boards._array[count].getPossibleMovesArraySize(); i++) {
            boards._array[count].doMove(temp, boards._array[count].getPossibleMoves()[i]);
            if (boardSet.find(boards._array[count]) != boardSet.end()) {
                boards._array[count].undoMove(temp);
            } else {
                boardSet.insert(boards._array[count]);
                board.copy(boards, count);
                count++;
            }
        }

    }
    boards.resize(count);
    return boards;
}


std::vector<std::array<Board, 2>> BoardOperations::find_intersection(const BoardArray& boards, const BoardArray& solves) {
    std::unordered_map<Board, std::vector<Board>, BoardHash> hash;
    Board temp_board;

    for (const Board& board : boards) {
        board.copyInto(temp_board);

        if (hash.find(temp_board) == hash.end()) {
            auto* temp = new Board(temp_board);
            hash[temp_board] = std::vector<Board>{*temp};
        } else {
            hash[temp_board].push_back(temp_board);
        }
    }

    std::vector<std::array<Board, 2>> pairs;

    Board solve_copy;
    for (const Board& solve : solves) {
        solve.copyInto(solve_copy);
        auto it = hash.find(solve_copy);

        if (it != hash.end()) {
            const std::vector<Board>& piece = it->second;
            for (const Board& board : piece) {
                pairs.push_back({board, solve_copy});
            }
        }
    }

    return pairs;
}




BoardArray BoardOperations::resizeArray(BoardArray& boards, int count) {
    for (int i = count; i < boards.size(); i++) {
        boards._array[i].~Board(); // call destructor of the ith Board object
    }
    boards.resize(count);
    return boards;
}

BoardArray BoardOperations::removeDuplicates(const BoardArray& boards) {
    std::unordered_set<Board, BoardHash> boardSet(boards.begin(), boards.end());
    return {boardSet.begin(), boardSet.end()};
}


void BoardOperations::sort(BoardArray& boards, const Board& solve) {

    for (Board& board : boards) {
        board.updateHeuristic(b0, b1, b2, b3, b4, b5, solve);

    }
    std::sort(boards.begin(), boards.end(), []
            (const Board& a, const Board& b) {
        return a.getHeuristic() < b.getHeuristic();
    });
}

std::string BoardOperations::getMovesetString(std::array<Board, 2> pair) {
    return pair[0].getMoves() + pair[1].getMovesReversed();
}

void BoardOperations::printDebug(double time, const std::string& message_str) {
    std::string debugInfo = Colors::Green + "   [%.5fs]" + Colors::White + " %s" + Colors::Reset + "\n";
    if (!myDebug) {return;}

    double newTime = Time::time(time);
    printf(debugInfo.c_str(), newTime, message_str.c_str());
}





