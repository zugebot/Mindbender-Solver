#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/intersection.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/memory.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/rotations.hpp"
#include "MindbenderSolver/unused/sorter.hpp"
#include "MindbenderSolver/utils/timer.hpp"
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>


int calculateColorDistributionHeuristic(C Board& currentBoard, C Board& goalBoard) {
    C int numColors = 8; // Assuming colors are numbered from 0 to 7
    C int gridSize = 6;  // 6x6 grid

    // Arrays to hold color counts for rows and columns
    int currentRowColorCounts[gridSize][numColors] = {0};
    int currentColColorCounts[gridSize][numColors] = {0};
    int goalRowColorCounts[gridSize][numColors] = {0};
    int goalColColorCounts[gridSize][numColors] = {0};

    // Populate color counts for the current board
    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            uint8_t color = currentBoard.getColor(col, row); // x = col, y = row
            currentRowColorCounts[row][color]++;
            currentColColorCounts[col][color]++;
        }
    }

    // Populate color counts for the goal board
    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            uint8_t color = goalBoard.getColor(col, row); // x = col, y = row
            goalRowColorCounts[row][color]++;
            goalColColorCounts[col][color]++;
        }
    }

    // Calculate total differences in color counts for rows and columns
    int totalRowDifferences = 0;
    int totalColDifferences = 0;

    for (int row = 0; row < gridSize; ++row) {
        for (int color = 0; color < numColors; ++color) {
            int diff = std::abs(currentRowColorCounts[row][color] - goalRowColorCounts[row][color]);
            totalRowDifferences += diff;
        }
    }

    for (int col = 0; col < gridSize; ++col) {
        for (int color = 0; color < numColors; ++color) {
            int diff = std::abs(currentColColorCounts[col][color] - goalColColorCounts[col][color]);
            totalColDifferences += diff;
        }
    }

    // Since shifting a row or column can adjust multiple cells,
    // we estimate the minimal number of shifts needed by dividing
    // the total differences by the grid size (6), rounding up.
    int estimatedRowShifts = (totalRowDifferences + gridSize - 1) / gridSize;
    int estimatedColShifts = (totalColDifferences + gridSize - 1) / gridSize;

    // The heuristic value is the sum of estimated row and column shifts
    int heuristicValue = estimatedRowShifts + estimatedColShifts;

    return heuristicValue;
}



struct SearchNode {
    Board board;
    int gCost{}; // Cost from start to current node
    int hCost{}; // Heuristic cost estimate to goal
    int fCost{}; // Total estimated cost (gCost + hCost)
    std::vector<ActStruct> actionsTaken; // Sequence of actions leading to this node

    bool operator>(C SearchNode& other) C {
        return fCost > other.fCost; // Min-heap based on fCost
    }
};

std::pair<Board, int> solvePuzzle(C Board& startBoard, C Board& goalBoard) {
    std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>> openSet;
    std::unordered_set<u64> closedSet; // Using hash of the board for fast lookup

    // Initial node
    SearchNode startNode;
    startNode.board = startBoard;
    startNode.gCost = 0;
    startNode.hCost = calculateColorDistributionHeuristic(startBoard, goalBoard);
    startNode.fCost = startNode.gCost + startNode.hCost;

    openSet.push(startNode);

    while (!openSet.empty()) {
        SearchNode currentNode = openSet.top();
        openSet.pop();

        // Check if the current board state is the goal state
        if (currentNode.board == goalBoard) {

            // Solution found; actionsTaken contains the sequence of moves
            // You can return currentNode.gCost or the actionsTaken vector
            return std::make_pair<>(currentNode.board, currentNode.gCost);
        }

        // Add current board state to closed set
        closedSet.insert(currentNode.board.getHash());

        // Generate possible moves
        for (int i = 0; i < 60; i++) {
            Board nextBoard = currentNode.board;
            allActStructList[i].action(nextBoard); // Apply action to the board
            nextBoard.precomputeHash2();
            nextBoard.memory.setNextNMove<1>(i);

            // Check if the next state has already been visited
            if (closedSet.find(nextBoard.memory.getMem()) != closedSet.end()) {
                continue; // Skip already visited states
            }

            // Calculate costs
            int gCost = currentNode.gCost + 1; // Each move costs 1
            int hCost = calculateColorDistributionHeuristic(nextBoard, goalBoard);
            int fCost = gCost + hCost;

            // Create new search node
            SearchNode nextNode;
            nextNode.board = nextBoard;
            nextNode.gCost = gCost;
            nextNode.hCost = hCost;
            nextNode.fCost = fCost;
            nextNode.actionsTaken = currentNode.actionsTaken;
            nextNode.actionsTaken.push_back(allActStructList[i]); // Record the action taken

            // Add to open set
            openSet.push(nextNode);
        }
    }

    // No solution found within search limits
    return std::make_pair<>(Board(), -1);
}









int main() {


    auto pair = BoardLookup::getBoardPair("8-5");
    Board board1 = pair->getInitialState(); // pair->getInitialState();
    Board board2 = pair->getSolutionState();

    // actions[15](board2);
    // actions[37](board2);
    // actions[54](board2);
    // actions[2](board2);
    // actions[7](board2);
    // actions[58](board2);
    // actions[12](board2);

    Timer timer;
    auto [board, cost] = solvePuzzle(board1, board2);

    std::cout << "Cost: " << cost << std::endl;
    std::cout << "Time: " << timer.getSeconds() << std::endl;
    std::cout << board.memory.asmStringForwards() << std::endl;
    std::cout << board1.toString(board) << std::endl;




    return 0;


}
