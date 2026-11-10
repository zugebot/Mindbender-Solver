package IDA;

import Boards.Board;
import Boards.Levels;

public class IterativeDeepeningAStar {
    private static final int MAX_DEPTH = 5;
    private static boolean foundSolution = false;
    private static Levels levels = new Levels();

    private static short s0 = 0;
    private static byte b0 = 0;
    private static byte b1 = 0;
    private static byte b2 = 0;
    private static byte b3 = 0;
    private static byte b4 = 0;
    private static int[] uncoveredRows = new int[6];
    private static int[] uncoveredColumns = new int[6];
    private static int differingCells = 0;
    private static int lanes = 0;
    private static int maxCover = 0;
    private static boolean isRow = false;
    private static int index = 0;

    public static void main(String[] args) {
        Board initialState = levels.b9_4;
        Board solution = levels.s9_4;
        // Initialize the board instances as needed

        // Perform the IDA* search
        iterativeDeepeningAStar(initialState, solution);

        // Print the solution and moves
        solution.print("Solution");
        System.out.println("Moves: " + solution.getMoves());
    }

    public static void iterativeDeepeningAStar(Board initialState, Board solution) {
        initialState.updateHeuristic(s0, b0, b1, b2, b3, b4, solution);
        int threshold = initialState.getHeuristic();

        while (!foundSolution) {
            short newThreshold = depthLimitedSearch(initialState, solution, 0, threshold);
            if (newThreshold == -1) {
                break;
            }
            threshold = newThreshold;
        }
    }

    private static short depthLimitedSearch(Board currentState, Board solution, int depth, int threshold) {
        if (depth == MAX_DEPTH) {
            return -1;
        }

        if (currentState.isGoalState(solution)) {
            foundSolution = true;
            return -1;
        }

        short minCost = Short.MAX_VALUE;
        byte[][] possibleMoves = currentState.getPossibleMoves();

        for (byte[] move : possibleMoves) {
            byte[] temp = new byte[6];
            currentState.doMove(temp, move[0], move[1], move[2]);
            currentState.updateHeuristic(s0, b0, b1, b2, b3, b4, solution);
            short cost = (short) (depth + 1 + currentState.getHeuristic());
            currentState.undoMove(temp);

            if (cost <= threshold) {
                short result = depthLimitedSearch(currentState, solution, depth + 1, threshold);
                if (foundSolution) {
                    return -1;
                } else if (result != -1) {
                    minCost = (short) Math.min(minCost, result);
                }
            } else {
                minCost = (short) Math.min(minCost, cost);
            }
        }

        return minCost == Short.MAX_VALUE ? -1 : minCost;
    }

}

