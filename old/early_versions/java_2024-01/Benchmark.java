import Boards.Board;
import Boards.Levels;
import Boards.BoardOperations;
import support.Bytes;

import java.util.stream.IntStream;

public class Benchmark {

    public static short s0 = 0;
    public static byte b0 = 0;
    public static byte b1 = 0;
    public static byte b2 = 0;
    public static byte b3 = 0;
    public static byte b4 = 0;

    public static void main(String[] args) {
        int warmup = 1_000;
        int numIterations = 1_000_000; // Number of times to call the function
        long startTime, endTime, totalTime;

        Levels levels = new Levels();
        Board board = levels.b4_3;
        Board other = levels.s4_3;


        byte[] temp = new byte[]{0, 0, 0, 0, 0, 0};

        BoardOperations ops = new BoardOperations(1_000_000, true, 30);

        Board[] boards = ops.generateUnique(board.possibleBoards(), 60);
        Board[] solves = ops.generateUnique(board.possibleBoards(), 60);

        System.out.println("Size: " + boards.length + " " + boards.length);

        // getScore1 = ~25ns
        // getScore2 = ~612ns
        // getScore3 = ~

        int[] uncoveredRows = new int[6];
        int[] uncoveredColumns = new int[6];
        int differingCells = 0;
        int lanes = 0;
        int maxCover = 0;
        boolean isRow = false;
        int index = -1;

        for (int iter = 0; iter < warmup; iter++) {
            board.getScore3(other, uncoveredRows, uncoveredColumns,
            differingCells, lanes, maxCover, isRow, index);
        }

        startTime = System.nanoTime();

        for (int iter = 0; iter < numIterations; iter++) {
            board.getScore3(other, uncoveredRows, uncoveredColumns,
                    differingCells, lanes, maxCover, isRow, index);
        }

        // Stop the timer
        endTime = System.nanoTime();

        totalTime = endTime - startTime;

        System.out.println("Total time: " + totalTime + " ns");
        System.out.println("Average time per call: " +
                ((double) totalTime / numIterations) + " ns");
    }


}
