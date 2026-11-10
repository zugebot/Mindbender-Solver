package Boards;

// native imports
import java.util.*;

// custom imports
import support.Colors;


public class BoardOperations {

    public static byte BOARD_SIZE = Board.BOARD_SIZE;

    public static String GREEN = Colors.Green;
    public static String MAGENTA = Colors.Magenta;
    public static String WHITE = Colors.White;
    public static String RESET = Colors.Reset;

    public static int MAX_SIZE;
    public static boolean DEBUG;
    public static int UNIQUE_GUESS;

    private byte i = 0;
    private short s0 = 0;
    private byte b0 = 0;
    private byte b1 = 0;
    private byte b2 = 0;
    private byte b3 = 0;
    private byte b4 = 0;
    private byte sect = 0;
    private byte amount = 0;


    public BoardOperations(int maxSize,
                           boolean debug,
                           int uniqueGuess) {
        BOARD_SIZE = Board.BOARD_SIZE;
        MAX_SIZE = maxSize;
        DEBUG = debug;
        UNIQUE_GUESS = uniqueGuess;
    }

    /**
     * easier to use extender FOR LATER ON OR SOMETHING
     **/
    public Board[] extend(Board[] board, Board solve, double time) {
        double start;

        start = System.nanoTime();
        Board[] boards = generateUnique(board, UNIQUE_GUESS);
        debug(start, "Generating...     [" + boards.length + "]");

        if (boards.length > MAX_SIZE) {

            start = System.nanoTime();
            sort(boards, solve);
            debug(start, "Sorting...        [" + "" + "]");

            start = System.nanoTime();
            boards = resizeArray(boards, MAX_SIZE);
            debug(start, "Resizing...       [" + boards.length + "]");

        }

        int end = boards.length - 1;
        System.out.println("   " +
                GREEN + "Candidates" + RESET +
                " Best" + MAGENTA + "[" +
                boards[0].getScoreString(solve) + "]" + RESET +
                " Worst" + MAGENTA + "[" +
                boards[end].getScoreString(solve) + "]" + RESET);

        return boards;
    }


    /**
     * Expands Itself
     **/
    public Board[] generateUnique(Board[] boardsInput, int multiple) {
        int count = 0;
        byte[] temp = new byte[BOARD_SIZE];
        Board[] boardArray = new Board[boardsInput.length * multiple];
        Set<Board> boardSet = new HashSet<>();

        for (Board board : boardsInput) {
            boardArray[count] = board.copy();
            for (byte[] move : board.getPossibleMoves()) {
                boardArray[count].doMove(temp, move);

                if (boardSet.contains(boardArray[count])) {
                    boardArray[count].undoMove(temp);
                } else {
                    boardSet.add(boardArray[count]);
                    boardArray[++count] = board.copy();
                }
            }
        }

        boardArray = resizeArray(boardArray, count);
        return boardArray;
    }

    /**
     * 1. Takes the board object and creates a hashmap out of it.
     * 2. iterates over solves, to see if the board is there
     **/
    public ArrayList<Board[]> findIntersections(Board[] boards, Board[] solves) {
        HashMap<Board, ArrayList<Board>> hash = new HashMap<>();

        for (Board board : boards) {
            hash.computeIfAbsent(board, k -> new ArrayList<>()).add(board);
        }

        // iterate over solves to see if any item is in the HashMap
        ArrayList<Board[]> pairs = new ArrayList<>(solves.length);
        ArrayList<Board> piece;

        for (Board solve : solves) {
            piece = hash.get(solve);

            if (piece != null) {
                for (Board board : piece) {
                    pairs.add(new Board[]{board, solve});
                }
            }
        }
        return pairs;
    }

    /**
     * Takes an input array, returns a new array with length count
     **/
    public Board[] resizeArray(Board[] boards, int count) {
        Board[] newBoards = new Board[count];
        System.arraycopy(boards, 0, newBoards, 0, count);
        return newBoards;
    }

    /**
     * Sorts an array of boards, against a passed solve argument
     **/
    public void sort(Board[] boards, Board solve) {
        b0 = 0; b1 = 0; b2 = 0; b3 = 0; b4 = 0;
        for (Board board : boards) {
            board.updateHeuristic(s0, b0, b1, b2, b3, b4, solve);
        }
        Arrays.sort(boards, (a, b) -> Short.compare(b.score, a.score));
    }

    /**
     * An easier way to parse the move string
     * **/
    public String getMovesetString(Board[] pair) {
        return pair[0].getMoves() + pair[1].getMovesReversed();
    }

    static public void debug(double time, String string) {
        String debugInfo = GREEN + "   [%." + 5 + "fs]" + WHITE + " %s" + RESET;
        if (!DEBUG) {
            return;
        }
        double newTime = (System.nanoTime() -  time) / 1_000_000_000;
        System.out.printf(debugInfo + "%n", newTime, string);
    }


}
