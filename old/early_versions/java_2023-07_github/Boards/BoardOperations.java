package Boards;

// native imports
import java.util.*;

// custom imports
import support.Colors;
import support.Time;



public class BoardOperations {

    public static byte BOARD_SIZE;

    public int myMaximumSize;
    public boolean myDebug;

    private byte iter = 0;


    public BoardOperations(int theMaximumSize, boolean theDebug) {
        BOARD_SIZE = Boards.Board.BOARD_SIZE;
        this.myMaximumSize = theMaximumSize;
        this.myDebug = theDebug;

    }


    /** Function to remove duplicates from a Board[] **/
    public Board[] removeDuplicates(Board[] boards) {
        Set<Board> boardSet = new HashSet<>(Arrays.asList(boards));
        Board[] filteredBoards = new Board[boardSet.size()];
        boardSet.toArray(filteredBoards);
        return filteredBoards;
    }


    /** Takes an input array, returns a new array with length count **/
    public Board[] resizeArray(Board[] boards, int count) {
        Board[] newBoards = new Board[count];
        System.arraycopy(boards, 0, newBoards, 0, count);
        return newBoards;
    }


    /** Sorts an array of boards, against a passed solve argument **/
    public void sort(Board[] boards, Board solve) {
        byte byte0 = 0; byte byte1 = 0; byte byte2 = 0;
        byte byte3 = 0; byte byte4 = 0; byte byte5 = 0;
        for (Board board : boards) {board.updateScore(
                byte0, byte1, byte2, byte3, byte4, byte5, solve);}
        Arrays.sort(boards,
                (Board a, Board b) -> b.score - a.score);
    }


    /** An easier way to parse the move string **/
    public String getMovesetString(Board[] pair) {
        return pair[0].getMoves() + pair[1].getMovesReversed();
    }


    /** 1. Takes the board object and creates a hashmap out of it.
        2. iterates over solves, to see if the board is there **/
    public ArrayList<Board[]> find_intersection(Board[] boards, Board[] solves) {
        HashMap<Board, ArrayList<Board>> hash = new HashMap<>();
        // if the board is not in the hashtable, add it
        for (Board Board : boards) {
            Board board = Board.copy(iter);
            // if the key (board array) is not in the hashmap
            if (hash.get(board) == null) {
                // create a new array and put it in the hashmap
                ArrayList<Board> temp = new ArrayList<>();
                temp.add(board);
                hash.put(board, temp);
            } else {
                // else, simply add the new board to the already existing array
                hash.get(board).add(board);
            }
        }
        // tell me the size
        // System.out.println("HashTable Size: " + hash.size());
        // iterate over solves to see if any item is in the HashMap
        ArrayList<Board[]> pairs = new ArrayList<>();

        for (Board Board : solves) {
            Board solve = Board.copy(iter);

            if (hash.get(solve) != null) {
                for (int j = 0; j < hash.get(solve).size(); j++) {

                    // create a list of solutions
                    Board[] pair = new Board[] {hash.get(solve).get(j), solve};
                    pairs.add(pair);
                }
            }
        }
        return pairs;
    }


    /** Expands Board **/
    public Board[] generate(Board boardInput) {
        Board[] boards = new Board[60];
        byte[] temp = new byte[BOARD_SIZE];
        byte count = 0;
        byte iter = 0;
        for (byte sect = 0; sect < BOARD_SIZE; sect++) {
            for (byte amount = 1; amount < BOARD_SIZE; amount++) {
                boards[count++] = boardInput.copy(iter).ROR(temp, iter, sect, amount);
                boards[count++] = boardInput.copy(iter).COD(temp, iter, sect, amount);
            }
        }
        return boards;
    }


    public void debug(double time, String string) {
        String debugInfo = Colors.Green + "   [%ss]" + Colors.White + " %s" + Colors.Reset;
        if (myDebug) {System.out.printf((debugInfo) + "%n", Time.time(time), string);}
    }


    /** easier to use extender FOR LATER ON OR SOMETHING **/
    public Board[] extend(Board[] board, Board solve, double time) {
        debug(time, "Generating...");
        Board[] boards = generate(board);
        debug(time, "Removing Duplicates...");
        boards = removeDuplicates(boards);
        if (boards.length > myMaximumSize) {
            debug(time, "Sorting...");
            sort(boards, solve);
            debug(time, "Resizing...");
            boards = resizeArray(boards, myMaximumSize);
        }
        debug(time, "Best Candidate: " + boards[0].getScoreString(solve));
        return boards;
    }


    /** Expands Itself **/
    public Board[] generate(Board[] boardsInput) {
        // technically I should allocate * 60,
        // but it never has that many unique boards
        int count = 0;
        int size = boardsInput.length * 55;

        Board[] boards = new Board[size];
        byte[] temp = new byte[BOARD_SIZE];
        byte sect;
        byte amount;
        byte iter = 0;

        // iterate over each board
        for (Board Board : boardsInput) {

            // if the last board used ROR.
            if (Board.getDirection(Board.moveNum) == 1) {
                // (ROR) only do it if it is above the boards high_sect
                for (sect = (byte) (Board.highSect + 1); sect < BOARD_SIZE; sect++) {
                    for (amount = 1; amount < BOARD_SIZE; amount++) {
                        Board.copy(boards, iter, count);
                        boards[count++].ROR(temp, iter, sect, amount);
                    }
                }
                // (COD) all sects count
                for (sect = 0; sect < BOARD_SIZE; sect++) {
                    for (amount = 1; amount < BOARD_SIZE; amount++) {
                        Board.copy(boards, iter, count);
                        boards[count++].COD(temp, iter, sect, amount);
                    }
                }
            }
            // if the last board used COD.
            if (Board.getDirection(Board.moveNum) == 2) {
                // (ROR) all sects count
                for (sect = 0; sect < BOARD_SIZE; sect++) {
                    for (amount = 1; amount < BOARD_SIZE; amount++) {
                        Board.copy(boards, iter, count);
                        boards[count++].ROR(temp, iter, sect, amount);

                    }
                }
                // (COD) only do it if it is above the boards high_sect
                for (sect = (byte) (Board.highSect + 1); sect < BOARD_SIZE; sect++) {
                    for (amount = 1; amount < BOARD_SIZE; amount++) {
                        Board.copy(boards, iter, count);
                        boards[count++].COD(temp, iter, sect, amount);
                    }
                }
            }

        }
        // end of iterating over each board
        boards = resizeArray(boards, count);
        return boards;
    }






    /** easier to use extender FOR LATER ON OR SOMETHING **/
    public Board[] extendFat(Board[] board, Board solve, double time, byte[] allowed) {
        debug(time, "Generating...");
        Board[] boards = generateFat(board, allowed);
        debug(time, "Removing Duplicates...");
        boards = removeDuplicates(boards);
        if (boards.length > myMaximumSize) {
            debug(time, "Sorting...");
            sort(boards, solve);
            debug(time, "Resizing...");
            boards = resizeArray(boards, myMaximumSize);
        }
        debug(time, "Best Candidate: " + boards[0].getScoreString(solve));
        return boards;
    }


    /** Expands Itself **/
    public Board[] generateFat(Board[] boardsInput, byte[] allowed) {
        // technically I should allocate * 60,
        // but it never has that many unique boards
        int count = 0;
        int size = boardsInput.length * 55;

        Board[] boards = new Board[size];
        byte[] temp = new byte[BOARD_SIZE];
        byte sect;
        byte amount;

        // iterate over each board
        for (Board Board : boardsInput) {

            // (COD) all sects count
            for (sect = 0; sect < allowed.length; sect++) {
                for (amount = 1; amount < BOARD_SIZE; amount++) {
                    Board.copy(boards, iter, count);
                    boards[count++].ROR(temp, iter, allowed[sect], amount);
                }
            }
            // (COD) only do it if it is above the boards high_sect
            for (sect = 0; sect < allowed.length; sect++) {
                for (amount = 1; amount < BOARD_SIZE; amount++) {
                    Board.copy(boards, iter, count);
                    boards[count++].COD(temp, iter, allowed[sect], amount);
                }
            }

        }
        // end of iterating over each board
        boards = resizeArray(boards, count);
        return boards;
    }
}
