package Board;

// native imports
import java.util.*;

// custom imports
import support.Colors;
import support.Time;



public class BoardOperations {

    public int maximum_size;
    public boolean debug;

    /** Function to remove duplicates from a Board[] **/
    public Board[] removeDuplicates(Board[] boards) {
        Set<Board> boardSet = new HashSet<>(Arrays.asList(boards)); // put boards in a set
        boards = null; // deallocate boards
        Board[] filteredBoards = new Board[boardSet.size()];
        boardSet.toArray(filteredBoards);
        return filteredBoards;
    }

    public BoardOperations(int _maximum_size, boolean _debug) {
        maximum_size = _maximum_size;
        debug = _debug;
    }


    /** Takes an input array, returns a new array with length count **/
    public Board[] resizeArray(Board[] boards, int count) {
        Board[] new_boards = new Board[count];
        System.arraycopy(boards, 0, new_boards, 0, count);
        return new_boards;
    }


    /** Sorts an array of boards, against a passed solve argument **/
    public void sort(Board[] boards, Board solve) {
        byte[] temp = new byte[6];
        for (Board board : boards) {board.updateScore(temp, solve);}
        Arrays.sort(boards,
                (Board a, Board b) -> b.score - a.score);
    }


    /** An easier way to parse the move string **/
    public String getMovesetString(Board[] pair) {
        return pair[0].getMoves() + pair[1].getMovesReverse();
    }
    

    /** 1. Takes the board object and creates a hashmap out of it.
        2. iterates over solves, to see if the board is there **/
    public ArrayList<Board[]> find_intersection(Board[] boards, Board[] solves) {
        HashMap<Board, ArrayList<Board>> hash = new HashMap<>();
        // if the board is not in the hashtable, add it
        for (Board Board : boards) {
            Board board = Board.copy();
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
            Board solve = Board.copy();

            if (hash.get(solve) != null) {
                for (int j = 0; j < hash.get(solve).size(); j++) {

                    // create a list of solutions
                    Board[] pair = new Board[2];
                    pair[0] = hash.get(solve).get(j);
                    pair[1] = solve;
                    pairs.add(pair);
                }
            }
        }
        return pairs;
    }


    /** Expands Board **/
    public Board[] generate(Board boardInput) {
        Board[] boards = new Board[60];
        byte[] temp = new byte[6];
        byte count = 0;
        for (byte sect = 0; sect < 6; sect++) {
            for (byte amount = 1; amount < 6; amount++) {
                boards[count++] = boardInput.copy().ROR(temp, sect, amount);
                boards[count++] = boardInput.copy().COD(temp, sect, amount);
            }
        }
        return boards;
    }


    public void debug(double time, String string) {
        String debugInfo = Colors.Green + "   [%ss]" + Colors.White + " %s" + Colors.Reset;
        if (debug) {System.out.printf((debugInfo) + "%n", Time.time(time), string);}
    }


    /** easier to use extender FOR LATER ON OR SOMETHING **/
    public Board[] extend(Board[] board, Board solve, double time) {
        debug(time, "Generating...");
        Board[] boards = generate(board);
        debug(time, "Removing Duplicates...");
        boards = removeDuplicates(boards);
        if (boards.length > maximum_size) {
            debug(time, "Sorting...");
            sort(boards, solve);
            debug(time, "Resizing...");
            boards = resizeArray(boards, maximum_size);
        }
        debug(time, "Best Candidate: " + boards[0].getScoreString(solve));
        return boards;
    }


    /** Expands Itself **/
    public Board[] generate(Board[] boardsInput) {
        // technically I should allocate * 60, but it never has that many unique boards
        int count = 0;
        int size = boardsInput.length * 55;

        Board[] boards = new Board[size];
        byte[] temp = new byte[6];
        byte sect;
        byte amount;

        // iterate over each board
        for (Board Board : boardsInput) {

            // if the last board used ROR.
            if (Board.getDirection() == 1) {
                // (ROR) only do it if it is above the boards high_sect
                for (sect=(byte) (Board.high_sect + 1); sect < 6; sect++) {
                    for (amount = 1; amount < 6; amount++) {
                        Board.copy(boards, count);
                        boards[count++].ROR(temp, sect, amount);
                    }
                }
                // (COD) all sects count
                for (sect = 0; sect < 6; sect++) {
                    for (amount = 1; amount < 6; amount++) {
                        Board.copy(boards, count);
                        boards[count++].COD(temp, sect, amount);
                    }
                }
            }
            // if the last board used COD.
            if (Board.getDirection() == 2) {
                // (ROR) all sects count
                for (sect = 0; sect < 6; sect++) {
                    for (amount = 1; amount < 6; amount++) {
                        Board.copy(boards, count);
                        boards[count++].ROR(temp, sect, amount);

                    }
                }
                // (COD) only do it if it is above the boards high_sect
                for (sect=(byte) (Board.high_sect + 1); sect < 6; sect++) {
                    for (amount = 1; amount < 6; amount++) {
                        Board.copy(boards, count);
                        boards[count++].COD(temp, sect, amount);
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
        if (boards.length > maximum_size) {
            debug(time, "Sorting...");
            sort(boards, solve);
            debug(time, "Resizing...");
            boards = resizeArray(boards, maximum_size);
        }
        debug(time, "Best Candidate: " + boards[0].getScoreString(solve));
        return boards;
    }


    /** Expands Itself **/
    public Board[] generateFat(Board[] boardsInput, byte[] allowed) {
        // technically I should allocate * 60, but it never has that many unique boards
        int count = 0;
        int size = boardsInput.length * 55;

        Board[] boards = new Board[size];
        byte[] temp = new byte[6];
        byte sect;
        byte amount;

        // iterate over each board
        for (Board Board : boardsInput) {


            // (COD) all sects count
            for (sect=0; sect < allowed.length; sect++) {
                for (amount=1; amount < 6; amount++) {
                    Board.copy(boards, count);
                    boards[count++].ROR(temp, allowed[sect], amount);
                }
            }

            // (COD) only do it if it is above the boards high_sect
            for (sect = 0; sect < allowed.length; sect++) {
                for (amount=1; amount < 6; amount++) {
                    Board.copy(boards, count);
                    boards[count++].COD(temp, allowed[sect], amount);
                }
            }

        }

        // end of iterating over each board
        boards = resizeArray(boards, count);
        return boards;
    }



















}
