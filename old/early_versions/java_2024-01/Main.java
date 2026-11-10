// Jerrin Shirks

// native imports

import Boards.Board;
import Boards.BoardOperations;
import Boards.Levels;
import support.Colors;
import support.Time;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;






public class Main {

    public static Levels levels = new Levels();
    public static final boolean debug = true;

    private static final String RESET = Colors.Reset;
    private static final String GREEN = Colors.Green;
    private static final String MAGENTA = Colors.Magenta;
    private static final String CYAN = Colors.Cyan;
    private static final String RED = Colors.Red;

    private static final int MAXIMUM_DEPTH = 14;
    private static final int MAXIMUM_SIZE = 100_000;

    public static void main(String[] args) throws IOException {
        String currentDirectory = System.getProperty("user.dir");
        System.out.println(currentDirectory);

        // starting details
        String level_name = "10-1";

        Board board = levels.b7_3;
        Board solve = levels.s7_3;

        Board[] boards = new Board[] {board};
        board.print("Mixed State");

        Board[] solves = new Board[] {solve};
        solve.print("Solve State");

        // starting states
        int depth = 0;
        BoardOperations ops = new BoardOperations(MAXIMUM_SIZE, debug, 45);

        double start = System.nanoTime();
        ArrayList<Board[]> pairs;
        String[] sides = new String[]{"boards", "solves"};

        // extends each one, alternating
        while (depth++ < MAXIMUM_DEPTH) {

            String side = sides[depth % 2];

            if (depth > 2) {
                System.out.println(MAGENTA + "\n[" + depth + "]" +
                        RESET + " Attempting search [" + side + "]");
            }

            if (depth == 1) { // if it is the first move
                System.out.println(MAGENTA + "[1]" + RESET + " Generating boards.");
                boards = boards[0].possibleBoards();

            } else if  (depth == 2) { // if it is the second move
                System.out.println(MAGENTA + "[2]" + RESET + " Generating Solves.");
                solves = solves[0].possibleBoards();

            } else if ((depth % 2) == 0) {
                boards = ops.extend(boards, solve, start);

            } else {
                solves = ops.extend(solves, board, start);
            }

            // finds matches, or "pairs"
            double lol = System.nanoTime();
            pairs = ops.findIntersections(boards, solves);
            BoardOperations.debug(lol, "Finding Intersection...");

            // nothing found
            if (pairs.size() == 0) {
                System.out.println(RED + "    No solutions found." + RESET);

            // if solutions found
            } else {


                String path = currentDirectory + "\\levels\\";

                String filename = path + level_name + "_c" + depth + ".txt";

                System.out.println("\n" + Colors.CyanBold + "**End Details**" + RESET);
                System.out.println("Puzzle Title  : " + CYAN + level_name + RESET);
                System.out.println("Moves To Solve: " + CYAN + depth + RESET);
                System.out.println("Solution Count: " + CYAN + pairs.size() + RESET);
                System.out.println("Solution Time : " + CYAN + Time.time(start) +  "s" + RESET);
                System.out.println("Out File Path : " + CYAN + filename + RESET);

                FileWriter out = new FileWriter(filename);

                // writes all the solutions to a file
                for (int i = 0; i < pairs.size(); i++) {
                    out.write(ops.getMovesetString(pairs.get(i)));
                    if (i != pairs.size() - 1) {
                        out.write("\n");
                    }
                }
                out.flush();
                out.close();
                System.exit(0);
                
            } // end of is solution?

        } // end of while loop

    } // end of main

} // end of Main class