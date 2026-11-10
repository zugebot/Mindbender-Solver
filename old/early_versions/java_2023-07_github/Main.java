// Jerrin Shirks

// native imports

import Boards.Board;
import Boards.BoardOperations;
import Chuzzle.Levels;
import support.Colors;
import support.Time;

import java.io.*;
import java.util.*;



public class Main {

    public static Levels levels = new Levels();
    public static final boolean isWindows = true;
    public static final boolean debug = true;

    public static void main(String[] args) throws IOException {

        // starting details
        String level_name = "4-3";
        Board board = levels.b4_3;
        Board solve = levels.s4_3;
        /*
        Psychic
        6-3 paiN
        6-4 paiN
        9-3 paiN
         */
        boolean isFat = false;
        byte[] allowed = {0, 1, 2, 3, 4, 5};
        String toAdd = "";  // ""R31 U21 ";
        int toAddDepth = 0;

        /*
        Board board = new Board(
                "R W W W W W" +
                "W W W R W R" +
                "W W W R R W" +
                "R W R R R W" +
                "W W R R W W" +
                "W R W R R R");

        Board solve = new Board(
                "W W W W W W" +
                "W R R R R W" +
                "W R W W R W" +
                "W R W R R W" +
                "W R W W W W" +
                "W R R R R R");
        */

        board.print("Boards");
        solve.print("Solve");

        // starting states
        int current_depth = 0;
        int maximum_depth = 20;
        int maximum_size = 300000;

        BoardOperations ops = new BoardOperations(maximum_size, debug);
        double start = System.currentTimeMillis();
        String side;
        ArrayList<Board[]> pairs;


        // initialize board + solve
        Board[] boards = new Board[1];
        boards[0] = board;
        Board[] solves = new Board[1];
        solves[0] = solve;

        // extends each one, alternating
        while (current_depth < maximum_depth) {
            current_depth++;

            side=new String[]{"boards", "solves"}[current_depth % 2];

            if (current_depth > 2) {
                System.out.println(Colors.Magenta + "\n[" + current_depth + "]" +
                        Colors.Reset + " Attempting search [" + side + "]");
            }

            if (current_depth == 1) { // if it is the first move
                System.out.println(Colors.Magenta + "[1]" + Colors.Reset + " Generating boards.");
                boards = ops.generate(board);
            } else if  (current_depth == 2) { // if it is the second move
                System.out.println(Colors.Magenta + "[2]" + Colors.Reset + " Generating Solves.");
                solves = ops.generate(solve);

            } else if ((current_depth % 2) == 0) {

                if (isFat) {
                    boards = ops.extendFat(boards, solve, start, allowed);
                } else {
                    boards = ops.extend(boards, solve, start);
                }
            } else {
                if (isFat) {
                    solves = ops.extendFat(solves, board, start, allowed);
                } else {
                    solves = ops.extend(solves, board, start);
                }
            }

            // finds matches, or "pairs"
            pairs = ops.find_intersection(boards, solves);

            // nothing found
            if (pairs.size() == 0) {
                System.out.println(Colors.Green + "   [" + Time.time(start) +
                        "s]" + Colors.Red + " no solutions found" + Colors.Reset);
            // if solutions found
            } else {
                int depth = current_depth + toAddDepth;

                System.out.println("\n" + Colors.CyanBold + "**End Details**" + Colors.Reset);
                System.out.println("Puzzle Title  : " + Colors.Cyan + level_name + Colors.Reset);
                System.out.println("Moves To Solve: " + Colors.Cyan + depth + Colors.Reset);
                System.out.println("Solution Count: " + Colors.Cyan + pairs.size() + Colors.Reset);
                System.out.println("Solution Time : " + Colors.Cyan + Time.time(start) +  "s" + Colors.Reset);

                String path;
                if (isWindows) {
                    path = "C:\\Users\\jerrin\\IdeaProjects\\Chuzzle Solver\\src\\levels\\";
                } else {
                    path = "levels\\";
                }


                String filename = path + level_name + "_c" + depth + ".txt";
                FileWriter out = new FileWriter(filename);

                // writes all the solutions to a file
                for (int i = 0; i < pairs.size(); i++) {
                    out.write(toAdd + ops.getMovesetString(pairs.get(i)));
                    if (i != pairs.size() - 1) {out.write("\n");}
                }
                out.flush();
                out.close();
                System.exit(0);
            } // end of is solution?

        } // end of while loop


    } // end of main


    /*
    public static void main(String[] args) throws IOException {
        BoardOperations ops = new BoardOperations();
        Levels levels = new Levels();

        starting details
        String level_name = "8-1";
        Board board = levels.b8_1;
        Board solve = levels.s8_1;
        int depth1 = 5;
        int depth2 = 5;
        int max_size = 150000;
        boolean debug = true;
        double start = System.currentTimeMillis();

        board.print("Board");
        solve.print("Solve");

        // expand board
        Board[] boards = ops.extendTo(board, depth1, solve, max_size, debug);
        System.out.println("(board) Took " + time(start) + "s to iterate to depth " + depth1);
        System.out.println("(boards) Ending Size: " + boards.length + "\n");

        // expand solve
        Board[] solves = ops.extendTo(solve, depth2, board, max_size, debug);
        System.out.println("(solve) Took " + time(start) + "s to iterate to depth " + depth2);
        System.out.println("(solves) Ending Size: " + solves.length + "\n");

        // report final time
        System.out.println("Total Time: " + time(start) + "s");

        // find intersection of boards and solves
        ArrayList<Board[]> pairs = ops.find_intersection(boards, solves);

        // report data
        String depth_str = String.valueOf(depth1 + depth2);
        System.out.println("Chuzzle Puzzle: " + level_name);

        // if no solutions found
        if (pairs.size() == 0) {
            System.out.println("There were 0 Solutions using a depth of " + depth_str + " ");

        // if solutions found
        } else {
            System.out.println("Moves To Solve: " + depth_str);
            System.out.println("Solution Count: " + pairs.size());
            String path = "C:\\Users\\jerrin\\IdeaProjects\\Chuzzle Solver\\src\\levels\\";
            String filename = path + level_name + "_c" + depth_str + ".txt";
            FileWriter out = new FileWriter(filename);

            for (int i = 0; i < pairs.size(); i++) {
                out.write(ops.getMovesetString(pairs.get(i)));
                if (i != pairs.size() - 1) {out.write("\n");}
            }

            out.flush();
            out.close();

        }
    }
     */
}






