import Boards.Board;
import Boards.Levels;

public class TestScore3 {
    public static void main(String[] args) {
        Levels levels = new Levels();

        Board board = levels.b7_4.copy();
        Board solve = levels.b7_4.copy();

        int moves = 3;

        board.print();
        board.randomMoves(moves);
        board.print();

        //int count = board.getScore3(solve);

        //System.out.println("Moves: " + moves + "\n" +
        //        "Score3 -> " + count);

        /*
         board.undoMoves(3);
         board.print();
         System.out.println(board.getMoves());
        */

    }
}

