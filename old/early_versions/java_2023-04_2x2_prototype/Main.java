/**
 * Author: Jerrin Shirks
 * Date: 4/19/2023
 */

public class Main {
    public static void main(String[] args) {
        Board.cacheFatData();

        System.out.println("Possible Moves: 48\n");

        Board board = new Board();
        board.setFatXY(Bytes.b3, Bytes.b3);
        board.getPossibleMoves();

        /*
        for (byte key : Board.moveMap.keySet()) {
            System.out.println("Key " + key);
            Support.printArray(Board.moveMap.get(key));
        }
         */

    }
}