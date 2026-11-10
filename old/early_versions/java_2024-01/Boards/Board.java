package Boards;

// native imports

import support.Bytes;
import support.Colors;

import java.util.Random;





public class Board {

    public static final byte BOARD_SIZE = 6;
    public static final byte MOVE_COUNT = 11;
    private static final short MODIFIER_SCORE_1 = 36;
    private static final short MODIFIER_SCORE_2 = 1;


    private static final Random RANDOM = new Random();
    private static final byte[][] MODULO_TABLE = generateModuloTable();
    public static final byte[][][] MOVESET_TABLE = generateMovesetTable();
    /**
     *  1. direction (0-1)
     *  <br>
     *  2. highSect (0-5)
     *  <br>
     *  3. the row of moves
     *  <br>
     *  4. each move object [direction, section, amount]
     */
    public static final byte[][][][] MOVE_TABLE = generateMoveTable();

    public byte[][] board = new byte[BOARD_SIZE][BOARD_SIZE];
    public byte[] moves = new byte[MOVE_COUNT];
    public byte highSect = 0;
    public byte moveNum = 0;
    public short score = 0;

    public Board() {}

    public Board(Board board) {
        copy(board);
    }

    public Board(String string) {
        load(string);
    }

    /**
     * returns the first 2 bytes of move at index
     **/
    public final byte getDirection(byte index) {
        return (byte) (moves[index] & 3);
    }

    public final byte getDirection() {
        return getDirection(moveNum);
    }

    public byte getAmount(byte index) {
        return(byte) ((moves[index] >> 5) & 7);
    }

    public final byte getAmount() {
        return getAmount(moveNum);
    }

    public final byte getAntiAmount() {
        return (byte) (BOARD_SIZE - getAmount(moveNum));
    }

    public final byte getAntiAmount(byte index) {
        return (byte) (BOARD_SIZE - getAmount(index));
    }

    public final byte getSection(byte index) {
        return (byte) ((moves[index] >> 2) & 7);
    }

    public final byte getSection() {
        return getSection(moveNum);
    }

    public final void setDirection(byte val) {
        moves[moveNum] =(byte) ((moves[moveNum] & 252) | val);
    }

    private static byte[][] generateModuloTable() {
        byte i, j;
        byte[][] table = new byte[BOARD_SIZE][BOARD_SIZE];
        for (i = 0; i < BOARD_SIZE; i++) {
            for (j = 0; j < BOARD_SIZE; j++) {
                table[i][j] = (byte) ((i + j) % BOARD_SIZE);
            }
        }
        return table;
    }

    private static byte getModulo(byte x, byte y) {
        return MODULO_TABLE[x][y];
    }

    private static byte[][][][] generateMoveTable() {
        byte[][][][] table = new byte[2][6][][];
        byte direction1, highsect, count, direction2, section, amount;
        int size;

        for (direction1 = 0; direction1 < 2; direction1++) {
            for (highsect = 0; highsect < 6; highsect++) {
                count = 0;
                size = 30 + 5 * (6 - highsect);
                table[direction1][highsect] = new byte[size][3];

                for (direction2 = 0; direction2 < 2; direction2++) {
                    // System.out.println();
                    if (direction1 == direction2) {
                        // Only does highsect+ moves, ignores lower
                        for (section = highsect; section < 6; section++) {
                            for (amount = 1; amount < 6; amount++) {
                                // System.out.println(count + "/" + size + " " + highsect + " " + section + " " + amount);
                                table[direction1][highsect][count++] = new byte[] {direction2, section, amount};
                            }
                        }
                        // This does all 30 possible moves
                    } else {
                        for (section = 0; section < 6; section++) {
                            for (amount = 1; amount < 6; amount++) {
                                // System.out.println(count + "/" + size + " " + highsect + " " + section + " " + amount);
                                table[direction1][highsect][count++] = new byte[]{direction2, section, amount};
                            }
                        }
                    }
                }
            }
        }
        return table;
    }

    public final byte[][] getPossibleMoves() {
        return MOVE_TABLE[getDirection()][highSect];
    }

    public static byte[][][] generateMovesetTable() {
        byte[][][] table = new byte[2][6][6];
        byte amount, section, direction, value;
        for (amount = 1; amount <= 5; amount++) {
            for (section = 0; section <= 5; section++) {
                for (direction = 0; direction <= 1; direction++) {
                    value = (byte) ((amount << 5) | (section << 2) | direction);
                    table[direction][section][amount] = value;
                }
            }
        }
        return table;
    }

    /**
     * allows loading a 36-character string of ints. also accepts letters.
     **/
    public final void load(String string) {
        if (string.isEmpty()) {
            return;
        }
        byte i;
        string = string.replace("\n", "").replace(" ", "");
        for (i = 0; i < 9; i++) {
            String letter = Character.toString("ROYGCBPW-".charAt(i));
            string = string.replace(letter, String.valueOf(i));
        }
        for (i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            board[i / BOARD_SIZE][i % BOARD_SIZE] =
                    (byte) Character.digit(string.charAt(i), 10);
        }
    }

    public final String getDirectionString(byte index) {
        return switch (getDirection(index)) {
            case 0 -> "R";
            case 1 -> "C";
            default -> "#";
        };
    }

    /**
     * get moveset from object and parse it into something readable.
     **/
    public final String getMoves() {
        StringBuilder moveset = new StringBuilder();
        for (byte i = 0; i < moveNum; i++) {
            moveset.append(getDirectionString(i));
            moveset.append(getSection(i));
            moveset.append(getAmount(i));
            moveset.append(" ");
        }
        return moveset.toString();
    }

    /**
     * gets reverse moveset from object and parse it into something
     * readable.
     **/
    public final String getMovesReversed() {
        byte val, i;
        StringBuilder moveset = new StringBuilder();
        for (i = moveNum; i > 0; i--) {
            val = Bytes.MINUS_ONE_TABLE[i];
            moveset.append(getDirectionString(val));
            moveset.append(getSection(val));
            moveset.append(getAntiAmount(val));
            if (i != 1) {moveset.append(" ");}
        }
        return moveset.toString();
    }


    /**
     * print colored board to cmd.
     * <br>
     * also uses a title.
     **/
    public final void print(String header) {
        System.out.println(header);
        this.print();
    }

    /**
     * print colored board to cmd.
     **/
    public final void print() {
        byte x, y;
        StringBuilder string = new StringBuilder();
        for (y = 0; y < BOARD_SIZE; y++) {
            StringBuilder row = new StringBuilder();
            for (x = 0; x < BOARD_SIZE; x++) {
                String slot = Byte.toString(board[y][x]);
                slot = String.format("%1s", slot);
                row.append(Colors.getColor(board[y][x])).append(slot).append(" ");
            }
            string.append(row).append("\n");
        }
        string.append(Colors.Reset);
        System.out.println(string);
    }

    public final void setMoveROR(byte section, byte amount) {
        highSect = section;
        moves[moveNum++] = (byte) (amount << 5 | section << 2 | Bytes.b0);
        setDirection(Bytes.b0);
    }

    public final void setMoveCOD(byte section, byte amount) {
        highSect = section;
        moves[moveNum++] = (byte) (amount << 5 | section << 2 | Bytes.b1);
        setDirection(Bytes.b1);
    }

    /**
     *
     * @param temp byte[6] - for usage of rotating sections.
     */
    public final void undoMove(byte[] temp) {
        moves[moveNum--] = 0;
        doMove(temp,
                (byte) (moves[moveNum] & 3),
                (byte) ((moves[moveNum] >> 2) & 7),
                (byte) (BOARD_SIZE - ((moves[moveNum] >> 5) & 7))
        );
        moves[moveNum--] = 0;
    }

    /**
     * Reverses the Nth last moves, thus undoing it.
     * @param moveAmount the amount of moves to undo.
     */
    public final void undoMoves(int moveAmount) {
        if (moveAmount > moveNum) {
            moveAmount = moveNum;
        }
        byte[] temp = new byte[6];
        for (int i = 0; i < moveAmount; i++) {
            undoMove(temp);
        }
    }

    /**
     *
     * @param temp byte[6]
     * @param direction 0: ROR, 1: COD
     * @param section 0-5
     * @param amount 1-5
     */
    public final void doMove(byte[] temp,
                             byte direction,
                             byte section,
                             byte amount) {
        if (direction == 0) {
            ROR(temp, section, amount);
        } else {
            COD(temp, section, amount);
        }
    }

    public final void doMove(byte[] temp,
                             byte[] move) {
        if (move[0] == 0) {
            ROR(temp, move[1], move[2]);
        } else {
            COD(temp, move[1], move[2]);
        }
    }

    public final void ROR_NEW(byte[] temp, int row, int amount) {
        switch (amount) {
            case 1:
                temp[0] = board[row][5];
                board[row][5] = board[row][4];
                board[row][4] = board[row][3];
                board[row][3] = board[row][2];
                board[row][2] = board[row][1];
                board[row][1] = board[row][0];
                board[row][0] = temp[0];
            case 2:
                temp[0] = board[row][5];
                temp[1] = board[row][4];
                board[row][5] = board[row][3];
                board[row][4] = board[row][2];
                board[row][3] = board[row][1];
                board[row][2] = board[row][0];
                board[row][1] = temp[0];
                board[row][0] = temp[1];
            case 3:
                temp[0] = board[row][0];
                temp[1] = board[row][1];
                temp[2] = board[row][2];
                board[row][0] = board[row][3];
                board[row][1] = board[row][4];
                board[row][2] = board[row][5];
                board[row][3] = temp[0];
                board[row][4] = temp[1];
                board[row][5] = temp[2];
            case 4:
                temp[0] = board[row][0];
                temp[1] = board[row][1];
                board[row][0] = board[row][2];
                board[row][1] = board[row][3];
                board[row][2] = board[row][4];
                board[row][3] = board[row][5];
                board[row][4] = temp[0];
                board[row][5] = temp[1];
            case 5:
                temp[0] = board[row][0];
                board[row][0] = board[row][1];
                board[row][1] = board[row][2];
                board[row][2] = board[row][3];
                board[row][3] = board[row][4];
                board[row][4] = board[row][5];
                board[row][5] = temp[0];
        }


    }


    /** Rotates a row of the board "amount" rightwards.
     * <br>
     * This variant takes an input temp array as to minimize memory allocation.
     * <br>
     * This variant also passed the looping temp var.
     **/
    public final void ROR(byte[] temp, byte row, byte amount) {
        // write row to temp with offset
        temp[MODULO_TABLE[amount][0]] = board[row][0];
        temp[MODULO_TABLE[amount][1]] = board[row][1];
        temp[MODULO_TABLE[amount][2]] = board[row][2];
        temp[MODULO_TABLE[amount][3]] = board[row][3];
        temp[MODULO_TABLE[amount][4]] = board[row][4];
        temp[MODULO_TABLE[amount][5]] = board[row][5];
        // rewrite temp back to row
        System.arraycopy(temp, 0, board[row], 0, BOARD_SIZE);
        // update moveset
        setMoveROR(row, amount);
    }



    /**
     * Rotates a column of the board "amount" downwards.
     * <br>
     * This variant takes an input temp array as to minimize memory allocation.
     **/
    public final void COD(byte[] temp, byte column, byte amount) {
        // store the row in temp variable
        temp[MODULO_TABLE[0][amount]] = board[0][column];
        temp[MODULO_TABLE[1][amount]] = board[1][column];
        temp[MODULO_TABLE[2][amount]] = board[2][column];
        temp[MODULO_TABLE[3][amount]] = board[3][column];
        temp[MODULO_TABLE[4][amount]] = board[4][column];
        temp[MODULO_TABLE[5][amount]] = board[5][column];
        // rewrite the row back with offset
        board[0][column] = temp[0];
        board[1][column] = temp[1];
        board[2][column] = temp[2];
        board[3][column] = temp[3];
        board[4][column] = temp[4];
        board[5][column] = temp[5];
        // update moveset
        setMoveCOD(column, amount);
    }

    public final boolean isGoalState(Board solve) {
        return equals(solve);
    }


    public final void updateHeuristic(short s0, byte b0, byte b1, byte b2,
                                byte b3, byte b4, Board solve) {
        score =(short) (
                getScore1(solve) +
                getScore2(s0, b0, b1, b2, b3, b4, solve)
        );
    }

    public final short getHeuristic() {
        return score;
    }

    /**
     * returns an int count for how many points == between.
     **/
    public final short getScore1(Board other) {
        return (short) (MODIFIER_SCORE_1 * (
                        ((board[0][0] == other.board[0][0]) ? 1 : 0) +
                        ((board[0][1] == other.board[0][1]) ? 1 : 0) +
                        ((board[0][2] == other.board[0][2]) ? 1 : 0) +
                        ((board[0][3] == other.board[0][3]) ? 1 : 0) +
                        ((board[0][4] == other.board[0][4]) ? 1 : 0) +
                        ((board[0][5] == other.board[0][5]) ? 1 : 0) +
                        ((board[1][0] == other.board[1][0]) ? 1 : 0) +
                        ((board[1][1] == other.board[1][1]) ? 1 : 0) +
                        ((board[1][2] == other.board[1][2]) ? 1 : 0) +
                        ((board[1][3] == other.board[1][3]) ? 1 : 0) +
                        ((board[1][4] == other.board[1][4]) ? 1 : 0) +
                        ((board[1][5] == other.board[1][5]) ? 1 : 0) +
                        ((board[2][0] == other.board[2][0]) ? 1 : 0) +
                        ((board[2][1] == other.board[2][1]) ? 1 : 0) +
                        ((board[2][2] == other.board[2][2]) ? 1 : 0) +
                        ((board[2][3] == other.board[2][3]) ? 1 : 0) +
                        ((board[2][4] == other.board[2][4]) ? 1 : 0) +
                        ((board[2][5] == other.board[2][5]) ? 1 : 0) +
                        ((board[3][0] == other.board[3][0]) ? 1 : 0) +
                        ((board[3][1] == other.board[3][1]) ? 1 : 0) +
                        ((board[3][2] == other.board[3][2]) ? 1 : 0) +
                        ((board[3][3] == other.board[3][3]) ? 1 : 0) +
                        ((board[3][4] == other.board[3][4]) ? 1 : 0) +
                        ((board[3][5] == other.board[3][5]) ? 1 : 0) +
                        ((board[4][0] == other.board[4][0]) ? 1 : 0) +
                        ((board[4][1] == other.board[4][1]) ? 1 : 0) +
                        ((board[4][2] == other.board[4][2]) ? 1 : 0) +
                        ((board[4][3] == other.board[4][3]) ? 1 : 0) +
                        ((board[4][4] == other.board[4][4]) ? 1 : 0) +
                        ((board[4][5] == other.board[4][5]) ? 1 : 0) +
                        ((board[5][0] == other.board[5][0]) ? 1 : 0) +
                        ((board[5][1] == other.board[5][1]) ? 1 : 0) +
                        ((board[5][2] == other.board[5][2]) ? 1 : 0) +
                        ((board[5][3] == other.board[5][3]) ? 1 : 0) +
                        ((board[5][4] == other.board[5][4]) ? 1 : 0) +
                        ((board[5][5] == other.board[5][5]) ? 1 : 0)));
    }

    /** takes 9 bytes of memory
     * byte[] scores is a byte[6], the only reason I pass it is to
     * avoid creating then destroying it 10 million times **/
    public final short getScore2(short scoreTotal, byte scoreMax, byte i,
                                 byte sect, byte offset, byte currentScore,
                                 Board other) {
        scoreTotal = 0;
        // Calculate score both horizontally and vertically
        for (sect = 0; sect < BOARD_SIZE; sect++) {
            scoreMax = 0;
            // Calculate offsets
            for (offset = 0; offset < BOARD_SIZE; offset++) {
                currentScore = 0;
                for (i = 0; i < BOARD_SIZE; i++) {
                    if (board[sect][getModulo(i, offset)] == other.board[sect][i]) {
                        currentScore++;
                    }
                    if (board[getModulo(i, offset)][sect] == other.board[i][sect]) {
                        currentScore++;
                    }
                }
                // Update the maximum score if necessary
                if (currentScore > scoreMax) {
                    scoreMax = currentScore;
                }
            }
            scoreTotal += scoreMax;
        }

        return (short)(scoreTotal * MODIFIER_SCORE_2);
    }


    /**
     * Calculates the minimum number of rows or columns (lanes) that need to be changed
     * to make this board identical to another board. A lane can either be a row or a column.
     * <br>
     * @param other The other board to compare with.
     * @return The minimum number of lanes that need to be changed.
     * <br><br>
     * This method first identifies all cells where this board differs from the other board.
     * It then repeatedly selects the row or column with the most uncovered differing cells
     * and marks it as covered. This process continues until all differing cells are covered.
     * <br><br>
     * The result is the minimum number of lanes that need to be changed to make this board
     * identical to the other board. This number is also the total number of rows and columns
     * that have been marked as covered during the process.
     * <br><br>
     * The time complexity of this method is O(n^2) and the space complexity is O(n),
     * where n is the side length of the board (6 in this case).
     * <br><br>
     * This method is optimized for both speed and memory usage, and it should work well
     * even for larger boards.
     */
    public final int getScore3(
            Board other,
            int[] uncoveredRows,
            int[] uncoveredColumns,
            int differingCells,
            int lanes,
            int maxCover,
            boolean isRow,
            int index) {
        // Find all differing cells and update the counts in uncoveredRows
        // and uncoveredColumns
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                if (this.board[i][j] != other.board[i][j]) {
                    uncoveredRows[i]++;
                    uncoveredColumns[j]++;
                    differingCells++;
                }
            }
        }
        // While there are still uncovered differing cells
        while (differingCells > 0) {
            maxCover = 0;
            isRow = false;
            index = -1;
            // Find the row or column that covers the most uncovered
            // differing cells
            for (int i = 0; i < 6; i++) {
                if (uncoveredRows[i] > maxCover) {
                    maxCover = uncoveredRows[i];
                    isRow = true;
                    index = i;
                }
                if (uncoveredColumns[i] > maxCover) {
                    maxCover = uncoveredColumns[i];
                    isRow = false;
                    index = i;
                }
            }

            // Cover the chosen row or column and update the counts in
            // uncoveredRows and uncoveredColumns
            if (isRow) {
                differingCells -= uncoveredRows[index];
                uncoveredRows[index] = 0;
                for (int j = 0; j < 6; j++) {
                    if (this.board[index][j] != other.board[index][j] &&
                            uncoveredColumns[j] > 0) {
                        uncoveredColumns[j]--;
                    }
                }
            } else {
                differingCells -= uncoveredColumns[index];
                uncoveredColumns[index] = 0;
                for (int j = 0; j < 6; j++) {
                    if (this.board[j][index] != other.board[j][index] &&
                            uncoveredRows[j] > 0) {
                        uncoveredRows[j]--;
                    }
                }
            }

            lanes++;
        }

        return lanes;
    }

    public final int getScore3(Board other) {
        int[] uncoveredRows = new int[6];
        int[] uncoveredColumns = new int[6];
        int differingCells = 0;
        int lanes = 0;
        int maxCover = 0;
        boolean isRow = false;
        int index = -1;
        return getScore3(other, uncoveredRows, uncoveredColumns,
                differingCells, lanes, maxCover, isRow, index);
    }


    
    public final String getScoreString(Board solve) {
        short s0 = 0; byte b0 = 0; byte b1 = 0;
        byte b2 = 0; byte b3 = 0; byte b4 = 0;

        int score1 = this.getScore1(solve);
        int score2 = this.getScore2(s0, b0, b1, b2, b3, b4, solve);
        return (score2 / MODIFIER_SCORE_2) + " " + score1;
    }


    /** returns a copy of the board object. Duh. **/
    public final Board copy() {
        // create new board object
        Board theBoard = new Board();
        // copy the board array
        System.arraycopy(board[0], 0, theBoard.board[0], 0, 6);
        System.arraycopy(board[1], 0, theBoard.board[1], 0, 6);
        System.arraycopy(board[2], 0, theBoard.board[2], 0, 6);
        System.arraycopy(board[3], 0, theBoard.board[3], 0, 6);
        System.arraycopy(board[4], 0, theBoard.board[4], 0, 6);
        System.arraycopy(board[5], 0, theBoard.board[5], 0, 6);
        // copies single byte objects
        theBoard.moveNum = moveNum;
        theBoard.highSect = highSect;
        // copy moveset
        if (moveNum > 0) {
            System.arraycopy(moves, 0, theBoard.moves, 0, moveNum);
        }
        return theBoard;
    }


    /** returns a copy of the board object.
     * This one is actually good and doesn't waste memory **/
    public final void copy(Board inputBoard) {
        // copy the board array
        System.arraycopy(board[0], 0, inputBoard.board[0], 0, 6);
        System.arraycopy(board[1], 0, inputBoard.board[1], 0, 6);
        System.arraycopy(board[2], 0, inputBoard.board[2], 0, 6);
        System.arraycopy(board[3], 0, inputBoard.board[3], 0, 6);
        System.arraycopy(board[4], 0, inputBoard.board[4], 0, 6);
        System.arraycopy(board[5], 0, inputBoard.board[5], 0, 6);
        // copies single byte objects
        inputBoard.moveNum = moveNum;
        inputBoard.highSect = highSect;
        // copy moveset
        System.arraycopy(moves, 0, inputBoard.moves, 0, MOVE_COUNT);
    }

    /** returns a copy of the board object.
     * This one is actually good and doesn't waste memory **/
    public final Board copy(Board[] boards, int index) {
        // create new board object
        boards[index] = new Board();
        // copy the board array
        System.arraycopy(board[0], 0, boards[index].board[0], 0, 6);
        System.arraycopy(board[1], 0, boards[index].board[1], 0, 6);
        System.arraycopy(board[2], 0, boards[index].board[2], 0, 6);
        System.arraycopy(board[3], 0, boards[index].board[3], 0, 6);
        System.arraycopy(board[4], 0, boards[index].board[4], 0, 6);
        System.arraycopy(board[5], 0, boards[index].board[5], 0, 6);
        // copies single byte objects
        boards[index].moveNum = moveNum;
        boards[index].highSect = highSect;
        // copy moveset
        System.arraycopy(moves, 0, boards[index].moves, 0, MOVE_COUNT);
        return boards[index];
    }

    public static void copy(Board board, Board[] boards, int index) {
        // create new board object
        boards[index] = new Board();
        // copy the board array
        System.arraycopy(board.board[0], 0, boards[index].board[0], 0, 6);
        System.arraycopy(board.board[1], 0, boards[index].board[1], 0, 6);
        System.arraycopy(board.board[2], 0, boards[index].board[2], 0, 6);
        System.arraycopy(board.board[3], 0, boards[index].board[3], 0, 6);
        System.arraycopy(board.board[4], 0, boards[index].board[4], 0, 6);
        System.arraycopy(board.board[5], 0, boards[index].board[5], 0, 6);
        // copies single byte objects
        boards[index].moveNum = board.moveNum;
        boards[index].highSect = board.highSect;
        // copy moveset
        System.arraycopy(board.moves, 0, boards[index].moves, 0, MOVE_COUNT);
    }


    /** Expands Board **/
    public final Board[] possibleBoards() {
        byte count = 0;
        Board[] boards = new Board[60];
        byte[] temp = new byte[BOARD_SIZE];
        for (byte[] move : MOVE_TABLE[0][0]) {
            copy(boards, count);
            boards[count].doMove(temp, move);
            count++;
        }
        return boards;
    }


    /** override for HashMap **/
    @Override
    public final int hashCode() {
        return (((((((((((((((((((((((((((((((((((17
                * 31 + board[0][0]) * 31 + board[0][1]) * 31 + board[0][2])
                * 31 + board[0][3]) * 31 + board[0][4]) * 31 + board[0][5])
                * 31 + board[1][0]) * 31 + board[1][1]) * 31 + board[1][2])
                * 31 + board[1][3]) * 31 + board[1][4]) * 31 + board[1][5])
                * 31 + board[2][0]) * 31 + board[2][1]) * 31 + board[2][2])
                * 31 + board[2][3]) * 31 + board[2][4]) * 31 + board[2][5])
                * 31 + board[3][0]) * 31 + board[3][1]) * 31 + board[3][2])
                * 31 + board[3][3]) * 31 + board[3][4]) * 31 + board[3][5])
                * 31 + board[4][0]) * 31 + board[4][1]) * 31 + board[4][2])
                * 31 + board[4][3]) * 31 + board[4][4]) * 31 + board[4][5])
                * 31 + board[5][0]) * 31 + board[5][1]) * 31 + board[5][2])
                * 31 + board[5][3]) * 31 + board[5][4]) * 31 + board[5][5];
    }

    /** override for HashMap **/
    @Override
    public final boolean equals(Object ob) {
        if (ob == this) {
            return true;
        }
        if (ob == null || ob.getClass() != getClass()) {
            return false;
        }
        Board other = (Board) ob;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != other.board[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }


    /** does count # of moves **/
    public final void randomMoves(int moveCount) {
        byte[] temp = new byte[6];
        byte direction, section, amount;
        for (int i = 0; i < moveCount; i++) {
            section = (byte) RANDOM.nextInt(5);
            amount  = (byte) RANDOM.nextInt(1, 5);
            direction = (byte) RANDOM.nextInt(2);
            doMove(temp, direction, section, amount);
        }
    }

}
