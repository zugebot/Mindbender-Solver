package Boards;

// native imports
import java.util.Arrays;
import java.util.Random;

// custom imports
import support.Colors;
import support.Bytes;





public class Board {

    public static final byte BOARD_SIZE = 6;
    public static final byte MOVE_COUNT = 11;

    private static final Random RANDOM = new Random();
    private static final byte[][] MODULO_TABLE = generateModuloTable();

    public byte[][] board = new byte[BOARD_SIZE][BOARD_SIZE];
    public byte[] moves = new byte[MOVE_COUNT];
    public byte highSect = 0;
    public byte moveNum = 0;
    public short score = 0;

    public Board() {}

    public Board(String string) {
        load(string);
    }

    /** returns the first 2 bytes of move at index **/
    public byte getDirection(byte index) {
        return (byte) (moves[index] & 3);
    }

    public byte getAmount(byte index) {
        return(byte) ((moves[index] & 224) >> 5);
    }

    public byte getSection(byte index) {
        return (byte) ((moves[index] & 28) >> 2);
    }

    public void setDirection(byte val) {
        moves[moveNum] =(byte) ((moves[moveNum] & 252) | val);
    }

    public void setMove(byte section, byte amount, byte dir) {
        highSect = section;
        moves[moveNum] = (byte) (amount << 5 | section << 2 | dir);
        moveNum++;
        setDirection(dir);

    }


    private static byte[][] generateModuloTable() {
        byte[][] table = new byte[BOARD_SIZE][BOARD_SIZE];
        for (byte i = 0; i < BOARD_SIZE; i++) {
            for (byte j = 0; j < BOARD_SIZE; j++) {
                table[i][j] = (byte) ((i + j) % BOARD_SIZE);
            }
        }
        return table;
    }

    private static byte getModulo(byte x, byte y) {
        return MODULO_TABLE[x][y];

    }

    /** allows loading a 36-character string of ints. also accepts letters. **/
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
        byte direction = getDirection(index);
        return switch (direction) {
            case 1 -> "R";
            case 2 -> "C";
            default -> "#";
        };
    }

    /** get moveset from object and parse it into something readable. **/
    public final String getMoves() {
        StringBuilder moveset =new StringBuilder();
        for (byte i = 0; i < moveNum; i++) {
            moveset.append(getDirectionString(i));
            moveset.append(getSection(i));
            moveset.append(getAmount(i));
            moveset.append(" ");
        }
        return moveset.toString();
    }

    /** gets reverse moveset from object and parse it into something readable. **/
    public final String getMovesReversed() {
        byte val;
        StringBuilder moveset = new StringBuilder();
        for (byte i = moveNum; i > 0; i--) {
            val = (byte) (i - 1);
            moveset.append(getDirectionString(val));
            moveset.append(getSection(val));
            moveset.append((byte) (BOARD_SIZE - getAmount(val)));
            if (i != 1) {moveset.append(" ");}
        }
        return moveset.toString();
    }


    /** print colored board to cmd. **/
    public final void print(String header) {
        StringBuilder string =new StringBuilder();

        System.out.println(header);
        byte x;
        byte y;
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
    
    
    /** Rotates a row of the board "amount" rightwards. **/

    public final void ROR(byte row, byte amount) {
        byte[] temp = new byte[BOARD_SIZE];
        byte i;

        // write row to temp with offset
        for (i = 0; i < BOARD_SIZE; i++) {
            temp[MODULO_TABLE[amount][i]] = board[row][i];
        }

        // rewrite temp back to row
        for (i = 0; i < BOARD_SIZE; i++) {
            board[row][i] = temp[i];
        }

        setMove(row, amount, Bytes.b1);
        //moveNum++;
        //setDirection(ONE);
    }

    /** Rotates a row of the board "amount" rightwards.
     * This variant takes an input temp array as to minimize memory allocation.
     * This variant also passed the looping temp var. **/
    public final Board ROR(byte[] temp, byte iter, byte row, byte amount) {
        // write row to temp with offset
        for (iter = 0; iter < BOARD_SIZE; iter++) {
            temp[MODULO_TABLE[amount][iter]] = board[row][iter];
        }
        // rewrite temp back to row
        // System.arraycopy(temp, 0, board[row], 0 , 6);
        for (iter = 0; iter < BOARD_SIZE; iter++) {
            board[row][iter] = temp[iter];
        }
        // update moveset
        setMove(row, amount, Bytes.b1);
        return this;
    }

    /** Rotates a column of the board "amount" downwards. **/
    public final void COD(byte col, byte amount) {
        byte[] tempCol = new byte[BOARD_SIZE];
        byte i;
        // store the row in temp variable
        for (i = 0; i < BOARD_SIZE; i++) {
            tempCol[MODULO_TABLE[i][amount]] = board[i][col];
        }
        // rewrite the row back with offset
        for (i = 0; i < BOARD_SIZE; i++) {
            board[i][col] = tempCol[i];
        }
        // update moveset
        setMove(col, amount, Bytes.b2);
    }

    /** Rotates a column of the board "amount" downwards.
     * This variant takes an input temp array as to minimize memory allocation. **/
    public final Board COD(byte[] temp, byte iter, byte col, byte amount) {
        // store the row in temp variable
        for (iter = 0; iter < BOARD_SIZE; iter++) {
            temp[MODULO_TABLE[iter][amount]] = board[iter][col];
        }
        // rewrite the row back with offset
        for (iter = 0; iter < BOARD_SIZE; iter++) {
            board[iter][col] = temp[iter];
        }
        // update moveset
        setMove(col, amount, Bytes.b2);
        return this;
    }

    /** Do I really need to cache the getScore method???
     * takes a byte[6] temp variable as to reduce memory nonsense. */
    public void updateScore(byte[] temp, Board solve) {
        score =(short) (getScore2(temp, solve) * 36 + this.getScore1(solve));
    }

    public void updateScore(byte scoreTotal, byte scoreMax, byte i, byte sect,
                            byte offset, byte currentScore, Board solve) {
        score =(short) (getScore2(scoreTotal, scoreMax, i, sect, offset,
                currentScore, solve) * 36 + this.getScore1(solve));
    }

    /** Do I really need to cache the getScore method???
     * takes a byte[6] temp variable as to reduce memory nonsense. */
    public void updateScore(byte[] temp, byte tempX, byte tempY, Board solve) {
        score =(short) (getScore2(temp, solve) * 36 + this.getScore1(solve));
    }


    /** takes 9 bytes of memory
     * byte[] scores is a byte[6], the only reason I pass it is to
     * avoid creating then destroying it 10 million times **/
    public short getScore2(byte[] temp, Board other) {
        byte scoreTotal = 0;
        byte scoreMax;
        byte i;
        byte sect;
        byte offset;
        byte currentScore;

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

        return scoreTotal;
    }


    /** takes 9 bytes of memory
     * byte[] scores is a byte[6], the only reason I pass it is to
     * avoid creating then destroying it 10 million times **/
    public short getScore2(byte scoreTotal, byte scoreMax, byte i, byte sect,
                           byte offset, byte currentScore, Board other) {
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

        return scoreTotal;
    }


    /** returns an int count for how many points == between. **/
    public short getScore1(Board other) {
        short count = 0;
        for (byte y = 0; y < BOARD_SIZE; y++) {
            for (byte x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] == other.board[y][x]) {
                    count++;
                }
            }
        }
        return count;
    }


    /** returns an int count for how many points == between.
     * has temp variables for faster speeds
     * **/
    public short getScore1(Board other, byte tempX, byte tempY) {
        short count = 0;
        for (tempY = 0; tempY < BOARD_SIZE; tempY++) {
            for (tempX = 0; tempX < BOARD_SIZE; tempX++) {
                if (board[tempY][tempX] == other.board[tempY][tempX]) {
                    count++;
                }
            }
        }
        return count;
    }
    
    
    public String getScoreString(Board solve) {
        String string = "";
        byte[] temp = new byte[BOARD_SIZE];
        int score1 = this.getScore1(solve);
        int score2 = this.getScore2(temp, solve);
        string += Integer.toString(score2);
        string += " ";
        string += Integer.toString(score1);
        return string;
    }


    /** returns a copy of the board object. Duh. **/
    public Board copy() {
        // create new board object
        Board newBoard = new Board();
        // copy the board array
        for (byte i = 0; i < BOARD_SIZE; i++) {
            System.arraycopy(board[i], 0, newBoard.board[i], 0 , BOARD_SIZE);
        }
        // copies single byte objects
        newBoard.moveNum = moveNum;
        newBoard.highSect = highSect;
        // copy moveset
            System.arraycopy(moves, 0, newBoard.moves, 0 , 10);
        return newBoard;
    }

    /** returns a copy of the board object. Duh. **/
    public Board copy(byte iter) {
        // create new board object
        Board newBoard = new Board();
        // copy the board array
        for (iter = 0; iter < BOARD_SIZE; iter++) {
            System.arraycopy(board[iter], 0, newBoard.board[iter], 0 , BOARD_SIZE);
        }
        // copies single byte objects
        newBoard.moveNum = moveNum;
        newBoard.highSect = highSect;
        // copy moveset
        System.arraycopy(moves, 0, newBoard.moves, 0 , MOVE_COUNT);
        return newBoard;
    }


    /** returns a copy of the board object.
     * This one is actually good and doesn't waste memory **/
    public void copy(Board[] boards, int index) {
        // create new board object
        boards[index] = new Board();
        // copy the board array
        for (byte i = 0; i < BOARD_SIZE; i++) {
            System.arraycopy(board[i], 0, boards[index].board[i], 0 , BOARD_SIZE);
        }
        // copies single byte objects
        boards[index].moveNum = moveNum;
        boards[index].highSect = highSect;
        // copy moveset
        System.arraycopy(moves, 0, boards[index].moves, 0 , MOVE_COUNT);
    }

    /** returns a copy of the board object.
     * This one is actually good and doesn't waste memory **/
    public void copy(Board[] boards, byte i, int boardIndex) {
        // create new board object
        boards[boardIndex] = new Board();
        // copy the board array
        for (i = 0; i < BOARD_SIZE; i++) {
            System.arraycopy(board[i], 0, boards[boardIndex].board[i], 0, BOARD_SIZE);
        }
        // copies single byte objects
        boards[boardIndex].moveNum = moveNum;
        boards[boardIndex].highSect = highSect;
        // copy moveset
        System.arraycopy(moves, 0, boards[boardIndex].moves, 0 , 9);
    }




    /** override for HashMap **/
    @Override
    public int hashCode() {
        return Arrays.deepHashCode(board);
    }


    /** override for HashMap **/
    @Override
    public boolean equals(Object ob) {
        // comment the next two lines out when you want speed, they are kinda useless
        if (ob == this) {return true;}
        if (ob == null || ob.getClass() != getClass()) {return false;}
        Board other = (Board) ob;
        // the real beef is here
        for (byte i = 0; i < BOARD_SIZE; i++) {
            if (!Arrays.equals(board[i], other.board[i])) {
                return false;
            }
        }
        return true;
    }


    /** does count # of moves **/
    public Board randomMoves(byte moveCount) {
        for (byte i = 0; i < moveCount; i++) {
            byte section = (byte) RANDOM.nextInt(5);
            byte amount  = (byte) RANDOM.nextInt(1, 5);
            if (RANDOM.nextInt(2) == 1) {
                ROR(section, amount);
            } else {
                COD(section, amount);
            }
        }
        return this;
    }

}
