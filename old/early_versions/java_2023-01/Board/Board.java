package Board;

// native imports
import java.util.Arrays;
import java.util.Random;

// custom imports
import support.Colors;





public class Board {

    public byte[][] board = new byte[6][6];
    public byte[] moves = new byte[11];
    public byte high_sect = 0;
    public byte move_num = 0;
    public short score = 0;

    // 2 0-6 values byte shifted
    // public byte fat_loc = 0;


    // constructors
    public Board() {}
    public Board(String string) {
        load(string);
    }


    /** allows loading a 36-character string of ints. also accepts letters. **/
    public void load(String str) {
        if (str.equals("")) {return;}
        byte i;
        str = str.replace("\n", "").replace(" ", "");
        // replaces letters for number chars
        for (i = 0; i < 9; i++) {
            String letter = Character.toString("ROYGCBPW-".charAt(i));
            str = str.replace(letter, String.valueOf(i));
        }
        // writes the string to board
        for (i = 0; i < 36; i++) {
            board[i / 6][i % 6] = (byte)Character.getNumericValue(str.charAt(i));
        }
    }



    /** returns the first 2 bytes of the last done move **/
    public byte getDirection() {
        return (byte) (moves[move_num] & 3);
    }
    public byte getDirection(byte index) {
        return (byte) (moves[index] & 3);
    }
    public void setDirection(byte val) {
        moves[move_num] =(byte) ((moves[move_num] & 252) | val);
    }
    public void setDirection(byte index, byte num) {
        moves[index] =(byte) ((moves[index] & 252) | num);
    }
    public String getDirectionString(byte index) {
        byte direction = getDirection(index);
        if (direction == 1) {
            return "R";
        } else if (direction == 2) {
            return "C";
        } else {
            return "#";
        }
    }
    /** returns the first 3-5 bytes of the last done move **/
    public byte getAmount() {
        return(byte) ((moves[move_num] & 224) >> 5);
    }
    public byte getAmount(byte index) {
        return(byte) ((moves[index] & 224) >> 5);
    }
    /** returns the first 6-8 bytes of the last done move **/
    public byte getSection() {
        return(byte) (byte) ((moves[move_num] & 28) >> 2);
    }
    public byte getSection(byte index) {
        return (byte) ((moves[index] & 28) >> 2);
    }
    /** yes. **/
    public void setMove(byte section, byte amount, byte dir) {
        moves[move_num] = (byte) (amount << 5 | section << 2 | dir);
    }


    /** get moveset from object and parse it into something readable. **/
    public String getMoves() {
        StringBuilder moveset =new StringBuilder();
        for (byte i = 0; i < move_num; i++) {
            moveset.append(getDirectionString(i));
            moveset.append(getSection(i));
            moveset.append(getAmount(i));
            moveset.append(" ");
        }
        return moveset.toString();
    }


    /** gets reverse moveset from object and parse it into something readable. **/
    public String getMovesReverse() {
        StringBuilder moveset =new StringBuilder();
        for (byte i = move_num; i > 0; i--) {
            moveset.append(getDirectionString((byte) (i - 1)));
            moveset.append(getSection((byte) (i - 1)));
            moveset.append((byte) (6 - getAmount((byte) (i - 1))));
            if (i != 1) {moveset.append(" ");}
        }
        return moveset.toString();
    }


    /** print colored board to cmd. **/
    public void print(String header) {
        StringBuilder string =new StringBuilder();

        System.out.println(header);
        for (byte y = 0; y < 6; y++) {
            StringBuilder row =new StringBuilder();
            for (byte x = 0; x < 6; x++) {
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
    public void ROR(byte row, byte amount) {
        byte[] temp = new byte[6];
        byte i;
        // write row to temp with offset
        for (i = 0; i < 6; i++) {
            temp[(i + amount) % 6] = board[row][i];
        }
        // rewrite temp back to row
        for (i = 0; i < 6; i++) {
            board[row][i] = temp[i];
        }
        // update moveset
        high_sect = row;
        setMove(row, amount, (byte) 1);
        // made to help with iterating, if the last move was not ROR
        move_num++;
        // update the direction
        if (getDirection() != 1) {setDirection((byte) 1);}
    }

    /** Rotates a row of the board "amount" rightwards.
     * This variant takes an input temp array as to minimize memory allocation. **/
    public Board ROR(byte[] temp, byte row, byte amount) {
        byte i;
        // write row to temp with offset
        for (i = 0; i < 6; i++) {
            temp[(i + amount) % 6] = board[row][i];
        }
        // rewrite temp back to row
        // System.arraycopy(temp, 0, board[row], 0 , 6);
        for (i = 0; i < 6; i++) {board[row][i] = temp[i];}
        // update moveset
        high_sect = row;
        setMove(row, amount, (byte) 1);
        // made to help with iterating, if the last move was not ROR
        move_num++;
        // update the direction
        if (getDirection() != 1) {setDirection((byte) 1);}
        return this;
    }

    /** Rotates a column of the board "amount" downwards. **/
    public void COD(byte col, byte amount) {
        byte[] tempCol = new byte[6];
        byte i;
        // store the row in temp variable
        for (i = 0; i < 6; i++) {
            tempCol[(i + amount) % 6] = board[i][col];
        }
        // rewrite the row back with offset
        for (i = 0; i < 6; i++) {
            board[i][col] = tempCol[i];
        }
        tempCol = null;
        // update moveset
        high_sect = col;
        setMove(col, amount, (byte) 2);
        move_num++;
        // update the direction
        if (getDirection() != 2) { setDirection((byte) 2);}
    }

    /** Rotates a column of the board "amount" downwards.
     * This variant takes an input temp array as to minimize memory allocation. **/
    public Board COD(byte[] temp, byte col, byte amount) {
        byte i;
        // store the row in temp variable
        for (i = 0; i < 6; i++) {
            temp[(i + amount) % 6] = board[i][col];
        }
        // rewrite the row back with offset
        for (i = 0; i < 6; i++) {
            board[i][col] = temp[i];
        }
        // update moveset
        high_sect = col;
        setMove(col, amount, (byte) 2);
        move_num++;
        // update the direction
        if (getDirection() != 2) { setDirection((byte) 2);}
        return this;
    }


    /** Do I really need to cache the getScore method???
     * takes a byte[6] temp variable as to reduce memory nonsense. */
    public void updateScore(byte[] temp, Board solve) {
        score =(short) (getScore2(temp, solve) * 36 + this.getScore1(solve));
    }


    /** takes 9 bytes of memory
     * byte[] scores is a byte[6], the only reason I pass it is to avoid creating then destroying it 10 million times **/
    public short getScore2(byte[] scores, Board other) {
        byte score_total = 0;
        byte score_max;
        byte i;
        byte sect;
        byte offset;
        
        // calculate score horizontally
        for (sect = 0; sect < 6; sect++) {
            // reset data
            Arrays.fill(scores, (byte)0);
            score_max = 0;
            // calculate offsets
            for (offset = 0; offset < 6; offset++) {
                for (i = 0; i < 6; i++) {
                    if  (board[sect][(i + offset) % 6] == other.board[sect][i]) {
                        scores[offset]++;
                    }
                }
            }
            // find max value in score_max
            for (i = 0; i < 6; i++) {
                if (scores[i] > score_max) {
                    score_max = scores[i];
                }
            }
            score_total += score_max;
        }

        // calculate score vertically
        for (sect = 0; sect < 6; sect++) {
            // reset data
            Arrays.fill(scores, (byte)0);
            score_max = 0;
            // calculate offsets
            for (offset = 0; offset < 6; offset++) {
                for (i = 0; i < 6; i++) {
                    if  (board[(i + offset) % 6][sect] == other.board[i][sect]) {
                        scores[offset]++;
                    }
                }
            }
            // find max value in score_max
            for (i = 0; i < 6; i++) {
                if (scores[i] > score_max) {
                    score_max = scores[i];
                }
            }
            score_total += score_max;
        }
        return score_total;
    }


    /** returns an int count for how many points == between. **/
    public short getScore1(Board other) {
        short count = 0;
        for (byte y = 0; y < 6; y++) {
            for (byte x = 0; x < 6; x++) {
                if (board[y][x] == other.board[y][x]) {
                    count++;
                }
            }
        }
        return count;
    }
    
    
    public String getScoreString(Board solve) {
        String string = "";
        byte[] temp = new byte[6];
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
        Board new_board = new Board();
        // copy the board array
        for (byte i = 0; i < 6; i++) {
            System.arraycopy(board[i], 0, new_board.board[i], 0 , 6);
        }
        // copies single byte objects
        new_board.move_num = move_num;
        new_board.high_sect = high_sect;
        // copy moveset
            System.arraycopy(moves, 0, new_board.moves, 0 , 10);
        return new_board;
    }


    /** returns a copy of the board object.
     * This one is actually good and doesn't waste memory **/
    public void copy(Board[] boards, int count) {
        // create new board object
        boards[count] = new Board();
        // copy the board array
        for (byte i = 0; i < 6; i++) {
            System.arraycopy(board[i], 0, boards[count].board[i], 0 , 6);
        }
        // copies single byte objects
        boards[count].move_num = move_num;
        boards[count].high_sect = high_sect;
        // copy moveset
        System.arraycopy(moves, 0, boards[count].moves, 0 , 9);
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
        for (byte i = 0; i < 6; i++) {
            if (!Arrays.equals(board[i], other.board[i])) {
                return false;
            }
        }
        return true;
    }


    /** does count # of moves **/
    public Board randomMoves(byte count) {
        Random rd = new Random();
        for (byte i = 0; i < count; i++) {
            byte section = (byte)rd.nextInt(5);
            byte amount  = (byte)rd.nextInt(1, 5);
            if (rd.nextInt(2) == 1) {
                ROR(section, amount);
            } else {
                COD(section, amount);
            }
        }
        return this;
    }

}
