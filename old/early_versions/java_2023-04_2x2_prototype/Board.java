import java.util.HashMap;

public class Board {

    public static final byte BOARD_SIZE = 6;
    public static HashMap<Byte, byte[][]> moveMap = new HashMap<>();

    private byte fatLoc = 0;

    public Board() {

    }

    public static void cacheFatData() {
        byte[][][][][] allMovable = new byte[BOARD_SIZE][BOARD_SIZE][2][BOARD_SIZE][];
        byte[][][][][] allAmounts = new byte[BOARD_SIZE][BOARD_SIZE][2][BOARD_SIZE][];
        byte[] defaultMovable = new byte[] {0, 1, 2, 3, 4, 5};
        byte[] defaultAmounts = new byte[] {1, 2, 3, 4, 5};
        byte[][] sectionsMovable = new byte[][] {
                {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5}, {0, 1, 2, 4, 5},
                {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};
        byte[][] sectionsAmounts = new byte[][] {
                {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 4, 5},
                {1, 3, 4, 5}, {2, 3, 4, 5}, {1, 2, 3, 4, 5}};

        for (byte _FatY = Bytes.b0; _FatY < BOARD_SIZE; _FatY++) {
            for (byte _FatX = Bytes.b0; _FatX < BOARD_SIZE; _FatX++) {
                for (byte _sect = Bytes.b0; _sect < Bytes.b2; _sect++) {
                    byte _fatLoc = (_sect == Bytes.b0) ? _FatX : _FatY;
                    for (byte _i = Bytes.b0; _i < BOARD_SIZE; _i++) {
                        allMovable[_FatX][_FatY][_sect][_i] =
                                (_i == _fatLoc || _i == _fatLoc + Bytes.b1) ?
                                sectionsMovable[_fatLoc] :
                                defaultMovable;
                        if (_i == _fatLoc) {
                            allAmounts[_FatX][_FatY][_sect][_i] = sectionsAmounts[_fatLoc];
                        } else if (_i == _fatLoc + Bytes.b1) {
                            allAmounts[_FatX][_FatY][_sect][_i] = Bytes.bNull;
                        } else {
                            allAmounts[_FatX][_FatY][_sect][_i] = defaultAmounts;
                        }
                    }
                }
            }
        }
        for (byte _FatX = Bytes.b0; _FatX < BOARD_SIZE - 1; _FatX++) {
            for (byte _FatY = Bytes.b0; _FatY < BOARD_SIZE - 1; _FatY++) {
                byte[][] extendedBoards = new byte[48][];
                byte _i = Bytes.b0;
                for (byte rc = 0; rc < 2; rc++) {
                    // this can be rewritten to only check the sections from allMovable
                    for (byte _sect = Bytes.b0; _sect < BOARD_SIZE; _sect++) {
                        for (byte amount : allAmounts[_FatX][_FatY][rc][_sect]) {
                            if (((rc == Bytes.b0) ? _FatX : _FatY) == _sect) {
                                extendedBoards[_i] = new byte[]{rc, _sect, amount, rc, (byte)(_sect + 1), amount};
                            } else {
                                extendedBoards[_i] = new byte[]{rc, _sect, amount};
                            }
                            _i++;
                        }
                    }
                }
                byte key = (byte) (((_FatX << Bytes.b3) & Bytes.b248) | (_FatY & Bytes.b7));
                moveMap.put(key, extendedBoards);
            }
        }


    }

    public void setFatXY(byte x, byte y) {
        fatLoc = (byte) (((x << Bytes.b3) & Bytes.b248) | (y & Bytes.b7));
    }

    public void setFatX(byte x) {
        fatLoc = (byte) ((x & Bytes.b31) | (x & Bytes.b7) << Bytes.b3);
    }

    public void setFatY(byte y) {
        fatLoc = (byte) ((this.fatLoc & Bytes.b248) | (y & Bytes.b7));
    }

    public byte getFatX() {
        return (byte) ((this.fatLoc >> Bytes.b3) & Bytes.b7);
    }

    public byte getFatY() {
        return (byte) (this.fatLoc & Bytes.b7);
    }

    public byte getFatByte() {
        return this.fatLoc;
    }

    public Board[] getPossibleMoves() {
        byte[][] moves = moveMap.get(this.fatLoc);

        Support.printArray(moves);
        return new Board[0];
    }


}