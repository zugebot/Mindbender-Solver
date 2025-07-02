
/**
     * Calculates the minimum number of rows or columns (lanes) that need to be changed
     * to make this board identical to another board. A lane can either be a row or a column.
     * <br>
     * @param theOther The other board to compare with.
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
     * This method is optimized (ironic given the last line) for both speed and memory usage, and it should work well
     * even for larger boards.
     */

    public final int getScore3(
            final Board theOther,
            final int[] theUncoveredRows,
            final int[] theUncoveredColumns) {

        int differingCells = 0;
        int lanes = 0;
        int maxCover = 0;
        boolean isRow = false;
        int index = -1;


        // Find all differing cells and update the counts in uncoveredRows
        // and uncoveredColumns
        // for C++, I can instantly sum the rows, but setAllColors about the columns
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (this.myBoard[i][j] != theOther.myBoard[i][j]) {
                    theUncoveredRows[i]++;
                    theUncoveredColumns[j]++;
                    differingCells++;
                }
            }
        }
        // While there are still uncovered differing cells
        // worst case this should only occur 6 times?
        while (differingCells > 0) {
            maxCover = 0;
            isRow = false;
            index = -1;
            // Find the row or column that covers the most uncovered differing cells
            // in C++, can probably reinterpret the bytes to see if either can be skipped,
            // base which level of checking I am doing off of getScore1?
            // can be recoded to find the index and value of the max in both?
            for (int i = 0; i < BOARD_SIZE; i++) {
                if (theUncoveredRows[i] > maxCover) {
                    maxCover = theUncoveredRows[i];
                    isRow = true;
                    index = i;
                }
                if (theUncoveredColumns[i] > maxCover) {
                    maxCover = theUncoveredColumns[i];
                    isRow = false;
                    index = i;
                }
            }

            if (index == -1) {
                break;
            }

            // Cover the chosen row or column and update the counts in
            // uncoveredRows and uncoveredColumns
            // I could cache the results of getScore1 for this ptrType
            if (isRow) {
                differingCells -= theUncoveredRows[index];
                theUncoveredRows[index] = 0;
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (this.myBoard[index][j] != theOther.myBoard[index][j]
                    && theUncoveredColumns[j] > 0) {
                        theUncoveredColumns[j]--;
                    }
                }
            } else {
                differingCells -= theUncoveredColumns[index];
                theUncoveredColumns[index] = 0;
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (this.myBoard[j][index] != theOther.myBoard[j][index]
                    && theUncoveredRows[j] > 0) {
                        theUncoveredRows[j]--;
                    }
                }
            }

            lanes++;
        }

        return lanes;
    }
