when creating a list of 60 boards, I know that




actions 00-04: share top [---] bottom [5x6]
actions 05-09: share top [1x6] bottom [4x6]
actions 10-14: share top [2x6] bottom [3x6]
actions 15-19: share top [3x6] bottom [2x6]
actions 20-24: share top [4x6] bottom [1x6]
actions 25-29: share top [5x6] bottom [---]

actions 30-34: share left [---] right [5x6]
actions 35-39: share left [1x6] right [4x6]
actions 40-44: share left [2x6] right [3x6]
actions 45-49: share left [3x6] right [2x6]
actions 50-54: share left [4x6] right [1x6]
actions 55-59: share left [5x6] right [---]

so instead of making a list of 60*sizeof(Board), I COULD do
struct {
    Board originalBoard
    uint64_t move_mask[60];
    uint64_t move_val[60];
}

I would need to store them in a way as to minimize memory usage, 
without forsaking ease of generating more boards from it.

the issue with creating 60 boards from one board right now,
is that for each board, I have to mask the original board state,
such that I can fit in the tiny change.







how about this?

# # - - # -
- - - # - #
# - # # # -
- # # # # -
- - - # # -
- # # # - -


for a series of moves:

current
 |  next
 |   |
 v   v
[R].[C].
[C].[R].
Pos [R, C]

so if a row that is moved through a column that will move next, if the 
1. intersection of that row and column
2. the original cell of the row (before either move)
3. the new cell of the column (after the 2 moves)

if the color at their intersection [i] does not change,
that means the moves [R] and [C] can swap.

if this is the case, how can I ensure only
move1, move2 occurs, and never
move2, move1?

well, if it catches that the next two moves:
1. satisfy the intersection rule
2. is a [C] followed by a [R]
3. it can skip it


given the actions:    0   1   2   3
[54].[19].[32].[6]: "C45 R34 C03 R12"

axn  bym
C45, R34

a = action1 / 30
x = (action1 % 30) / 5
n = action1 % 5

b = action2 / 30
y = (action2 % 30) / 5
m = action2 % 5



if a != b:
    if a != 0: // 'R'
        int x = (action1 % 30) / 5;
        int y = (action2 % 30) / 5;
        int m = 1 + action1 % 5;
        int n = 1 + action2 % 5;
        if board.intersectionRC(x, y, m, n):
            continue;

       |  num - 3 | num - 3 & 0b11
-3 101 |          | 
-2 110 |          | 
-1 111 |          | 
 0 000 | 101      | 001 = 1
 1 001 | 110      | 010 = 2
 2 010 | 111      | 011 = 3
 3 011 | 000      | 000 = 0
 4 100 | 001      | 001 = 1
 5 101 | 010      | 010 = 2






















