

Say I have a board that requires going to a depth of 11.
What is the best way to go about solving it?



Say I have

JVec<Memory> Left5 ====== JVec<Board> Right5

and I want

.                     <----->  JVec<Memory, BUFFER_SIZE> Right6Sub  <---                     .
. JVec<Memory> Left5  <----->  JVec<Memory, BUFFER_SIZE> Right7Sub  <---  JVec<Board> Right5 .
.                     <----->  JVec<Memory, BUFFER_SIZE> Right8Sub  <---                     .

how do I break these up into blocks such that one could start and stop a search?

Given a buffer size BUFFER_SIZE, given we want to go to a depth of DEPTH past 5 from the right,

given we have a table of possible sizes for how big a board --> depth will grow, such as
ARRAY_FROM_DEPTH = {60, 60^2, 60^3}

we will take 

so 

def FIND_SOLVES(MEMORIES_LEFT, BOARDS_RIGHT, DEPTH)
    ARRAY_FROM_DEPTH = {60, 60^2, 60^3}
    boards_needed = (BOARDS_RIGHT.size() / ARRAY_FROM_DEPTH[DEPTH])
    .
    for (i = 0; i < boards_size - boards_needed; i += boards_needed) {
        func<DEPTH>(MEMORIES_LEFT, BOARDS_RIGHT, [i], [i + boards_needed])
    }
    right_before = boards_size - boards_size % boards_needed
    func<DEPTH>(MEMORIES_LEFT, BOARDS_RIGHT, [right_before], [boards_size])












































ALU for HW

address space = number of locations (k = 2 ^ n locations)
addressibility: number of bits per location (byte-addressable)
m bits

k = 2^n tall, m wide

inputs / outputs for 
. multiplexor (with n select bits)
. n-bit decoder

build truth table for following logic circuit

know what
symbol triangle
symbol small circle
symbol square curved one side
symbol dash






