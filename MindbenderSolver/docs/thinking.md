

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
    MEMORIES_RIGHT = JVec(BUFFER_SIZE);
    for (i = 0; i < boards_size - boards_needed; i += boards_needed) {
        func<DEPTH>(MEMORIES_LEFT, BOARDS_RIGHT, [MEMORIES_RIGHT][&], [i], [i + boards_needed])
    }
    right_before = boards_size - boards_size % boards_needed
    func<DEPTH>(MEMORIES_LEFT, BOARDS_RIGHT, [MEMORIES_RIGHT][&], [right_before], [boards_size])


func<> should just be like the code from perms.tpp, but
- it takes a [vector][input] instead of a [board][input].
- func will:
  1. call the generator that fills [MEMORIES_RIGHT][&]
  2. sort [MEMORIES_RIGHT][&]
  3. find the intersection of [MEMORIES_LEFT][&] and [MEMORIES_RIGHT][&]
  4. using a thread lock, write any solutions found to a file
  5. using a thread lock, write to a file the current progress (so it could be resumed later).


Would it be faster if BOARDS_RIGHT were sorted first?
it could be sorted:
1. how they are normally sorted (easy to implement)
2. by the last move (would help prevent cache misses?)
3. by a score heuristic (yes or no)










given

std::vector<Memory>, where Memory == JVec<u8>,



































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






