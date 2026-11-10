cmake_minimum_required(VERSION 3.24)
project(Chuzzle_Solver_C__)

set(CMAKE_CXX_STANDARD 17)

add_executable(Chuzzle_Solver_C__ main.cpp
        support/colors.h support/time.h support/bytes.h
        Board/Board.h Board/BoardOperations.h Board/Levels.h Board/BoardArray.h
        Board/BoardArray.cpp Board/Board.cpp Board/Board.cpp Board/BoardOperations.cpp
        Board/BoardHash.h support/bytes.cpp Board/BoardPair.cpp Board/BoardPair.h)
