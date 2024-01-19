cmake_minimum_required(VERSION 3.24)
project(MindbenderSolver)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_definitions(-USE_MOVES)

include_directories(${CMAKE_SOURCE_DIR}/)
file(GLOB_RECURSE MINDBENDER_SOLVER_SOURCES
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.cpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.hpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.c"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.h"
)

add_executable(MindbenderSolver
        fib.cpp
        ${MINDBENDER_SOLVER_SOURCES}
        fib.cpp
)
