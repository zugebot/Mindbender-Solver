cmake_minimum_required(VERSION 3.24)
project(MindbenderSolver)


set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


include_directories(${CMAKE_SOURCE_DIR}/)

file(GLOB_RECURSE MINDBENDER_SOLVER_SOURCES
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.cpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.hpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.tpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/utils/*.cpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/utils/*.hpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/utils/*.tpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/include.hpp")
file(GLOB_RECURSE INCLUDE_SOURCES
        "${CMAKE_SOURCE_DIR}/include/*.cpp"
        "${CMAKE_SOURCE_DIR}/include/*.hpp")


add_executable(MindbenderSolver
        ${MINDBENDER_SOLVER_SOURCES}
        ${INCLUDE_SOURCES}
        main.cpp
        # MindbenderSolver/programs/TestKnownSolutions.cpp
        # MindbenderSolver/programs/testHeuristics.cpp
        # MindbenderSolver/programs/testSorting.cpp
        # MindbenderSolver/programs/UnitTesting.cpp
)

# find_package(TBB REQUIRED)
# target_link_libraries(MindbenderSolver TBB::tbb)
