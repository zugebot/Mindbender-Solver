cmake_minimum_required(VERSION 3.24)
project(MindbenderSolver)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


include_directories(${CMAKE_SOURCE_DIR}/)

file(GLOB_RECURSE MINDBENDER_SOLVER_SOURCES
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.cpp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/*.hpp")
file(GLOB_RECURSE INCLUDE_SOURCES
        "${CMAKE_SOURCE_DIR}/include/*.cpp"
        "${CMAKE_SOURCE_DIR}/include/*.hpp")


add_executable(MindbenderSolver
        ${MINDBENDER_SOLVER_SOURCES}
        ${INCLUDE_SOURCES}
        main.cpp
        # testHeuristics.cpp
        # testSorting.cpp
        # TestKnownSolutions.cpp
        # UnitTesting.cpp
)

# find_package(TBB REQUIRED)
# target_link_libraries(MindbenderSolver TBB::tbb)
