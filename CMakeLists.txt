cmake_minimum_required(VERSION 3.24)
project(MindbenderSolver)


# include(CheckCXXCompilerFlag)


find_library(BOOST_FOUND Boost)
if(BOOST_FOUND)
    message(STATUS "Boost found")
    add_compile_definitions(BOOST_FOUND)
else()
    message(STATUS "Boost not found")
endif()


find_library(TBB_FOUND TBB)
if(TBB_FOUND)
    message(STATUS "TBB found")
    add_compile_definitions(TBB_FOUND)
    find_package(TBB REQUIRED)
    target_link_libraries(MindbenderSolver TBB::tbb)
else()
    message(STATUS "TBB not found")
endif()


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static")



include_directories(${CMAKE_SOURCE_DIR}/)

file(GLOB_RECURSE MINDBENDER_SOLVER_SOURCES
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.*pp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/utils/*.*pp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/include.hpp")
file(GLOB_RECURSE INCLUDE_SOURCES
        "${CMAKE_SOURCE_DIR}/include/*.*pp")



add_executable(MindbenderSolver
        ${MINDBENDER_SOLVER_SOURCES}
        ${INCLUDE_SOURCES}
        # main.cpp
        MindbenderSolver/programs/testDFS.cpp
        # MindbenderSolver/programs/testPerms.cpp
        # MindbenderSolver/code/perms_gen.hpp
        # MindbenderSolver/programs/ensurePuzzlesValid.cpp
        # MindbenderSolver/code/B1B2Only.hpp
        # MindbenderSolver/docs/expand.hpp
        # MindbenderSolver/programs/TestKnownSolutions.cpp
        # MindbenderSolver/programs/testJVec.cpp
        # MindbenderSolver/programs/testHeuristics.cpp
        # MindbenderSolver/programs/testSorting.cpp
        # MindbenderSolver/programs/UnitTesting.cpp
)



# check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512_1)
# check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512_2)
# check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512_3)
# check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512_4)
# if(COMPILER_SUPPORTS_AVX512_1 AND COMPILER_SUPPORTS_AVX512_2
#         AND COMPILER_SUPPORTS_AVX512_3 AND COMPILER_SUPPORTS_AVX512_4)
target_compile_options(MindbenderSolver PRIVATE -mbmi2 -msse4.1)
# else()
#     message(WARNING "AVX-512 not supported by the compiler.")
# endif()
