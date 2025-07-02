cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)


unset(USE_CUDA CACHE)
option(USE_CUDA "Enable CUDA support" OFF) # "ON" or "OFF"


unset(USE_TRIMMED_ROTATIONS CACHE)
option(USE_TRIMMED_ROTATIONS "Smaller memory for rotation functions" ON)
if (USE_TRIMMED_ROTATIONS)
    add_compile_definitions(-DUSE_TRIMMED_ROTATIONS)
endif()



# project decl
if (USE_CUDA)
    project(MindbenderSolver LANGUAGES C CXX CUDA)
    add_compile_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")

    # set files to be interpreted as .cu
    set_source_files_properties(
            "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/rotations.cpp"
            PROPERTIES LANGUAGE CUDA
    )
    set_source_files_properties(
            "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/board.cpp"
            PROPERTIES LANGUAGE CUDA
    )
    set_source_files_properties(
            "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/memory.cpp"
            PROPERTIES LANGUAGE CUDA
    )

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
else()
    project(MindbenderSolver LANGUAGES C CXX)
    message(STATUS "CUDA support disabled")
endif()


if (USE_BOARD_HASH)
    add_compile_definitions(-DUSE_BOARD_HASH)
endif()


# find Boost
find_library(BOOST_FOUND Boost)
if (BOOST_FOUND)
    message(STATUS "Boost found")
    add_compile_definitions(BOOST_FOUND)
else()
    message(STATUS "Boost not found")
endif()


# find TBB
find_library(TBB_FOUND TBB)
if (TBB_FOUND)
    message(STATUS "TBB found")
    add_compile_definitions(TBB_FOUND)
    find_package(TBB REQUIRED)
else()
    message(STATUS "TBB not found")
endif()


# include dirs
include_directories(${CMAKE_SOURCE_DIR})
if (USE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()


# file structure of project
file(GLOB_RECURSE MINDBENDER_SOLVER_SOURCES
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.*pp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/code/*.*cu"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/utils/*.*pp"
        "${CMAKE_SOURCE_DIR}/MindbenderSolver/include.hpp"
)
file(GLOB_RECURSE INCLUDE_SOURCES
        "${CMAKE_SOURCE_DIR}/include/*.*pp"
)
set(PROGRAM_FOLDER "${CMAKE_SOURCE_DIR}/MindbenderSolver/programs")

# set(MAIN_FILE "${PROGRAM_FOLDER}/testLayeredDFS.cpp")
set(MAIN_FILE "${PROGRAM_FOLDER}/testKnownSolutions.cpp")
# set(MAIN_FILE "${PROGRAM_FOLDER}/testPuzzlesValid.cpp")


if (USE_CUDA)
    set_source_files_properties(${MAIN_FILE} PROPERTIES LANGUAGE CUDA)
endif()


add_executable(MindbenderSolver
        ${MAIN_FILE}
        ${MINDBENDER_SOLVER_SOURCES}
        ${INCLUDE_SOURCES}
)


# link TBB if found
if (TBB_FOUND)
    target_link_libraries(MindbenderSolver PRIVATE TBB::tbb)
endif()


# compiler-specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MindbenderSolver PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -funroll-loops -fstrict-aliasing -fomit-frame-pointer>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Xcompiler=-march=native,-Xcompiler=-flto,-Xcompiler=-funroll-loops,-Xcompiler=-fstrict-aliasing,-Xcompiler=-fomit-frame-pointer>
    )

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(MindbenderSolver PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL /FA>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/O2 /GL">
        )
    else()
        # Debug mode
        target_compile_options(MindbenderSolver PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Zi /FA>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/Zi">
        )
    endif()

endif()


# CUDA being dumb
if (USE_CUDA)
    set_target_properties(MindbenderSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${MINDBENDER_SOLVER_SOURCES} PROPERTIES LANGUAGE CUDA)
    message(STATUS "CUDA Toolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()
