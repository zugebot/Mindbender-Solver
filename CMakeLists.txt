cmake_minimum_required(VERSION 3.24)


set(CMAKE_CXX_STANDARD 23)
option(USE_CUDA "Enable CUDA support" ON) # "ON" or "OFF"


# project decl
if (USE_CUDA)
    project(MindbenderSolver LANGUAGES C CXX CUDA)
    add_compile_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
else()
    project(MindbenderSolver LANGUAGES C CXX)
    message(STATUS "CUDA support disabled")
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
set(MAIN_FILE "${CMAKE_SOURCE_DIR}/MindbenderSolver/programs/testDFS.cpp")
set_source_files_properties(${MAIN_FILE} PROPERTIES LANGUAGE CUDA)
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
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -flto -funroll-loops -fstrict-aliasing -fomit-frame-pointer>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Xcompiler=-march=native,-Xcompiler=-flto,-Xcompiler=-funroll-loops,-Xcompiler=-fstrict-aliasing,-Xcompiler=-fomit-frame-pointer>
    )
    target_link_options(MindbenderSolver PRIVATE -flto)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(MindbenderSolver PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
        )
    else()
        # For Debug builds, add only necessary flags without conflicting optimizations
        target_compile_options(MindbenderSolver PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Zi>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
        )
    endif()
endif()


# CUDA being dumb
if (USE_CUDA)
    set_target_properties(MindbenderSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${MINDBENDER_SOLVER_SOURCES} PROPERTIES LANGUAGE CUDA)
    message(STATUS "CUDA Toolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()
