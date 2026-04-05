cmake_minimum_required(VERSION 3.24)

# ------------------------------------------------------------
# Project options
# ------------------------------------------------------------
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_SIMT_ROTATIONS "Use a SIMT-friendly rotation implementation with smaller GPU code size" ON)
option(USE_BOARD_HASH "Enable board hashing" OFF)

# ------------------------------------------------------------
# Project declaration
# ------------------------------------------------------------
if (USE_CUDA)
    project(MindbenderSolver LANGUAGES C CXX CUDA)
    enable_language(CUDA)

    message(STATUS "CUDA support enabled")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_compile_definitions(USE_CUDA)

    find_package(CUDAToolkit REQUIRED)

    set(CUDA_SOURCE_FILES
            "${CMAKE_SOURCE_DIR}/code/rotations.cpp"
            "${CMAKE_SOURCE_DIR}/code/board.cpp"
            "${CMAKE_SOURCE_DIR}/code/memory.cpp"
    )

    set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

    string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
else()
    project(MindbenderSolver LANGUAGES C CXX)
    message(STATUS "CUDA support disabled")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Subprojects
# ------------------------------------------------------------
add_subdirectory(mouse)

# ------------------------------------------------------------
# Compile definitions
# ------------------------------------------------------------
if (USE_SIMT_ROTATIONS)
    add_compile_definitions(USE_SIMT_ROTATIONS)
endif()

if (USE_BOARD_HASH)
    add_compile_definitions(USE_BOARD_HASH)
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Skipping TBB lookup for MSVC")
else()
    find_package(TBB QUIET)
    if (TBB_FOUND)
        message(STATUS "TBB found")
        add_compile_definitions(TBB_FOUND)
    else()
        message(STATUS "TBB not found")
    endif()
endif()

find_library(BOOST_LIBRARY NAMES Boost boost)
if (BOOST_LIBRARY)
    message(STATUS "Boost found")
    add_compile_definitions(BOOST_FOUND)
else()
    message(STATUS "Boost not found")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CHUZZLE_VIEWER_DIR "${CMAKE_SOURCE_DIR}/pyth/viewer-dp")
set(CHUZZLE_STUDIO_FILE "${CHUZZLE_VIEWER_DIR}/chuzzle_mouse_dp_studio.py")
set(CHUZZLE_VIEWER_PYTHON "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------
include_directories("${CMAKE_SOURCE_DIR}")

if (USE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDA Toolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# ------------------------------------------------------------
# Source discovery
# ------------------------------------------------------------
set(COMMON_SOURCES
        "${CMAKE_SOURCE_DIR}/code/board.cpp"
        "${CMAKE_SOURCE_DIR}/code/memory.cpp"
        "${CMAKE_SOURCE_DIR}/code/perms.cpp"
        "${CMAKE_SOURCE_DIR}/code/rotations.cpp"
        "${CMAKE_SOURCE_DIR}/code/rotations_col.cpp"
        "${CMAKE_SOURCE_DIR}/code/rotations_row.cpp"

        "${CMAKE_SOURCE_DIR}/utils/jvec.cpp"
)

if (USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
            "${CMAKE_SOURCE_DIR}/code/*.cu"
    )
    list(APPEND COMMON_SOURCES ${CUDA_SOURCES})
endif()

set(SHARED_SOURCES
        ${COMMON_SOURCES}
)

# ------------------------------------------------------------
# Helper to create one executable per program
# ------------------------------------------------------------
function(add_solver_program target_name main_file)
    add_executable(${target_name} EXCLUDE_FROM_ALL
            "${main_file}"
            ${SHARED_SOURCES}
    )

    if (TBB_FOUND)
        target_link_libraries(${target_name} PRIVATE TBB::tbb)
    endif()

    if (MSVC)
        if (USE_CUDA)
            if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                target_compile_options(${target_name} PRIVATE
                        $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL /FA>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2,-Xcompiler=/GL>
                )
            else()
                target_compile_options(${target_name} PRIVATE
                        $<$<COMPILE_LANGUAGE:CXX>:/Zi /FA>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
                )
            endif()
        else()
            if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                target_compile_options(${target_name} PRIVATE /O2 /GL /FA)
            else()
                target_compile_options(${target_name} PRIVATE /Zi /FA)
            endif()
        endif()

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if (USE_CUDA)
            if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                target_compile_options(${target_name} PRIVATE
                        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mbmi2 -funroll-loops -fstrict-aliasing -fomit-frame-pointer>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Xcompiler=-march=native,-Xcompiler=-mbmi2,-Xcompiler=-funroll-loops,-Xcompiler=-fstrict-aliasing,-Xcompiler=-fomit-frame-pointer>
                )
            else()
                target_compile_options(${target_name} PRIVATE
                        $<$<COMPILE_LANGUAGE:CXX>:-O0 -g -mbmi2>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O0,-Xcompiler=-g,-Xcompiler=-mbmi2>
                )
            endif()
        else()
            if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                target_compile_options(${target_name} PRIVATE
                        -O3 -march=native -mbmi2 -funroll-loops -fstrict-aliasing -fomit-frame-pointer
                )
            else()
                target_compile_options(${target_name} PRIVATE
                        -O0 -g -mbmi2
                )
            endif()
        endif()
    endif()

    if (USE_CUDA)
        set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
endfunction()

# ------------------------------------------------------------
# Programs
# ------------------------------------------------------------
add_solver_program(testLayeredDFS         "${CMAKE_SOURCE_DIR}/programs/testLayeredDFS.cpp")
add_solver_program(testKnownSolutions     "${CMAKE_SOURCE_DIR}/programs/testKnownSolutions.cpp")
add_solver_program(testPuzzlesValid       "${CMAKE_SOURCE_DIR}/programs/testPuzzlesValid.cpp")
add_solver_program(testLeftGenRightCache  "${CMAKE_SOURCE_DIR}/programs/testLeftGenRightCache.cpp")
add_solver_program(testRotations          "${CMAKE_SOURCE_DIR}/programs/testRotations.cpp")
add_solver_program(testDepth4Direct       "${CMAKE_SOURCE_DIR}/programs/testDepth4Direct.cpp")
add_solver_program(testDepth4DirectFat    "${CMAKE_SOURCE_DIR}/programs/testDepth4DirectFat.cpp")
add_solver_program(findSolutions           "${CMAKE_SOURCE_DIR}/programs/findSolutions.cpp")
add_solver_program(CFCC                   "${CMAKE_SOURCE_DIR}/programs/CFCC.cpp")
add_solver_program(expandAllLevels        "${CMAKE_SOURCE_DIR}/programs/expandAllLevels.cpp")