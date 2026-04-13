cmake_minimum_required(VERSION 3.24)

add_library(ChuzzleMouse SHARED
        src/chuzzle_mouse.c
)

target_include_directories(ChuzzleMouse
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    # target_compile_options(ChuzzleMouse PRIVATE /O2)
else()
    target_compile_options(ChuzzleMouse PRIVATE -O3)
    target_link_libraries(ChuzzleMouse PRIVATE m)
endif()

set_target_properties(ChuzzleMouse PROPERTIES
        OUTPUT_NAME "chuzzle_mouse"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pyth/viewer"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pyth/viewer"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mouse"
)