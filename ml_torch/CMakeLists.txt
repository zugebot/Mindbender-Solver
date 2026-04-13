cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_PREFIX_PATH "C:/dev/libtorch-win-shared-with-deps-debug-2.11.0+cu130/libtorch")
else()
    set(CMAKE_PREFIX_PATH "C:/dev/libtorch-win-shared-with-deps-2.11.0+cu130/libtorch")
endif()

project(mindbender_ml_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{LIBTORCH_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
endif()

find_package(Torch REQUIRED)

set(ML_TORCH_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/train/train_deep_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/train/infer_deep_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/model/distance_net.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../utils/jvec.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../code/board.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../code/perms.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../code/rotations.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../code/rotations_row.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../code/rotations_col.cpp"
)

set(ML_TORCH_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/train"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(ml_torch
        ${ML_TORCH_SOURCES}
)

set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(ROOT_BUILD_DIR "${PROJECT_ROOT_DIR}/build")

set_target_properties(ml_torch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${ROOT_BUILD_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${ROOT_BUILD_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${ROOT_BUILD_DIR}"
)

target_include_directories(ml_torch PRIVATE ${ML_TORCH_INCLUDE_DIRS})

target_link_libraries(ml_torch PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(ml_torch PRIVATE cxx_std_20)
target_compile_definitions(ml_torch PRIVATE NOMINMAX)

if(MSVC)
    target_compile_options(ml_torch PRIVATE /wd4267)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ml_torch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ml_torch>
    )
endif()

message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
message(STATUS "ML_TORCH_INCLUDE_DIRS = ${ML_TORCH_INCLUDE_DIRS}")
message(STATUS "ML_TORCH_SOURCES = ${ML_TORCH_SOURCES}")
message(STATUS "ML_TORCH built successfully - exe output: ${ROOT_BUILD_DIR}/ml_torch(.exe)")
