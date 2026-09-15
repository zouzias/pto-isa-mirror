# st_a5 — A5 host mock single-instruction tests (__NPU_ARCH__=3101).
# Verify supported A5 TileOp formula predictions without device intrinsics.
#
# Standalone CMake project. Build and run with:
#   cmake -S tests/costmodel/st_a5 -B tests/costmodel/st_a5/build -DCMAKE_BUILD_TYPE=Debug
#   cmake --build tests/costmodel/st_a5/build -j 4
#   ctest --test-dir tests/costmodel/st_a5/build --output-on-failure
cmake_minimum_required(VERSION 3.18)

project(pto_test_costmodel_a5 CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../include/pto/costmodel/cmake/pto_costmodel.cmake)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Wno-macro-redefined -Wno-ignored-attributes)

add_subdirectory(testcase)
