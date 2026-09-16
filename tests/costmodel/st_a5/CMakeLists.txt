# st_a5 — A5 host mock single-instruction tests (__NPU_ARCH__=3101).
# Verify supported A5 TileOp formula predictions without device intrinsics.
#
# Standalone CMake project. Build and run with:
#   python3 include/pto/costmodel/a5/formula_costmodel/gen_formula_params_header.py
#   cmake -S tests/costmodel/st_a5 -B tests/costmodel/st_a5/build -DCMAKE_BUILD_TYPE=Debug
#   cmake --build tests/costmodel/st_a5/build -j 4
#   ctest --test-dir tests/costmodel/st_a5/build --output-on-failure
cmake_minimum_required(VERSION 3.18)

project(pto_test_costmodel_a5 CXX)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_definitions(__COSTMODEL __NPU_ARCH__=3101 PTO_COMM_NOT_SUPPORTED)

# Match the A2/A3 host-build environment for PTO device headers.
add_compile_options(
    -Wno-macro-redefined -Wno-ignored-attributes
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_subdirectory(testcase)
