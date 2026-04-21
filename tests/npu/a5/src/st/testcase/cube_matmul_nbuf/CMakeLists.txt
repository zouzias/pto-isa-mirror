# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
# --------------------------------------------------------------------------------

# Build kernel library (all 5 configurations in one translation unit)
add_library(cube_matmul_nbuf_kernel SHARED cube_matmul_nbuf_kernel.cpp)
target_compile_options(cube_matmul_nbuf_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-cube
    -DREGISTER_BASE
    -std=c++17
)
target_include_directories(cube_matmul_nbuf_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(cube_matmul_nbuf_kernel PRIVATE --cce-fatobj-link)

# Build test executable
add_executable(cube_matmul_nbuf main_nbuf.cpp)
target_compile_options(cube_matmul_nbuf PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_compile_definitions(cube_matmul_nbuf PRIVATE TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(cube_matmul_nbuf PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../common
)

target_link_directories(cube_matmul_nbuf PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(cube_matmul_nbuf PRIVATE
    cube_matmul_nbuf_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
)
