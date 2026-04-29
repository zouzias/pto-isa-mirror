# -----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------
#
# pto_aiv_treduce — AIV-side trigger for the PTO gated-reduce CKE (Step 3).
#
# Two output libraries:
#   - libpto_aiv_treduce_kernel.so   (vec arch; the actual AIV kernel)
#   - libpto_aiv_treduce.so          (host-side launcher exporting C entry point;
#                                     dlopen'd by pto/aiv/detail/treduce_launcher.hpp)
#
# Build expectations: same toolchain conventions as kernels/manual/a5/gemm_ar
# (CCE compiler, ASCEND_HOME_PATH set, dav-c310-vec for vec arch).

cmake_minimum_required(VERSION 3.16)

# CCE compiler (bisheng) MUST be set before project() so CMake doesn't
# probe gcc/clang first. Mirrors kernels/manual/a5/gemm_ar/CMakeLists.txt:22-24.
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_aiv_treduce LANGUAGES CXX)

if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH not set; source CANN set_env.sh first")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Resolve arch-specific kernel platform root (mirror of gemm_ar/CMakeLists.txt).
if(EXISTS ${ASCEND_HOME_PATH}/x86_64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/aarch64-linux)
else()
    message(FATAL_ERROR "Cannot find Ascend kernel include root under ${ASCEND_HOME_PATH}")
endif()

set(KERNEL_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
)

set(CCE_OPTS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

# ============================================================================
# AIV Kernel Library — vec arch
# ============================================================================
add_library(pto_aiv_treduce_kernel SHARED kernel.cpp)
target_compile_options(pto_aiv_treduce_kernel PRIVATE
    ${CCE_OPTS}
    --cce-aicore-arch=dav-c310-vec
    -DMEMORY_BASE -std=c++17
)
target_include_directories(pto_aiv_treduce_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(pto_aiv_treduce_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(pto_aiv_treduce_kernel PRIVATE runtime)
target_link_options(pto_aiv_treduce_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Host Launcher Library — exports pto_aiv_treduce_launch via C ABI
# ============================================================================
add_library(pto_aiv_treduce SHARED launcher.cpp)
target_compile_options(pto_aiv_treduce PRIVATE
    -std=c++17 -fPIC -Wall
)
target_include_directories(pto_aiv_treduce PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/acl
)
target_link_directories(pto_aiv_treduce PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(pto_aiv_treduce PRIVATE
    pto_aiv_treduce_kernel    # for the kernel-launch thunk symbol
    ascendcl runtime
)

# ----------------------------------------------------------------------------
# Helper: copy both .so to a known dist/ dir so users can LD_LIBRARY_PATH it
# without chasing CMake's binary tree.
# ----------------------------------------------------------------------------
set(TREDUCE_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
add_custom_command(TARGET pto_aiv_treduce POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TREDUCE_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pto_aiv_treduce_kernel>
        $<TARGET_FILE:pto_aiv_treduce>
        ${TREDUCE_DIST_DIR}/
)

message(STATUS "pto_aiv_treduce: built libpto_aiv_treduce_kernel.so + libpto_aiv_treduce.so → ${TREDUCE_DIST_DIR}")
