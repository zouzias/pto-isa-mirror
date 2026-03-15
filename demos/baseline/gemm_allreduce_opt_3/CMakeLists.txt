# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# ============================================================================
# GEMM AllReduce Dual Stream Demo - CMakeLists.txt
#
# This CMake file builds:
# - gemm_compute_kernel: GEMM computation kernel (Cube-only: dav-c220-cube)
#   Computes tiles and signals comm kernel via queue mechanism
# - comm_kernel: Communication kernel (Vector-only: dav-c220-vec)
#   Polls compute queues, sends data to remote, receives and reduces
# - gemm_allreduce: Host executable that launches both kernels on dual streams
#
# Architecture:
#   Compute Stream (Cube)           Communication Stream (Vec)
#   ┌─────────────────┐             ┌─────────────────────────────────────┐
#   │ Compute Tile 0  │──Queue──────│ Poll Queue → TPUT to remote         │
#   │ Compute Tile 1  │             │ → Write signal → Check recv_count   │
#   │ Compute Tile 2  │             │ → TREDUCE when all data received    │
#   │      ...        │             └─────────────────────────────────────┘
#   └─────────────────┘
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Use bisheng compiler for both host and device code
# Must be set BEFORE project() for CMake to detect them correctly
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_gemm_allreduce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# shmem 安装根目录（需在环境中提供）
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please export SHMEM_HOME_PATH to shmem install root.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
    message(STATUS "SHMEM_HOME_PATH: ${SHMEM_HOME_PATH}")
endif()

# SHMEM 版本选择: ASCEND_SHMEM = 老版本(shmem_api.h), CANN_SHMEM = 新版本(shmem.h)
if(DEFINED ENV{SHMEM_BACKEND})
    set(SHMEM_BACKEND $ENV{SHMEM_BACKEND})
endif()
if(NOT DEFINED SHMEM_BACKEND)
    if(EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem.h")
        set(SHMEM_BACKEND "CANN_SHMEM")
    elseif(EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem_api.h")
        set(SHMEM_BACKEND "ASCEND_SHMEM")
    else()
        message(FATAL_ERROR "Cannot find shmem.h or shmem_api.h in ${SHMEM_HOME_PATH}/shmem/include/")
    endif()
endif()
message(STATUS "SHMEM_BACKEND: ${SHMEM_BACKEND}")

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
    )

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

# 公共 include 路径
# NOTE: pto-comm-isa include path MUST come first to override CANN's pto_tile.hpp
# which lacks RawDType and default constructor
include_directories(
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# 根据 SHMEM 版本设置不同的 include 路径和编译宏
if(SHMEM_BACKEND STREQUAL "ASCEND_SHMEM")
    add_compile_definitions(ASCEND_SHMEM)
    set(SHMEM_INCLUDE_DIRS
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    )
    set(SHMEM_LIB_DIRS
        ${SHMEM_HOME_PATH}/shmem/lib
        ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    )
else()
    add_compile_definitions(CANN_SHMEM)
    set(SHMEM_INCLUDE_DIRS
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/shmem/include/device
        ${SHMEM_HOME_PATH}/../src/device
    )
    set(SHMEM_LIB_DIRS
        ${SHMEM_HOME_PATH}/shmem/lib
    )
endif()

# Common kernel include directories
set(KERNEL_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase
    ${PROJECT_SOURCE_DIR}
    ${SHMEM_INCLUDE_DIRS}
    # AscendC include paths for kernel_operator.h
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
)

# ============================================================================
# GEMM Compute Kernel (Cube-only mode: dav-c220-cube)
# This kernel computes GEMM tiles and signals comm kernel via queue mechanism
# ============================================================================
add_library(gemm_compute_kernel SHARED gemm_compute_kernel.cpp)
target_compile_options(gemm_compute_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    --cce-aicore-arch=dav-c220-cube
    -DMEMORY_BASE 
    -std=c++17
)
target_include_directories(gemm_compute_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(gemm_compute_kernel PRIVATE
    ${SHMEM_LIB_DIRS}
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(gemm_compute_kernel PRIVATE shmem runtime)
target_link_options(gemm_compute_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Communication Kernel (Vector-only mode: dav-c220-vec)
# This kernel polls compute queues, sends data to remote ranks, and performs
# reduction when all data is received
# ============================================================================
add_library(comm_kernel SHARED comm_kernel.cpp)
target_compile_options(comm_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE 
    -std=c++17
)
target_include_directories(comm_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(comm_kernel PRIVATE
    ${SHMEM_LIB_DIRS}
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(comm_kernel PRIVATE gemm_compute_kernel shmem runtime)
target_link_options(comm_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Host Executable
# Launches both kernels on dual streams for parallel execution
# ============================================================================
add_executable(gemm_allreduce main.cpp)
target_compile_options(gemm_allreduce PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(gemm_allreduce PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase
    ${PROJECT_SOURCE_DIR}
    ${SHMEM_INCLUDE_DIRS}
)

target_link_directories(gemm_allreduce PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    ${SHMEM_LIB_DIRS}
)

target_link_libraries(gemm_allreduce PRIVATE
    gemm_compute_kernel
    comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread shmem
)

# ============================================================================
# Benchmark Comm Kernel (Vector-only mode: dav-c220-vec)
# Device-side kernel for benchmarking single comm core
# ============================================================================
add_library(benchmark_comm_kernel SHARED benchmark_comm_kernel.cpp)
target_compile_options(benchmark_comm_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE 
    -std=c++17
)
target_include_directories(benchmark_comm_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(benchmark_comm_kernel PRIVATE
    ${SHMEM_LIB_DIRS}
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(benchmark_comm_kernel PRIVATE shmem runtime)
target_link_options(benchmark_comm_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Benchmark: Single Comm Core Performance Test
# ============================================================================
add_executable(benchmark_single_comm_core benchmark_single_comm_core.cpp)
target_compile_options(benchmark_single_comm_core PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(benchmark_single_comm_core PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase
    ${PROJECT_SOURCE_DIR}
    ${SHMEM_INCLUDE_DIRS}
)

target_link_directories(benchmark_single_comm_core PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    ${SHMEM_LIB_DIRS}
)

target_link_libraries(benchmark_single_comm_core PRIVATE
    benchmark_comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread shmem
)

# ============================================================================
# Benchmark: Single Compute Core Performance Test
# ============================================================================
add_executable(benchmark_single_compute_core benchmark_single_compute_core.cpp)
target_compile_options(benchmark_single_compute_core PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(benchmark_single_compute_core PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase
    ${PROJECT_SOURCE_DIR}
    ${SHMEM_INCLUDE_DIRS}
)

target_link_directories(benchmark_single_compute_core PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    ${SHMEM_LIB_DIRS}
)

target_link_libraries(benchmark_single_compute_core PRIVATE
    gemm_compute_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)

# ============================================================================
# Benchmark: Optimal Configuration Calculator
# ============================================================================
add_executable(benchmark_optimal_config benchmark_optimal_config.cpp)
target_compile_options(benchmark_optimal_config PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(benchmark_optimal_config PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase
    ${PROJECT_SOURCE_DIR}
    ${SHMEM_INCLUDE_DIRS}
)

target_link_directories(benchmark_optimal_config PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    ${SHMEM_LIB_DIRS}
)

target_link_libraries(benchmark_optimal_config PRIVATE
    gemm_compute_kernel
    comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread shmem
)
