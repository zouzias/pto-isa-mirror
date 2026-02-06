# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(pto_example)

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

if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please set SHMEM_HOME_PATH environment variable.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

# SHMEM version selection: ASCEND_SHMEM = legacy (shmem_api.h), CANN_SHMEM = new (shmem.h)
if(DEFINED ENV{SHMEM_BACKEND})
    set(SHMEM_BACKEND $ENV{SHMEM_BACKEND})
endif()
if(NOT DEFINED SHMEM_BACKEND)
    # Auto-detect: prefer new shmem.h
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

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

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
    set(CCE_PRINT_FLAG "--cce-enable-print")
else()
    set(CCE_PRINT_FLAG "")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# Set shmem-specific includes based on backend
if(SHMEM_BACKEND STREQUAL "ASCEND_SHMEM")
    add_compile_definitions(ASCEND_SHMEM)
    include_directories(
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    )
else()  # CANN_SHMEM
    add_compile_definitions(CANN_SHMEM)
    include_directories(
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/shmem/include/device
        ${SHMEM_HOME_PATH}/../src/device
    )
endif()

set(NAME gemm_ar)

# ============================================================================
# Kernel library (runs on AI cores with Cube architecture)
# ============================================================================
add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
target_compile_options(${NAME}_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    ${CCE_PRINT_FLAG} 
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE 
    -std=c++17
)
target_include_directories(${NAME}_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    $ENV{ASCEND_HOME_PATH}/pkg_inc/
    $ENV{ASCEND_HOME_PATH}/pkg_inc/profiling/
    $ENV{ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_directories(${NAME}_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/lib
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${NAME}_kernel PRIVATE
    shmem
    runtime
)
target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Common include paths
# ============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PROJECT_SOURCE_DIR}/../../..
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# ============================================================================
# Host executable
# ============================================================================
add_executable(${NAME} main.cpp)
target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(${NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${PROJECT_SOURCE_DIR}
)

target_link_directories(${NAME} PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    ${SHMEM_HOME_PATH}/shmem/lib
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
)

target_link_libraries(${NAME} PRIVATE
    ${NAME}_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
    shmem
)
