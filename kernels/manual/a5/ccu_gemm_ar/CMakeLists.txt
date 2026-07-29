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
# CCU GEMM AllReduce — Pull Reduce + Push Broadcast (Persistent CCU)
#
# Targets:
#   ccu_gemm_ar_compute_kernel  — Cube-only kernel (dav-c310-cube)
#   ccu_gemm_ar_progress_kernel — Vector-only AIV progress kernel (dav-c310-vec)
#   ccu_gemm_allreduce          — Host executable (CCU launcher)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_ccu_gemm_allreduce)

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

if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A5)

# Matrix dimensions (configurable via -DCONFIG_G_M=... -DCONFIG_G_K=... -DCONFIG_G_N=...)
if(DEFINED CONFIG_G_M)
    add_compile_definitions(CONFIG_G_M=${CONFIG_G_M})
endif()
if(DEFINED CONFIG_G_K)
    add_compile_definitions(CONFIG_G_K=${CONFIG_G_K})
endif()
if(DEFINED CONFIG_G_N)
    add_compile_definitions(CONFIG_G_N=${CONFIG_G_N})
endif()

# Tile sizes (configurable via -DCONFIG_G_BASE_M=... -DCONFIG_G_BASE_K=... -DCONFIG_G_BASE_N=...)
if(DEFINED CONFIG_G_BASE_M)
    add_compile_definitions(CONFIG_G_BASE_M=${CONFIG_G_BASE_M})
endif()
if(DEFINED CONFIG_G_BASE_K)
    add_compile_definitions(CONFIG_G_BASE_K=${CONFIG_G_BASE_K})
endif()
if(DEFINED CONFIG_G_BASE_N)
    add_compile_definitions(CONFIG_G_BASE_N=${CONFIG_G_BASE_N})
endif()

# CCU mission parallelism K (configurable via -DCONFIG_CCU_MISSION_PARALLEL=...)
if(DEFINED CONFIG_CCU_MISSION_PARALLEL)
    add_compile_definitions(CONFIG_CCU_MISSION_PARALLEL=${CONFIG_CCU_MISSION_PARALLEL})
endif()

# AIV->CCU progress handshake depth (1 = stable, 2 = cross-group pipelining)
if(DEFINED CONFIG_CCU_PIPE_DEPTH)
    add_compile_definitions(CONFIG_CCU_PIPE_DEPTH=${CONFIG_CCU_PIPE_DEPTH})
endif()

add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

if(EXISTS ${ASCEND_HOME_PATH}/x86_64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/aarch64-linux)
else()
    message(FATAL_ERROR "Cannot find Ascend kernel include root under ${ASCEND_HOME_PATH}/x86_64-linux or aarch64-linux")
endif()

# CANN 9 layout keeps pkg_inc/lib64 under ${ASCEND_HOME_PATH}/<arch>-linux/, while
# older installs expose them directly under ${ASCEND_HOME_PATH}/. Match ST's
# PTO_TOOLKIT_DIR fallback so ccu_gemm_ar and treduce_ccu see the same roots.
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/lib64)
    set(ASCEND_LIB_ROOT ${ASCEND_KERNEL_PLATFORM_ROOT}/lib64)
elseif(EXISTS ${ASCEND_HOME_PATH}/lib64)
    set(ASCEND_LIB_ROOT ${ASCEND_HOME_PATH}/lib64)
else()
    message(FATAL_ERROR "Cannot find lib64 under ${ASCEND_KERNEL_PLATFORM_ROOT} or ${ASCEND_HOME_PATH}")
endif()

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc)
    set(ASCEND_PKG_INC_ROOT ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc)
elseif(EXISTS ${ASCEND_HOME_PATH}/pkg_inc)
    set(ASCEND_PKG_INC_ROOT ${ASCEND_HOME_PATH}/pkg_inc)
else()
    set(ASCEND_PKG_INC_ROOT ${ASCEND_KERNEL_PLATFORM_ROOT}/pkg_inc)
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
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
)

# hcomm CCU headers:
# - cann-9.2 public: hcomm/ccu/ccu_launch.h under include/
# - cann-9.2 internal v1 microcode (ccu_kernel.h etc.): under
#   <arch>/asc/include/adv_api/hccl/internal/hcomm/pkg_inc
# Override with -DHCOMM_PKG_INC=... or export HCOMM_PKG_INC=...
set(_HCOMM_CCU_PROBE "")
set(_HCOMM_CCU_INTERNAL "")
if(DEFINED HCOMM_PKG_INC)
    set(_HCOMM_CCU_PROBE ${HCOMM_PKG_INC})
elseif(DEFINED ENV{HCOMM_PKG_INC})
    set(_HCOMM_CCU_PROBE $ENV{HCOMM_PKG_INC})
elseif(EXISTS "${ASCEND_PKG_INC_ROOT}/hcomm/ccu/ccu_kernel.h")
    set(_HCOMM_CCU_PROBE ${ASCEND_PKG_INC_ROOT})
elseif(EXISTS "${ASCEND_HOME_PATH}/include/hcomm/ccu/ccu_kernel.h")
    set(_HCOMM_CCU_PROBE ${ASCEND_HOME_PATH}/include)
endif()

# cann-9.2: ccu_kernel.h lives only under the internal tree.
set(_HCOMM_INTERNAL_CANDIDATES
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/internal/hcomm/pkg_inc
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc
    ${ASCEND_HOME_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc
)
foreach(_cand IN LISTS _HCOMM_INTERNAL_CANDIDATES)
    if(EXISTS "${_cand}/hcomm/ccu/ccu_kernel.h")
        set(_HCOMM_CCU_INTERNAL ${_cand})
        break()
    endif()
endforeach()

if(NOT _HCOMM_CCU_PROBE AND NOT _HCOMM_CCU_INTERNAL)
    message(FATAL_ERROR
        "Cannot find hcomm/ccu/ccu_kernel.h.\n"
        "  source set_env.sh so ASCEND_HOME_PATH points at cann-9.2, or set:\n"
        "    cmake ... -DHCOMM_PKG_INC=<path/to/pkg_inc>\n"
        "  Checked: ${ASCEND_PKG_INC_ROOT}, ${ASCEND_HOME_PATH}/include, internal pkg_inc")
endif()
if(_HCOMM_CCU_PROBE)
    message(STATUS "ccu_gemm_ar: hcomm CCU headers from ${_HCOMM_CCU_PROBE}")
endif()
if(_HCOMM_CCU_INTERNAL)
    message(STATUS "ccu_gemm_ar: hcomm CCU internal headers from ${_HCOMM_CCU_INTERNAL}")
endif()
message(STATUS "ccu_gemm_ar: ASCEND_KERNEL_PLATFORM_ROOT=${ASCEND_KERNEL_PLATFORM_ROOT}")
message(STATUS "ccu_gemm_ar: ASCEND_LIB_ROOT=${ASCEND_LIB_ROOT}")

set(HOST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_HOME_PATH}/include/hcomm
    ${ASCEND_HOME_PATH}/include/hcomm/ccu
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/hcomm
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/hcomm/ccu
)
if(_HCOMM_CCU_INTERNAL)
    list(APPEND HOST_INCLUDE_DIRS ${_HCOMM_CCU_INTERNAL})
endif()
if(_HCOMM_CCU_PROBE)
    list(APPEND HOST_INCLUDE_DIRS ${_HCOMM_CCU_PROBE})
endif()
list(APPEND HOST_INCLUDE_DIRS
    ${ASCEND_PKG_INC_ROOT}
    ${ASCEND_PKG_INC_ROOT}/hcomm/ccu
    ${ASCEND_PKG_INC_ROOT}/profiling
    ${ASCEND_PKG_INC_ROOT}/runtime/runtime
)

if(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api)
endif()
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
endif()
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api)
endif()

set(HOST_LINK_DIRS
    ${ASCEND_LIB_ROOT}
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

# Block allocation (configurable via -DCOMPUTE_BLOCKS=N -DCOMM_BLOCKS=M)
if(DEFINED COMPUTE_BLOCKS)
    set(BLOCK_DEFS_COMPUTE -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCKS})
    set(BLOCK_DEFS_ALL -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCKS})
else()
    set(BLOCK_DEFS_COMPUTE "")
    set(BLOCK_DEFS_ALL "")
endif()
if(DEFINED COMM_BLOCKS)
    list(APPEND BLOCK_DEFS_ALL -DCONFIG_COMM_BLOCK_NUM=${COMM_BLOCKS})
endif()
if(DEFINED COMM_GROUP_TILES)
    list(APPEND BLOCK_DEFS_COMPUTE -DCONFIG_COMM_GROUP_TILES=${COMM_GROUP_TILES})
    list(APPEND BLOCK_DEFS_ALL -DCONFIG_COMM_GROUP_TILES=${COMM_GROUP_TILES})
endif()

set(HOST_COMPILE_DEFS ${BLOCK_DEFS_ALL})

# ============================================================================
# CCU GEMM Compute Kernel (Cube-only: dav-c310-cube)
# ============================================================================
add_library(ccu_gemm_ar_compute_kernel SHARED compute_kernel.cpp)
target_compile_options(ccu_gemm_ar_compute_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-cube
    -DMEMORY_BASE -std=c++17
    ${BLOCK_DEFS_COMPUTE}
)
target_include_directories(ccu_gemm_ar_compute_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(ccu_gemm_ar_compute_kernel PRIVATE ${ASCEND_LIB_ROOT})
target_link_libraries(ccu_gemm_ar_compute_kernel PRIVATE runtime)
target_link_options(ccu_gemm_ar_compute_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# CCU GEMM AIV progress kernel (Vector-only: dav-c310-vec)
#   groupDone + groupReady TNOTIFY + fused progress CKE (+ unpack / gate).
# ============================================================================
add_library(ccu_gemm_ar_progress_kernel SHARED progress_kernel.cpp)
target_compile_options(ccu_gemm_ar_progress_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-vec
    -DMEMORY_BASE -std=c++17
    ${BLOCK_DEFS_ALL}
)
target_include_directories(ccu_gemm_ar_progress_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(ccu_gemm_ar_progress_kernel PRIVATE ${ASCEND_LIB_ROOT})
target_link_libraries(ccu_gemm_ar_progress_kernel PRIVATE runtime)
target_link_options(ccu_gemm_ar_progress_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Host Executable
# ============================================================================
add_executable(ccu_gemm_allreduce main.cpp)
target_compile_options(ccu_gemm_allreduce PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} ${HOST_COMPILE_DEFS})
target_include_directories(ccu_gemm_allreduce PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(ccu_gemm_allreduce PUBLIC ${HOST_LINK_DIRS})
target_link_libraries(ccu_gemm_allreduce PRIVATE
    ccu_gemm_ar_compute_kernel
    ccu_gemm_ar_progress_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hccl hcomm m tiling_api register platform c_sec dl nnopbase pthread
)
