# -----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------
#
# pto_aiv_treduce — AIV-side trigger for the PTO gated-reduce CKE (Step 3).
#
# One output library:
#   - libpto_aiv_treduce.so   (vec arch; contains BOTH the AIV kernel
#                              `pto_aiv_treduce_kernel` and the host-side
#                              `extern "C" pto_aiv_treduce_launch` launcher
#                              that dispatches via `KernelName<<<...>>>(args)`.)
#
# This is the same one-translation-unit pattern as kernels/manual/a5/gemm_ar
# (`comm_kernel.cpp` holds both kernel + `launchGemmCommAll`). On this CANN /
# bisheng combo, the `<<<...>>>` triple-bracket dispatch can ONLY be emitted
# inside a bisheng-compiled CCE TU, so we cannot keep the launcher in a plain
# C++ TU compiled by g++. Earlier attempt to do so (separate launcher.cpp +
# `pto_aiv_treduce_kernel_thunk`) failed at link time.

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
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
)

set(CCE_OPTS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    # ── HYPOTHESIS H5 (DebugTunnel printf deadlock) — 2026-04-29 ────────────
    # `--cce-enable-print` makes bisheng emit the launch thunk with an extra
    # `cce::internal::DebugTunnelData*` parameter, and the device-side printf
    # implementation expects host runtime to manage a print payload buffer
    # via `cce::internal::PrintPayload::OnHostInitialize/OnHostFinish`. Our
    # hand-rolled `extern "C" pto_aiv_treduce_launch` does NOT do that init,
    # so the device kernel may park on the first `AscendC::printf` waiting
    # for a payload buffer that never materializes — symptom matches Step 3
    # observation: host sees `[PTO_GATE] AIV released` (launch returned 0),
    # device-side `[AIV/treduce] kernel ran:` never fires, CCU stream hangs
    # at WaitCKE, rc=124 timeout. To validate, comment this out AND comment
    # out the two `AscendC::printf` calls in kernel.cpp, rebuild, rerun
    # run_gate_tests_rs.sh. If D2_AIV_SWEEP gets any rc=0 row, H5 is
    # confirmed and we drop print permanently (or wire DebugTunnel init).
    # --cce-enable-print  # disabled — see hypothesis block above
)

# ============================================================================
# AIV Kernel + Host Launcher Library — single TU, vec arch
# ============================================================================
# Both `__global__ __aicore__ pto_aiv_treduce_kernel` and the host-exported
# `extern "C" pto_aiv_treduce_launch` live in kernel.cpp. Bisheng splits them
# automatically based on the `__global__ __aicore__` annotation; the host
# launcher emits the bisheng-internal launch thunk, which is what the example
# resolves through dlopen + dlsym.
add_library(pto_aiv_treduce SHARED kernel.cpp)
target_compile_options(pto_aiv_treduce PRIVATE
    ${CCE_OPTS}
    --cce-aicore-arch=dav-c310-vec
    -DMEMORY_BASE -std=c++17
)
target_include_directories(pto_aiv_treduce PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(pto_aiv_treduce PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(pto_aiv_treduce PRIVATE runtime)
target_link_options(pto_aiv_treduce PRIVATE --cce-fatobj-link)

# ----------------------------------------------------------------------------
# Helper: copy the .so to a known dist/ dir so users can LD_LIBRARY_PATH it
# (or set PTO_AIV_TREDUCE_LIB to the absolute path) without chasing CMake's
# binary tree.
# ----------------------------------------------------------------------------
set(TREDUCE_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
add_custom_command(TARGET pto_aiv_treduce POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TREDUCE_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pto_aiv_treduce>
        ${TREDUCE_DIST_DIR}/
)

message(STATUS "pto_aiv_treduce: built libpto_aiv_treduce.so (kernel + launcher) → ${TREDUCE_DIST_DIR}")
