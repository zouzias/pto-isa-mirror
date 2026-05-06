# -----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------
#
# pto_gated_reduce_scatter — pilot pto-native gated CCU kernel (phase 2).
#
# Output:
#   - libpto_gated_reduce_scatter.so : host-side CCU kernel implementation
#                                      (hcomm::CcuKernel subclass) + factory.
#
# Architectural rationale and pilot scope: see
# `include/pto/ccu/pto_gated_kernel.hpp` (the public ABI header). hccl repo
# is intentionally NOT modified by this lib — all coupling is to hcomm
# pkg_inc public headers + libhcomm.so symbols only.
#
# Standalone build (matches `kernels/host/pto_ccu_host/`):
#
#     cmake -S kernels/host/gated_reduce_scatter \
#           -B build/pto_gated_reduce_scatter
#     cmake --build build/pto_gated_reduce_scatter -j
#
# After build, point the ST to the artifact:
#
#     export LD_LIBRARY_PATH=$PWD/build/pto_gated_reduce_scatter/dist:$LD_LIBRARY_PATH
#     # or:
#     export PTO_GATED_RS_LIB=$PWD/build/pto_gated_reduce_scatter/dist/libpto_gated_reduce_scatter.so
#
# This CMakeLists assumes:
#   - $ENV{ASCEND_HOME_PATH} points at a CANN install (e.g.
#     /usr/local/Ascend/ascend-toolkit/latest) that provides:
#       - $ASCEND_HOME_PATH/include/hcomm/ccu/...     (hcomm pkg_inc)
#       - $ASCEND_HOME_PATH/include/hccl/...           (hccl_types.h etc)
#       - $ASCEND_HOME_PATH/lib64/libhcomm.so          (CCU runtime symbols)
#     If hcomm pkg_inc lives elsewhere on the build host (e.g. the in-source
#     `hcomm/pkg_inc/` checkout in development), set
#     `-DHCOMM_PKG_INC_DIR=<abs path to hcomm/pkg_inc>` to override.

cmake_minimum_required(VERSION 3.16)
project(pto_gated_reduce_scatter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ────────────────────────────────────────────────────────────────────────────
# pto-isa repo root (`pto-isa_main/`) — used for resolving public include
# paths like `<pto/ccu/pto_gated_kernel.hpp>` and the registry hpp.
# ────────────────────────────────────────────────────────────────────────────
get_filename_component(PTO_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
message(STATUS "pto_gated_reduce_scatter: PTO_LIB_PATH = ${PTO_LIB_PATH}")

# ────────────────────────────────────────────────────────────────────────────
# CANN toolkit / hcomm pkg_inc resolution.
#
# Priority:
#   1. Explicit `-DHCOMM_PKG_INC_DIR=...` (dev override against in-source
#      hcomm checkout).
#   2. `$ASCEND_HOME_PATH/include` (CANN install, default on dev rigs).
# ────────────────────────────────────────────────────────────────────────────
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(WARNING "pto_gated_reduce_scatter: ASCEND_HOME_PATH not set; "
                "build will fail unless -DHCOMM_PKG_INC_DIR is supplied.")
    endif()
endif()

if(NOT DEFINED HCOMM_PKG_INC_DIR)
    set(HCOMM_PKG_INC_DIR "${ASCEND_HOME_PATH}/include")
endif()
message(STATUS "pto_gated_reduce_scatter: HCOMM_PKG_INC_DIR = ${HCOMM_PKG_INC_DIR}")

if(NOT DEFINED HCOMM_LIB_DIR)
    set(HCOMM_LIB_DIR "${ASCEND_HOME_PATH}/lib64")
endif()
message(STATUS "pto_gated_reduce_scatter: HCOMM_LIB_DIR = ${HCOMM_LIB_DIR}")

# ────────────────────────────────────────────────────────────────────────────
# Library target.
#
# Sources:
#   - pto_gated_kernel.cc — Algorithm() / GeneArgs() / factory
#
# This lib also needs the gate registry (Publish/TryGet) to compile. The
# header is in pto-isa public include, the implementation in
# `kernels/host/pto_ccu_host/pto_gate_registry.cc`. Two options:
#   a) Build pto_ccu_host first and `target_link_libraries(... pto_ccu_host)`.
#   b) Inline the same .cc into our SHARED so the registry is self-contained.
#
# We pick (a) so there's a single registry instance per process — important
# because Publish() / TryGet() both share the same `static std::unordered_map`
# and we don't want two copies fighting for descriptor slots.
# ────────────────────────────────────────────────────────────────────────────
add_library(pto_gated_reduce_scatter SHARED
    pto_gated_kernel.cc
)

target_compile_options(pto_gated_reduce_scatter PRIVATE
    -fPIC
    -fvisibility=hidden  # default-hide, factory `MakeGatedKernelCreator` uses
                         # __attribute__((visibility("default"))) explicitly
                         # via export macro below if extern "C" lookup is
                         # ever needed (currently linked, not dlopen'd).
    -Wall -Wextra
    -O2
)

target_include_directories(pto_gated_reduce_scatter PUBLIC
    ${PTO_LIB_PATH}/include
    ${HCOMM_PKG_INC_DIR}
    ${HCOMM_PKG_INC_DIR}/hcomm
    ${HCOMM_PKG_INC_DIR}/hccl
)

target_link_directories(pto_gated_reduce_scatter PUBLIC
    ${HCOMM_LIB_DIR}
)

# Link order:
#   - hcomm     : CcuKernel base class + CCU IR primitives (CreateVariable,
#                 WaitEvent, RecordEvent, Load, LocalCopyNb)
#   - pthread   : registry uses std::mutex
#
# Registry symbols (`pto::ccu::Publish` / `TryGet`) come from a `pto_ccu_host`
# sibling target. Resolve via add_subdirectory IF the parent build pulled it
# in; otherwise fall back to direct .so link (build host must have built
# `kernels/host/pto_ccu_host` first, OR set `-DPTO_CCU_HOST_LIB=<abs path>`).
if(TARGET pto_ccu_host)
    target_link_libraries(pto_gated_reduce_scatter PRIVATE pto_ccu_host)
elseif(DEFINED PTO_CCU_HOST_LIB)
    target_link_libraries(pto_gated_reduce_scatter PRIVATE ${PTO_CCU_HOST_LIB})
else()
    message(WARNING
        "pto_gated_reduce_scatter: neither `pto_ccu_host` target nor "
        "PTO_CCU_HOST_LIB defined — linking will fail unless the parent build "
        "wires up the registry separately. Build "
        "`kernels/host/pto_ccu_host` first, then re-cmake this with "
        "`-DPTO_CCU_HOST_LIB=<...>/dist/libpto_ccu_host.so`.")
endif()

target_link_libraries(pto_gated_reduce_scatter PRIVATE
    hcomm
    pthread
)

# ────────────────────────────────────────────────────────────────────────────
# Dist directory — copy the built .so to a stable location for ST consumers.
# ────────────────────────────────────────────────────────────────────────────
set(PTO_GATED_RS_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
add_custom_command(TARGET pto_gated_reduce_scatter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_GATED_RS_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pto_gated_reduce_scatter>
        ${PTO_GATED_RS_DIST_DIR}/
)

message(STATUS
    "pto_gated_reduce_scatter: built libpto_gated_reduce_scatter.so → "
    "${PTO_GATED_RS_DIST_DIR}")
