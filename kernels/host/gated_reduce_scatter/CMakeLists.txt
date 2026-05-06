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
# CANN install layout (per `tests/npu/a5/comm/st/testcase/CMakeLists.txt`):
#   ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu/...   ← hcomm + hccl public headers
#   ${ASCEND_HOME_PATH}/pkg_inc/hccl/...
#   ${ASCEND_HOME_PATH}/include/acl/...          ← ACL runtime
#   ${ASCEND_HOME_PATH}/lib64/libhcomm.so        ← runtime libs
#
# Override priority for HCOMM_PKG_INC_DIR (where `hcomm/ccu/ccu_kernel.h`
# lives, with `hcomm/` as the immediate child of the include root):
#   1. -DHCOMM_PKG_INC_DIR=<abs> at cmake configure time.
#   2. $ENV{HCOMM_PKG_INC_DIR}.
#   3. ${ASCEND_HOME_PATH}/pkg_inc          (CANN install default).
#   4. ${ASCEND_HOME_PATH}/include          (some older CANN layouts).
#   5. <repo>/hcomm/pkg_inc                 (in-source hcomm checkout, dev).
#
# We probe with find_path so configure fails early with a clear error if
# none of the candidates contain the header.
# ────────────────────────────────────────────────────────────────────────────
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    endif()
endif()

if(NOT DEFINED HCOMM_PKG_INC_DIR AND DEFINED ENV{HCOMM_PKG_INC_DIR})
    set(HCOMM_PKG_INC_DIR $ENV{HCOMM_PKG_INC_DIR})
endif()

if(NOT DEFINED HCOMM_PKG_INC_DIR)
    find_path(_HCOMM_PROBE
        NAMES hcomm/ccu/ccu_kernel.h
        PATHS
            ${ASCEND_HOME_PATH}/pkg_inc
            ${ASCEND_HOME_PATH}/include
            ${PTO_LIB_PATH}/../../hcomm/pkg_inc       # sibling hcomm checkout
            $ENV{HOME}/hcomm/pkg_inc
        NO_DEFAULT_PATH
    )
    if(NOT _HCOMM_PROBE)
        message(FATAL_ERROR
            "pto_gated_reduce_scatter: cannot locate hcomm/ccu/ccu_kernel.h. "
            "Searched:\n"
            "  ${ASCEND_HOME_PATH}/pkg_inc\n"
            "  ${ASCEND_HOME_PATH}/include\n"
            "  ${PTO_LIB_PATH}/../../hcomm/pkg_inc\n"
            "  $ENV{HOME}/hcomm/pkg_inc\n"
            "Set -DHCOMM_PKG_INC_DIR=<abs path> (the dir whose immediate child "
            "is `hcomm/`) and re-run cmake.")
    endif()
    set(HCOMM_PKG_INC_DIR ${_HCOMM_PROBE})
endif()
message(STATUS "pto_gated_reduce_scatter: HCOMM_PKG_INC_DIR = ${HCOMM_PKG_INC_DIR}")

# ACL headers (`acl/acl.h` + friends) live under ${ASCEND_HOME_PATH}/include
# on canonical CANN installs.
if(NOT DEFINED CANN_INCLUDE_DIR)
    if(IS_DIRECTORY "${ASCEND_HOME_PATH}/include")
        set(CANN_INCLUDE_DIR "${ASCEND_HOME_PATH}/include")
    endif()
endif()
message(STATUS "pto_gated_reduce_scatter: CANN_INCLUDE_DIR = ${CANN_INCLUDE_DIR}")

# ────────────────────────────────────────────────────────────────────────────
# Locate hcomm internal headers that pkg_inc transitively includes WITHOUT
# any directory prefix (e.g. `#include "hcomm_primitives.h"` instead of
# `#include "hcomm/hcomm_primitives.h"`).
#
# Why this matters: `pkg_inc/hcomm/ccu/ccu_kernel.h` -> `ccu_kernel_arg.h`
# -> `hcomm_primitives.h`, and `ccu_kernel.h` itself -> `hccl_types.h`,
# `hccl_res.h` -> `hcomm_res_defs.h`. None of those internal headers carry
# a directory prefix in the include directive, so the include search list
# must contain whatever directory each of them lives in.
#
# CANN install conventions vary by version:
#   - Some put `hcomm_primitives.h` under `${ASCEND}/include/hcomm/`
#   - Some put `hccl_types.h`        under `${ASCEND}/include/hccl/`
#   - Some put both alternates also under `${ASCEND}/x86_64-linux/include/...`
#     (multi-arch install — observed on user rig 2026-05-06)
#   - Dev rigs use the in-source layout `<repo>/hcomm/include/{hcomm,hccl}/`
#
# Strategy: enumerate a list of representative internal headers, find_path
# each one across all known candidate roots, collect every distinct
# directory into HCOMM_INTERNAL_INC_DIRS. This way one cmake configure run
# discovers every needed include root no matter how the layout splits.
# ────────────────────────────────────────────────────────────────────────────
set(_pto_gated_internal_hdrs
    hcomm_primitives.h
    hccl_types.h
    hcomm_res_defs.h
)

set(_pto_gated_probe_roots
    # Pkg-inc rooted (rare for these flat headers, but cheap to probe).
    ${HCOMM_PKG_INC_DIR}
    ${HCOMM_PKG_INC_DIR}/hcomm
    ${HCOMM_PKG_INC_DIR}/hccl
    # ${ASCEND_HOME_PATH}/include/{hcomm,hccl}
    ${CANN_INCLUDE_DIR}
    ${CANN_INCLUDE_DIR}/hcomm
    ${CANN_INCLUDE_DIR}/hccl
    # Multi-arch CANN installs (e.g. cann-9.0.0/x86_64-linux/include/...).
    ${ASCEND_HOME_PATH}/x86_64-linux/include
    ${ASCEND_HOME_PATH}/x86_64-linux/include/hcomm
    ${ASCEND_HOME_PATH}/x86_64-linux/include/hccl
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include/hcomm
    ${ASCEND_HOME_PATH}/aarch64-linux/include/hccl
    # Dev rig — sibling hcomm source checkout next to pto-isa.
    ${PTO_LIB_PATH}/../../hcomm/include
    ${PTO_LIB_PATH}/../../hcomm/include/hcomm
    ${PTO_LIB_PATH}/../../hcomm/include/hccl
    $ENV{HOME}/hcomm/include
    $ENV{HOME}/hcomm/include/hcomm
    $ENV{HOME}/hcomm/include/hccl
)

set(HCOMM_INTERNAL_INC_DIRS "")
set(_missing_internal_hdrs "")
foreach(_hdr ${_pto_gated_internal_hdrs})
    string(MAKE_C_IDENTIFIER "_PROBE_${_hdr}" _probe_var)
    find_path(${_probe_var}
        NAMES ${_hdr}
        PATHS ${_pto_gated_probe_roots}
        NO_DEFAULT_PATH
    )
    if(${_probe_var})
        message(STATUS
            "pto_gated_reduce_scatter:   ${_hdr} -> ${${_probe_var}}")
        list(APPEND HCOMM_INTERNAL_INC_DIRS ${${_probe_var}})
    else()
        list(APPEND _missing_internal_hdrs ${_hdr})
    endif()
endforeach()

if(_missing_internal_hdrs)
    message(FATAL_ERROR
        "pto_gated_reduce_scatter: cannot locate the following hcomm "
        "internal header(s): ${_missing_internal_hdrs}\n"
        "Searched roots:\n  ${_pto_gated_probe_roots}\n"
        "Run e.g. `find $ASCEND_HOME_PATH ~/hcomm -name <hdr> 2>/dev/null` "
        "to locate them, then add the parent dir(s) via "
        "-DHCOMM_PKG_INC_DIR / -DCANN_INCLUDE_DIR or extend the probe list.")
endif()

list(REMOVE_DUPLICATES HCOMM_INTERNAL_INC_DIRS)
message(STATUS
    "pto_gated_reduce_scatter: HCOMM_INTERNAL_INC_DIRS = "
    "${HCOMM_INTERNAL_INC_DIRS}")

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
    ${HCOMM_INTERNAL_INC_DIRS}
)
if(DEFINED CANN_INCLUDE_DIR)
    target_include_directories(pto_gated_reduce_scatter PUBLIC ${CANN_INCLUDE_DIR})
endif()

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
