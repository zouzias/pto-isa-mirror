# -----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------
#
# pto_ccu_host — host-side helper library for PTO gated CCU kernels (Step 3+).
#
# One output:
#   - libpto_ccu_host.so : Publish/TryGet registry + extern "C"
#                          `PtoCcuTryGetLastReduceScatterGateDescriptor` dlsym
#                          target, consumed by example main.cc to read the gate
#                          descriptor published by the in-process gated CCU
#                          kernel after `HcclReduceScatter` has been enqueued.
#
# This library is plain C++ (no AscendC / no CCE), built with the system host
# compiler (g++). Kept distinct from `kernels/manual/a5/treduce/` (which is
# bisheng/CCE for the AIV side) to avoid leaking CCE flags into the host TU.
#
# This CMakeLists is standalone-buildable, mirroring
# `kernels/manual/a5/treduce/CMakeLists.txt`. From the project root:
#
#   cmake -S kernels/host/pto_ccu_host -B build/pto_ccu_host
#   cmake --build build/pto_ccu_host -j
#
# Migration history (pto-isa pilot phase 1, 2026-05-06): implementation
# semantics lifted as-is from
# `hccl/src/ops/reduce/template/ccu/kernel/pto_gate_registry.{h,cc}`, but
# hccl source is not modified during phase 1 — both registries exist in
# parallel and example main.cc round-trips through pto-isa when
# `HCCL_PTO_USE_ISA_GATED=1` is set, validating the new lib without disturbing
# the existing flow. Phase 2 will retire the hccl-side registry entirely once
# hcomm `HcclThreadAcquire` and hccl `CcuKernelArg*` types reach pkg_inc.
# See plan `.cursor/plans/pto-isa_pilot_+_d2_host_fix_0f29f560.plan.md` Part A.

cmake_minimum_required(VERSION 3.16)
project(pto_ccu_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Where the public pto headers live. `${CMAKE_CURRENT_LIST_DIR}/../../..`
# resolves to the pto-isa repo root.
get_filename_component(PTO_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
message(STATUS "pto_ccu_host: PTO_LIB_PATH = ${PTO_LIB_PATH}")

add_library(pto_ccu_host SHARED pto_gate_registry.cc)

target_compile_options(pto_ccu_host PRIVATE
    -fPIC
    # NOTE (phase 2 update, 2026-05-06): originally `-fvisibility=hidden` was
    # set here because phase 1 only consumed this lib via dlsym on the
    # `extern "C"` bridge symbols (`PtoCcuPublish_C` /
    # `PtoCcuTryGetLastReduceScatterGateDescriptor`). Phase 2 (gated kernel
    # migration) now also *links* against this lib for the C++ API
    # `pto::ccu::Publish` / `pto::ccu::TryGet` — both from the gated kernel
    # lib (`libpto_gated_reduce_scatter.so`) and from the standalone ST
    # (`gated_rs_st`). Hidden visibility here would demote those C++ symbols
    # to STV_HIDDEN and break the link
    # (`undefined reference to pto::ccu::TryGet(...)`, observed 2026-05-06).
    # If we ever need a stricter ABI surface, introduce a unified `PTO_API`
    # macro and re-enable hidden.
    -Wall -Wextra
    -O2
)

target_include_directories(pto_ccu_host PUBLIC
    ${PTO_LIB_PATH}/include
)

# Pure std::unordered_map + std::mutex; pthread is the only transitive runtime
# dep. Linked explicitly so static-libstdc++ builds also work.
target_link_libraries(pto_ccu_host PRIVATE pthread)

# Mirror `treduce/CMakeLists.txt:136-142`: copy the .so to a known dist/ dir
# so users can `LD_LIBRARY_PATH=<build>/dist:$LD_LIBRARY_PATH` (or set
# `PTO_CCU_HOST_LIB=<build>/dist/libpto_ccu_host.so`) without chasing CMake's
# generator-specific output tree.
set(PTO_CCU_HOST_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
add_custom_command(TARGET pto_ccu_host POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_CCU_HOST_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pto_ccu_host>
        ${PTO_CCU_HOST_DIST_DIR}/
)

message(STATUS "pto_ccu_host: built libpto_ccu_host.so → ${PTO_CCU_HOST_DIST_DIR}")
