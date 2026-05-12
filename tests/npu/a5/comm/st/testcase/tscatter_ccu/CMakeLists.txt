# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# --------------------------------------------------------------------------------
#
# tscatter_ccu — CCU-based TScatter ST (standalone build).
#
# Builds two targets:
#   1. tscatter_ccu_kernel (shared lib, bisheng) — AIV CKE trigger kernel
#   2. tscatter_ccu_st     (executable, g++)     — host test driver
#
# The CCU backend (registry, kernel types) is header-only, no extra lib needed.
#
# Build:
#   cmake -S tests/npu/a5/comm/st/testcase/tscatter_ccu -B build/tscatter_ccu_st \
#       -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
#   cmake --build build/tscatter_ccu_st -j
#
# Run:
#   mpirun -n 2 ./tscatter_ccu_st

cmake_minimum_required(VERSION 3.16)
project(tscatter_ccu_st LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED PTO_CXX11_ABI)
    if(DEFINED ENV{PTO_CXX11_ABI})
        set(PTO_CXX11_ABI $ENV{PTO_CXX11_ABI})
    else()
        set(PTO_CXX11_ABI 0)
    endif()
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${PTO_CXX11_ABI})

# pto-isa repo root (7 levels up from this CMakeLists)
get_filename_component(PTO_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../../../.." ABSOLUTE)
message(STATUS "tscatter_ccu_st: PTO_LIB_PATH = ${PTO_LIB_PATH}")

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
            ${PTO_LIB_PATH}/../../hcomm/pkg_inc
        NO_DEFAULT_PATH
    )
    if(_HCOMM_PROBE)
        set(HCOMM_PKG_INC_DIR ${_HCOMM_PROBE})
    endif()
endif()

if(NOT DEFINED CANN_INCLUDE_DIR)
    if(IS_DIRECTORY "${ASCEND_HOME_PATH}/include")
        set(CANN_INCLUDE_DIR "${ASCEND_HOME_PATH}/include")
    endif()
endif()

# Host-arch kernel platform root
set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
if(NOT EXISTS "${PTO_TOOLKIT_DIR}")
    set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

# AscendC kernel headers (tikcpp path varies by CANN version)
set(PTO_ASCENDC_KERNEL_INCLUDE "")
if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
    set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
    set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
endif()

# ===========================================================================
# 1. AIV trigger kernel (bisheng-compiled shared library)
# ===========================================================================
if(NOT DEFINED CCE_AICORE_ARCH)
    set(CCE_AICORE_ARCH "dav-c310-vec")
endif()

add_library(tscatter_ccu_kernel SHARED tscatter_ccu_kernel.cpp)

target_compile_options(tscatter_ccu_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    --cce-aicore-arch=${CCE_AICORE_ARCH}
    -DREGISTER_BASE
    -std=c++17
    -Wno-non-c-typedef-for-linkage
    -Wno-braced-scalar-init
)

target_include_directories(tscatter_ccu_kernel PRIVATE
    ${PTO_LIB_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    ${PTO_TOOLKIT_DIR}/asc
    ${PTO_TOOLKIT_DIR}/asc/include
    ${PTO_TOOLKIT_DIR}/asc/include/interface
    ${PTO_TOOLKIT_DIR}/asc/include/utils
    ${PTO_TOOLKIT_DIR}/asc/impl/basic_api
    ${PTO_TOOLKIT_DIR}/ascendc/include
    ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api
    ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api/impl
    ${PTO_TOOLKIT_DIR}/include/ascendc
    ${PTO_TOOLKIT_DIR}/include/ascendc/basic_api
    ${PTO_TOOLKIT_DIR}/ascendc/include/highlevel_api
    ${PTO_ASCENDC_KERNEL_INCLUDE}
)

target_link_directories(tscatter_ccu_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(tscatter_ccu_kernel PRIVATE runtime)
target_link_options(tscatter_ccu_kernel PRIVATE --cce-fatobj-link)

# ===========================================================================
# 2. Host test driver (g++-compiled executable)
# ===========================================================================
add_executable(tscatter_ccu_st main.cc)

target_compile_options(tscatter_ccu_st PRIVATE -Wall -Wextra -O2)

target_include_directories(tscatter_ccu_st PRIVATE
    ${PTO_LIB_PATH}/include
    ${HCOMM_PKG_INC_DIR}
    ${CANN_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${ASCEND_HOME_PATH}/x86_64-linux/include
    ${PTO_TOOLKIT_DIR}/include/hccl
)

target_link_directories(tscatter_ccu_st PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)

# GTest (follows costmodel/st pattern): prefer system, FetchContent fallback.
# Project uses _GLIBCXX_USE_CXX11_ABI=0; if system GTest was built with =1,
# symbols like GetBoolAssertionFailureMessage won't link — fall back to source build.
find_package(GTest QUIET)
if(GTest_FOUND AND UNIX)
    get_target_property(_gtest_loc GTest::gtest IMPORTED_LOCATION)
    if(NOT _gtest_loc)
        get_target_property(_gtest_loc GTest::gtest LOCATION)
    endif()
    if(_gtest_loc AND EXISTS "${_gtest_loc}")
        execute_process(
            COMMAND bash -c "nm -C '${_gtest_loc}' 2>/dev/null | grep -q 'GetBoolAssertionFailureMessage\\[abi:cxx11\\]'"
            RESULT_VARIABLE _gtest_has_cxx11)
        if(_gtest_has_cxx11 EQUAL 0)
            message(STATUS "System GTest uses CXX11 ABI but project needs ABI=0; will FetchContent instead")
            set(GTest_FOUND FALSE)
        endif()
    endif()
    unset(_gtest_loc)
    unset(_gtest_has_cxx11)
endif()

if(GTest_FOUND)
    set(PTO_GTEST_TARGET GTest::gtest)
else()
    include(FetchContent)
    message(STATUS "Fetching googletest v1.14.0 via FetchContent")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
    FetchContent_MakeAvailable(googletest)
    set(PTO_GTEST_TARGET gtest)
endif()

target_link_libraries(tscatter_ccu_st PRIVATE
    tscatter_ccu_kernel
    hccl hcomm ascendcl runtime
    ${PTO_GTEST_TARGET}
    stdc++ c_sec dl pthread m
)

set(PTO_ST_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
add_custom_command(TARGET tscatter_ccu_st POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PTO_ST_DIST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:tscatter_ccu_st>
        $<TARGET_FILE:tscatter_ccu_kernel>
        ${PTO_ST_DIST_DIR}/
)
