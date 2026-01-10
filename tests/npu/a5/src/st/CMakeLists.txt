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
project(pto_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

find_package(GTest REQUIRED)

# Some testcases depend on low-level runtime headers like `runtime/rt.h`,
# which may live under `.../pkg_inc/runtime` in some CANN installations.
set(_ascend_runtime_inc_candidates
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
)
file(GLOB _ascend_runtime_inc_glob
    "/usr/local/Ascend/cann-*/aarch64-linux/pkg_inc/runtime"
    "/usr/local/Ascend/cann-*/x86_64-linux/pkg_inc/runtime"
)
list(APPEND _ascend_runtime_inc_candidates ${_ascend_runtime_inc_glob})
list(REMOVE_DUPLICATES _ascend_runtime_inc_candidates)
find_path(ASCEND_RUNTIME_INCLUDE_DIR
    NAMES runtime/rt.h
    PATHS ${_ascend_runtime_inc_candidates}
    NO_DEFAULT_PATH
)
if(NOT ASCEND_RUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find runtime/rt.h. Checked: ${_ascend_runtime_inc_candidates}")
endif()
get_filename_component(ASCEND_PKG_INC_DIR "${ASCEND_RUNTIME_INCLUDE_DIR}" DIRECTORY)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes
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

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_PKG_INC_DIR}
    ${ASCEND_PKG_INC_DIR}/profiling
    ${ASCEND_RUNTIME_INCLUDE_DIR}
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

add_subdirectory(testcase)
