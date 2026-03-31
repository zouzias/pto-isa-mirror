# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(pto_test_costmodel LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
    set(CMAKE_CXX_STANDARD 23)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(PTO_COSTMODEL_TEST_ARCH "a2a3" CACHE STRING "Costmodel test architecture subdirectory.")
set_property(CACHE PTO_COSTMODEL_TEST_ARCH PROPERTY STRINGS a2a3 a5)

if (PTO_COSTMODEL_TEST_ARCH STREQUAL "a2a3")
    add_definitions(-D__COSTMODEL -D__NPU_ARCH__=2201 -DPTO_COMM_NOT_SUPPORTED)
elseif (PTO_COSTMODEL_TEST_ARCH STREQUAL "a5")
    add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)
else()
    message(FATAL_ERROR "Unsupported PTO_COSTMODEL_TEST_ARCH: ${PTO_COSTMODEL_TEST_ARCH}")
endif()

message("-- COMPILER: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; Using standard C++${CMAKE_CXX_STANDARD}")

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -std=c++${CMAKE_CXX_STANDARD}
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-O2)
endif()

if (UNIX AND NOT APPLE)
    add_link_options(
        -Wl,-z,relro
        -Wl,-z,now
    )
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_subdirectory(${PTO_COSTMODEL_TEST_ARCH})
