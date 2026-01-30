# coding=utf-8
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
project(JIT_GEMM)

set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if("${GLIBCXX_USE_CXX11_ABI}" STREQUAL "0" OR "${GLIBCXX_USE_CXX11_ABI}" STREQUAL "1")
    message(STATUS "Using _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
  else()
    message(WARNING "GLIBCXX_USE_CXX11_ABI is set but not 0/1 (got '${GLIBCXX_USE_CXX11_ABI}'). Not defining _GLIBCXX_USE_CXX11_ABI.")
  endif()
endif()

# user-defined configuration
set(SOC_VERSION "ascend910b1" CACHE STRING "system on chip type")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(PTO_LIB_PATH "$ENV{PTO_LIB_PATH}" CACHE STRING "PTO library path")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB _OPPLUGIN_PYBIND_OPS_SRCS csrc/host/*.cpp)

set(_OPPLUGIN_PYBIND_NAME gemm_jit)

set_source_files_properties(
  csrc/kernel/gemm_kernel.cpp
  PROPERTIES
  COMPILE_FLAGS "-xcce"
)

# ===== Kernel (AiCore) =====
ascendc_library(gemm_kernel STATIC
    csrc/kernel/gemm_kernel.cpp
)

ascendc_include_directories(gemm_kernel PRIVATE
    ${PTO_LIB_PATH}/include
    ${PTO_LIB_PATH}/include/common
)

add_library(${_OPPLUGIN_PYBIND_NAME} SHARED ${_OPPLUGIN_PYBIND_OPS_SRCS})

target_link_libraries(${_OPPLUGIN_PYBIND_NAME} PRIVATE
  gemm_kernel
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
)

target_include_directories(gemm_jit PRIVATE
  "${CMAKE_BINARY_DIR}/include"
  "${PTO_LIB_PATH}/include"
  "${PTO_LIB_PATH}/include/common"
)


