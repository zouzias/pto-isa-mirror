# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/version.info" PTO_ISA_VERSION_LINE REGEX "^Version=")
string(REPLACE "Version=" "" PTO_ISA_VERSION "${PTO_ISA_VERSION_LINE}")

project(pto_isa VERSION "${PTO_ISA_VERSION}" LANGUAGES C CXX)

option(PTO_ISA_ENABLE_CMAKE_PACKAGE "Generate/install CMake package config (find_package)" ON)
option(PTO_ISA_INSTALL_CMAKE_PACKAGE "Install CMake package config (requires install step)" OFF)

set(PTO_ISA_BACKEND "auto" CACHE STRING "Backend: auto|cpu|npu")
set_property(CACHE PTO_ISA_BACKEND PROPERTY STRINGS auto cpu npu)
set(PTO_ISA_NPU_ADDRESSING "auto" CACHE STRING "NPU addressing: auto|memory|register")
set_property(CACHE PTO_ISA_NPU_ADDRESSING PROPERTY STRINGS auto memory register)

add_library(pto-isa-includes INTERFACE)
add_library(pto-isa::includes ALIAS pto-isa-includes)
set_target_properties(pto-isa-includes PROPERTIES EXPORT_NAME includes)

target_include_directories(pto-isa-includes INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(pto-isa-headers INTERFACE)
add_library(pto-isa::headers ALIAS pto-isa-headers)
target_link_libraries(pto-isa-headers INTERFACE pto-isa-includes)
set_target_properties(pto-isa-headers PROPERTIES EXPORT_NAME headers)

set(_PTO_ISA_DETECTED_BACKEND "")
set(_PTO_ISA_DETECTED_NPU_ADDR "")

if (PTO_ISA_BACKEND STREQUAL "cpu")
  set(_PTO_ISA_DETECTED_BACKEND "cpu")
elseif(PTO_ISA_BACKEND STREQUAL "npu")
  set(_PTO_ISA_DETECTED_BACKEND "npu")
elseif(PTO_ISA_BACKEND STREQUAL "auto")
  find_program(_PTO_ISA_NPU_SMI NAMES npu-smi npu_smi)
  if (_PTO_ISA_NPU_SMI)
    execute_process(
      COMMAND "${_PTO_ISA_NPU_SMI}" info
      TIMEOUT 2
      RESULT_VARIABLE _PTO_ISA_SMI_RC
      OUTPUT_VARIABLE _PTO_ISA_SMI_OUT
      ERROR_VARIABLE _PTO_ISA_SMI_ERR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    if (_PTO_ISA_SMI_RC EQUAL 0)
      string(TOLOWER "${_PTO_ISA_SMI_OUT}" _PTO_ISA_SMI_OUT_LC)
      # Rules:
      # - 910B1/910B2/910C* => MEMORY_BASE
      # - 950XX            => REGISTER_BASE
      if (_PTO_ISA_SMI_OUT_LC MATCHES "910b|910c")
        set(_PTO_ISA_DETECTED_BACKEND "npu")
        set(_PTO_ISA_DETECTED_NPU_ADDR "memory")
      elseif(_PTO_ISA_SMI_OUT_LC MATCHES "950")
        set(_PTO_ISA_DETECTED_BACKEND "npu")
        set(_PTO_ISA_DETECTED_NPU_ADDR "register")
      endif()
    endif()
  endif()

  if (NOT _PTO_ISA_DETECTED_BACKEND)
    # No usable npu-smi -> default to CPU simulation build.
    set(_PTO_ISA_DETECTED_BACKEND "cpu")
  endif()
else()
  message(FATAL_ERROR "Invalid PTO_ISA_BACKEND='${PTO_ISA_BACKEND}'. Use auto|cpu|npu.")
endif()

if (_PTO_ISA_DETECTED_BACKEND STREQUAL "cpu")
  target_compile_definitions(pto-isa-headers INTERFACE __CPU_SIM)
  message(STATUS "pto-isa: backend=cpu (define __CPU_SIM)")
elseif(_PTO_ISA_DETECTED_BACKEND STREQUAL "npu")
  if (PTO_ISA_NPU_ADDRESSING STREQUAL "memory")
    set(_PTO_ISA_DETECTED_NPU_ADDR "memory")
  elseif(PTO_ISA_NPU_ADDRESSING STREQUAL "register")
    set(_PTO_ISA_DETECTED_NPU_ADDR "register")
  elseif(PTO_ISA_NPU_ADDRESSING STREQUAL "auto")
    # keep detected
  else()
    message(FATAL_ERROR "Invalid PTO_ISA_NPU_ADDRESSING='${PTO_ISA_NPU_ADDRESSING}'. Use auto|memory|register.")
  endif()

  if (_PTO_ISA_DETECTED_NPU_ADDR STREQUAL "memory")
    target_compile_definitions(pto-isa-headers INTERFACE MEMORY_BASE)
    message(STATUS "pto-isa: backend=npu addressing=memory (define MEMORY_BASE)")
  elseif(_PTO_ISA_DETECTED_NPU_ADDR STREQUAL "register")
    target_compile_definitions(pto-isa-headers INTERFACE REGISTER_BASE)
    message(STATUS "pto-isa: backend=npu addressing=register (define REGISTER_BASE)")
  else()
    message(FATAL_ERROR "pto-isa: backend=npu but addressing could not be determined; set -DPTO_ISA_NPU_ADDRESSING=memory|register.")
  endif()
endif()

if (PTO_ISA_ENABLE_CMAKE_PACKAGE)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(PTO_ISA_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/pto-isa")

  export(
    TARGETS pto-isa-includes pto-isa-headers
    NAMESPACE pto-isa::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/pto-isaTargets.cmake"
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/pto-isaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pto-isaConfig.cmake"
    INSTALL_DESTINATION "${PTO_ISA_CMAKE_INSTALL_DIR}"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pto-isaConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  if (PTO_ISA_INSTALL_CMAKE_PACKAGE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(TARGETS pto-isa-includes pto-isa-headers EXPORT pto-isaTargets)
    install(
      EXPORT pto-isaTargets
      FILE pto-isaTargets.cmake
      NAMESPACE pto-isa::
      DESTINATION "${PTO_ISA_CMAKE_INSTALL_DIR}"
    )
    install(
      FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pto-isaConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pto-isaConfigVersion.cmake"
      DESTINATION "${PTO_ISA_CMAKE_INSTALL_DIR}"
    )
  endif()
endif()

set(_PTO_ISA_TOP_LEVEL OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(_PTO_ISA_TOP_LEVEL ON)
endif()

option(PTO_ISA_ENABLE_RUN_PACKAGE "Enable makeself run-package packaging (top-level only)" ${_PTO_ISA_TOP_LEVEL})

if (PTO_ISA_ENABLE_RUN_PACKAGE AND PTO_ISA_INSTALL_CMAKE_PACKAGE)
  message(FATAL_ERROR "PTO_ISA_ENABLE_RUN_PACKAGE and PTO_ISA_INSTALL_CMAKE_PACKAGE cannot be enabled at the same time.")
endif()

if (PTO_ISA_ENABLE_RUN_PACKAGE)
  include(cmake/package.cmake)
  pack_built_in()
endif()
