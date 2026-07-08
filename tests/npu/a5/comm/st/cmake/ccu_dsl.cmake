# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

option(PTO_ENABLE_CCU_DSL "Enable CCU DSL support through an external SotaCompiler sotac executable" OFF)
set(PTO_SOTAC_EXECUTABLE "" CACHE FILEPATH "Path to the SotaCompiler sotac executable")
set(PTO_CCU_ADAPTER_SO "" CACHE FILEPATH "Path to libccu_offline_adapter_hcomm.so")
set(PTO_CCU_ADAPTER_SOURCE_DIR "" CACHE PATH "Path to the CCU adapter source tree")
set(PTO_CCU_ADAPTER_INCLUDE_DIR "" CACHE PATH "Path containing ccu_offline_adapter_c.h")

foreach(_pto_ccu_dsl_var
        PTO_SOTAC_EXECUTABLE
        PTO_CCU_ADAPTER_SO
        PTO_CCU_ADAPTER_SOURCE_DIR
        PTO_CCU_ADAPTER_INCLUDE_DIR)
    if((NOT DEFINED ${_pto_ccu_dsl_var} OR "${${_pto_ccu_dsl_var}}" STREQUAL "")
            AND DEFINED ENV{${_pto_ccu_dsl_var}})
        set(${_pto_ccu_dsl_var} "$ENV{${_pto_ccu_dsl_var}}")
    endif()
endforeach()

function(pto_require_sotac OUT_EXE)
    if(NOT PTO_SOTAC_EXECUTABLE)
        message(FATAL_ERROR
            "PTO_ENABLE_CCU_DSL=ON requires PTO_SOTAC_EXECUTABLE. "
            "Download or build sotac and pass -DPTO_SOTAC_EXECUTABLE=/path/to/sotac.")
    endif()
    if(NOT EXISTS "${PTO_SOTAC_EXECUTABLE}")
        message(FATAL_ERROR "PTO_SOTAC_EXECUTABLE does not exist: ${PTO_SOTAC_EXECUTABLE}")
    endif()
    set(${OUT_EXE} "${PTO_SOTAC_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(pto_resolve_ccu_adapter_include OUT_INCLUDE_DIR)
    set(_adapter_include "${PTO_CCU_ADAPTER_INCLUDE_DIR}")
    if(NOT _adapter_include AND PTO_CCU_ADAPTER_SOURCE_DIR)
        set(_candidate "${PTO_CCU_ADAPTER_SOURCE_DIR}/hcomm/include")
        if(EXISTS "${_candidate}/ccu_offline_adapter_c.h")
            set(_adapter_include "${_candidate}")
        endif()
    endif()
    if(NOT _adapter_include OR NOT EXISTS "${_adapter_include}/ccu_offline_adapter_c.h")
        message(FATAL_ERROR
            "CCU DSL requires ccu_offline_adapter_c.h. "
            "Pass -DPTO_CCU_ADAPTER_INCLUDE_DIR=/path/to/adapter/hcomm/include "
            "or -DPTO_CCU_ADAPTER_SOURCE_DIR=/path/to/adapter.")
    endif()
    set(${OUT_INCLUDE_DIR} "${_adapter_include}" PARENT_SCOPE)
endfunction()

function(pto_import_ccu_adapter OUT_TARGET)
    if(TARGET pto_ccu_offline_adapter)
        set(${OUT_TARGET} pto_ccu_offline_adapter PARENT_SCOPE)
        return()
    endif()

    pto_resolve_ccu_adapter_include(_adapter_include)

    set(_adapter_so "${PTO_CCU_ADAPTER_SO}")
    if(NOT _adapter_so AND PTO_CCU_ADAPTER_SOURCE_DIR)
        set(_candidate "${PTO_CCU_ADAPTER_SOURCE_DIR}/build/hcomm/libccu_offline_adapter_hcomm.so")
        if(EXISTS "${_candidate}")
            set(_adapter_so "${_candidate}")
        endif()
    endif()
    if(_adapter_so AND NOT EXISTS "${_adapter_so}")
        message(FATAL_ERROR "PTO_CCU_ADAPTER_SO does not exist: ${_adapter_so}")
    endif()

    if(NOT _adapter_so)
        message(FATAL_ERROR
            "CCU DSL requires an existing CCU offline adapter shared library. "
            "Pass -DPTO_CCU_ADAPTER_SO=/path/to/libccu_offline_adapter_hcomm.so "
            "or -DPTO_CCU_ADAPTER_SOURCE_DIR=/path/to/adapter with build/hcomm/libccu_offline_adapter_hcomm.so.")
    endif()

    add_library(pto_ccu_offline_adapter SHARED IMPORTED GLOBAL)
    set_target_properties(pto_ccu_offline_adapter PROPERTIES
        IMPORTED_LOCATION "${_adapter_so}"
        INTERFACE_INCLUDE_DIRECTORIES "${_adapter_include}"
    )
    set(${OUT_TARGET} pto_ccu_offline_adapter PARENT_SCOPE)
endfunction()

function(pto_ccu_dsl_compile)
    cmake_parse_arguments(PTO_DSL "" "TARGET;SOURCE;OUT_DIR;TARGET_VERSION" "" ${ARGN})
    if(NOT PTO_DSL_TARGET OR NOT PTO_DSL_SOURCE OR NOT PTO_DSL_OUT_DIR)
        message(FATAL_ERROR "pto_ccu_dsl_compile requires TARGET, SOURCE and OUT_DIR")
    endif()
    if(NOT PTO_DSL_TARGET_VERSION)
        set(PTO_DSL_TARGET_VERSION "100")
    endif()
    pto_require_sotac(_sotac)
    set(_stamp "${PTO_DSL_OUT_DIR}/.sotac.stamp")
    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${PTO_DSL_OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PTO_DSL_OUT_DIR}"
        COMMAND "${_sotac}" "${PTO_DSL_SOURCE}" "--backend-version=${PTO_DSL_TARGET_VERSION}" -o "${PTO_DSL_OUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
        DEPENDS "${PTO_DSL_SOURCE}"
        COMMENT "Compiling CCU DSL ${PTO_DSL_SOURCE}"
        VERBATIM
    )
    add_custom_target(${PTO_DSL_TARGET} DEPENDS "${_stamp}")
endfunction()

function(pto_comm_ccu_dsl_st NAME DSL_SOURCE)
    if(NOT PTO_ENABLE_CCU_DSL)
        message(FATAL_ERROR "CCU DSL testcase '${NAME}' requires -DPTO_ENABLE_CCU_DSL=ON")
    endif()

    pto_import_ccu_adapter(_adapter_target)

    pto_comm_ccu_st(${NAME})

    set(_dsl_out_dir "${CMAKE_CURRENT_BINARY_DIR}/ccu_dsl_out")
    pto_ccu_dsl_compile(
        TARGET ${NAME}_microcode
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${DSL_SOURCE}"
        OUT_DIR "${_dsl_out_dir}"
        TARGET_VERSION 100
    )

    add_dependencies(${NAME} ${NAME}_microcode)
    target_link_libraries(${NAME} PRIVATE ${_adapter_target})
    target_compile_definitions(${NAME} PRIVATE
        PTO_CCU_DSL_TX_MICROCODE="${_dsl_out_dir}/microcode/000_tgather_ccu_dsl_tx/microcode_output.txt"
        PTO_CCU_DSL_RX_MICROCODE="${_dsl_out_dir}/microcode/001_tgather_ccu_dsl_rx/microcode_output.txt"
    )
endfunction()
