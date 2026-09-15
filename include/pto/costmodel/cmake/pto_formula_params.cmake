# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include_guard(GLOBAL)

# Keep the path available even when functions are called from sibling directories.
get_filename_component(_pto_formula_source "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set_property(GLOBAL PROPERTY PTO_FORMULA_SOURCE "${_pto_formula_source}")

function(pto_prepare_formula_params out_target)
    get_property(_source GLOBAL PROPERTY PTO_FORMULA_SOURCE)
    string(MD5 _id "${_source}")
    set(_target "pto_formula_params_${_id}")
    if(NOT TARGET "${_target}")
        add_library("${_target}" INTERFACE)
        set(_generated "${CMAKE_BINARY_DIR}/pto_costmodel_generated/${_id}/include")
        foreach(_arch a2a3 a5)
            set(_dir "${_source}/${_arch}/formula_costmodel")
            set(_relative "pto/costmodel/${_arch}/formula_costmodel/formula_params_generated.hpp")
            if(EXISTS "${_dir}/gen_formula_params_header.py")
                find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
                # Configure-time generation also makes a direct cmake --install safe.
                # CSV/script edits trigger CMake regeneration before compilation.
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                    "${_dir}/formula_params.csv" "${_dir}/gen_formula_params_header.py")
                execute_process(
                    COMMAND "${Python3_EXECUTABLE}" "${_dir}/gen_formula_params_header.py"
                        --output "${_generated}/${_relative}"
                    RESULT_VARIABLE _result)
                if(NOT _result EQUAL 0)
                    message(FATAL_ERROR "Failed to generate ${_arch} costmodel formula parameters")
                endif()
                set_property(TARGET "${_target}" APPEND PROPERTY PTO_FORMULA_HEADERS "${_generated}/${_relative}")
            elseif(NOT EXISTS "${_dir}/formula_params_generated.hpp")
                message(FATAL_ERROR "PTO installation is missing ${_relative}")
            endif()
        endforeach()
        target_include_directories("${_target}" INTERFACE "${_generated}")
    endif()
    set("${out_target}" "${_target}" PARENT_SCOPE)
endfunction()

function(pto_install_formula_params destination)
    pto_prepare_formula_params(_target)
    get_target_property(_headers "${_target}" PTO_FORMULA_HEADERS)
    foreach(_header IN LISTS _headers)
        get_filename_component(_dir "${_header}" DIRECTORY)
        string(REGEX MATCH "pto/costmodel/[^/]+/formula_costmodel$" _relative "${_dir}")
        install(FILES "${_header}" DESTINATION "${destination}/${_relative}" COMPONENT pto-isa)
    endforeach()
endfunction()
