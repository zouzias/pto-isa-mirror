# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_cube_st(textract_small_m)

set(small_m_kernel_source "${CMAKE_CURRENT_SOURCE_DIR}/textract_small_m_kernel.cpp")
set(small_m_cases_header "${CMAKE_CURRENT_SOURCE_DIR}/textract_small_m_cases.h")

# CANN 9.0 auto scheduling can crash when these shape variants share one translation unit.
if(AUTO_MODE)
    set_property(TARGET textract_small_m_kernel PROPERTY SOURCES "")
    foreach(case_key RANGE 1 13)
        set(case_source "${CMAKE_CURRENT_BINARY_DIR}/textract_small_m_${case_key}.cpp")
        file(GENERATE OUTPUT "${case_source}" CONTENT
            "#define PTO_TEXTRACT_SMALL_M_CASE ${case_key}\n#include \"${small_m_kernel_source}\"\n")
        set_source_files_properties("${case_source}" PROPERTIES OBJECT_DEPENDS
            "${small_m_kernel_source};${small_m_cases_header}")
        target_sources(textract_small_m_kernel PRIVATE "${case_source}")
    endforeach()
endif()

# Bisheng's host depfile omits headers included only by the device compilation.
get_filename_component(small_m_pto_include "${PROJECT_SOURCE_DIR}/../../../../../include/pto" ABSOLUTE)
file(GLOB_RECURSE small_m_pto_headers CONFIGURE_DEPENDS
    "${small_m_pto_include}/*.h" "${small_m_pto_include}/*.hpp" "${small_m_pto_include}/*.inl")
get_target_property(small_m_kernel_sources textract_small_m_kernel SOURCES)
set_property(SOURCE ${small_m_kernel_sources} APPEND PROPERTY OBJECT_DEPENDS
    ${small_m_pto_headers} "${small_m_cases_header}")
