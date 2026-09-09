# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include_guard(GLOBAL)

get_filename_component(_pto_costmodel_public_dir "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_pto_costmodel_root "${_pto_costmodel_public_dir}/../../../.." REALPATH)

# Enable PTO Host costmodel for a consumer target. The consumer uses this same
# entry point for both architectures; A5's VfSim dependencies stay internal.
#
#   include(<pto-root>/include/pto/costmodel/cmake/pto_costmodel.cmake)
#   pto_enable_costmodel(my_target ARCH A2A3) # or A5
function(pto_enable_costmodel target)
    cmake_parse_arguments(PTO_COSTMODEL "" "ARCH" "" ${ARGN})
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "pto_enable_costmodel: target '${target}' does not exist")
    endif()
    if(NOT PTO_COSTMODEL_ARCH)
        message(FATAL_ERROR "pto_enable_costmodel: ARCH must be A2A3 or A5")
    endif()

    string(TOUPPER "${PTO_COSTMODEL_ARCH}" _pto_costmodel_arch)
    target_include_directories("${target}" PRIVATE "${_pto_costmodel_root}/include")
    target_compile_definitions("${target}" PRIVATE __COSTMODEL PTO_COMM_NOT_SUPPORTED)

    if(_pto_costmodel_arch STREQUAL "A2A3")
        target_compile_definitions("${target}" PRIVATE __NPU_ARCH__=2201)
        return()
    endif()

    if(_pto_costmodel_arch STREQUAL "A5")
        target_compile_definitions("${target}" PRIVATE __NPU_ARCH__=3101)
        include("${_pto_costmodel_root}/pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake")
        target_enable_a5_vf_mock("${target}")
        return()
    endif()

    message(FATAL_ERROR "pto_enable_costmodel: unsupported ARCH '${PTO_COSTMODEL_ARCH}'; use A2A3 or A5")
endfunction()
