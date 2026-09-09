# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT TEST_EXECUTABLE OR NOT VFSIM_CONFIGS OR NOT RELOCATED_ROOT)
    message(FATAL_ERROR "relocated VfSim test is missing an input path")
endif()

set(_bin_dir "${RELOCATED_ROOT}/bin")
set(_config_parent "${RELOCATED_ROOT}/pkg_inc/pto/costmodel/vfsim")
set(_run_dir "${RELOCATED_ROOT}/run")
file(MAKE_DIRECTORY "${_bin_dir}" "${_config_parent}" "${_run_dir}")
file(COPY "${TEST_EXECUTABLE}" DESTINATION "${_bin_dir}")
file(COPY "${VFSIM_CONFIGS}" DESTINATION "${_config_parent}")

get_filename_component(_executable_name "${TEST_EXECUTABLE}" NAME)
execute_process(
    COMMAND "${_bin_dir}/${_executable_name}" --gtest_filter=TAdd.float_1x64
    WORKING_DIRECTORY "${_run_dir}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "relocated VfSim prediction failed:\n${_output}\n${_error}")
endif()
