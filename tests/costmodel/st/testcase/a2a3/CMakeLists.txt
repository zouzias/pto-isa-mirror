# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Costmodel STs for the NPU A2/A3 backend.
#
# Each op has a single self-contained source file `<op>/<op>.cpp` that
# directly exercises the PTO instruction and checks the cycle reported by
# `tile.GetCycle()` against a calibrated expected value (see cost_check.hpp).

function(pto_costmodel_a2a3_st NAME)
    add_executable(${NAME} ${NAME}/${NAME}.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -std=c++23)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../include/
        ${PROJECT_SOURCE_DIR}/../../../include/common
        ${PROJECT_SOURCE_DIR}/common
    )
    target_link_libraries(${NAME} PRIVATE
        GTest::gtest_main
        Threads::Threads
    )
    if (UNIX)
        target_link_libraries(${NAME} PRIVATE m)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(${NAME} PRIVATE dl)
    endif()
endfunction()

set(A2A3_TESTCASES
    tabs
    tadd
    tadds
    tcmp
    tcolmax
    tcolmin
    tcolsum
    tdiv
    tdivs
    texp
    tmins
    tmax
    tmaxs
    tmin
    tmul
    tmuls
    tmatmul
    tneg
    trowexpand
    trowmax
    trowmin
    trowsum
    trecip
    trsqrt
    tsqrt
    tsubs
    tsub
    tgather
    tcvt
    tmrgsort
    tsel
    tsels
    tsort32
    ttrans
    tload
    tloadconv
    tmov
    textract
    tscatter
)

foreach(TC ${A2A3_TESTCASES})
    if(NOT DEFINED TEST_CASE OR TEST_CASE STREQUAL "${TC}")
        pto_costmodel_a2a3_st(${TC})
    endif()
endforeach()
