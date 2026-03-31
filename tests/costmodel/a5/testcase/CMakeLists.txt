# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_costmodel_st NAME)
    set(FILES_LIST main.cpp)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_kernel.cpp)
        list(APPEND FILES_LIST ${NAME}_kernel.cpp)
    endif()
    add_executable(${NAME} ${FILES_LIST})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -std=c++${CMAKE_CXX_STANDARD})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../include
        ${PROJECT_SOURCE_DIR}/common
    )

    target_link_libraries(${NAME} PRIVATE Threads::Threads)
    if (UNIX)
        target_link_libraries(${NAME} PRIVATE m)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(${NAME} PRIVATE dl)
    endif()
endfunction()

find_package(Threads REQUIRED)

set(ALL_TESTCASES
    runtime_stub
    tload
    tadd
    tadds
    tsub
    tsubs
    tmul
    tmuls
    tdiv
    tdivs
    tmax
    tmaxs
    tmin
    tmins
    tabs
    taxpy
    tneg
    texp
    tlog
    tsqrt
    trsqrt
    tnot
    tand
    tands
    tor
    tors
    txor
    txors
    tshl
    tshls
    tshr
    tshrs
    tlrelu
    trelu
    tstore
)

set(PTO_COSTMODEL_CASES "all" CACHE STRING "Semicolon-separated list of costmodel cases to add. Use all to add every case.")
if (NOT PTO_COSTMODEL_CASES STREQUAL "all")
    string(REPLACE "," ";" PTO_COSTMODEL_CASES "${PTO_COSTMODEL_CASES}")
    set(SELECTED_TESTCASES ${PTO_COSTMODEL_CASES})
else()
    set(SELECTED_TESTCASES ${ALL_TESTCASES})
endif()

foreach(TESTCASE IN LISTS SELECTED_TESTCASES)
    if (NOT TESTCASE IN_LIST ALL_TESTCASES)
        message(FATAL_ERROR "Unknown PTO costmodel testcase: ${TESTCASE}")
    endif()
    add_subdirectory(${TESTCASE})
endforeach()
