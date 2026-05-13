# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_comm_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)

    # Align comm ST kernels with the A5 vec ST configuration by default.
    # Allow external override when a different arch or memory model is needed.
    if(NOT DEFINED CCE_AICORE_ARCH)
        set(CCE_AICORE_ARCH "dav-c310-vec")
    endif()
    if(NOT DEFINED CCE_MEMORY_MODEL)
        set(CCE_MEMORY_MODEL "REGISTER_BASE")
    endif()

    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${CCE_AICORE_ARCH} -D${CCE_MEMORY_MODEL} -std=c++17
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init)

    # Host-arch aware include root: x86_64 first, fallback to aarch64.
    set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
    if(NOT EXISTS "${PTO_TOOLKIT_DIR}")
        set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
    endif()

    # Newer CANN may place AscendC kernel headers under tikcpp.
    set(PTO_ASCENDC_KERNEL_INCLUDE "")
    if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
        set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
    elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
        set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
    endif()

    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        ${PTO_TOOLKIT_DIR}/asc
        ${PTO_TOOLKIT_DIR}/asc/include
        ${PTO_TOOLKIT_DIR}/ascendc/include
        ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api
        ${PTO_TOOLKIT_DIR}/include/ascendc
        ${PTO_TOOLKIT_DIR}/include/ascendc/basic_api
        ${PTO_TOOLKIT_DIR}/asc/impl/basic_api
        ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api/impl
        ${PTO_TOOLKIT_DIR}/asc/include/interface
        ${PTO_TOOLKIT_DIR}/asc/include/utils
        ${PTO_ASCENDC_KERNEL_INCLUDE}
    )
    target_link_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/lib64
    )
    target_link_libraries(${NAME}_kernel PRIVATE
        hcomm runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS}
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase GTest::gtest pthread
    )
endfunction()


# CCU collective tests: same kernel build as pto_comm_st, but host additionally
# links hccl and probes for hcomm CCU headers.  Source is main.cc (not main.cpp).
function(pto_comm_ccu_st NAME)
    # --- kernel (identical to pto_comm_st) ---
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)

    if(NOT DEFINED CCE_AICORE_ARCH)
        set(CCE_AICORE_ARCH "dav-c310-vec")
    endif()
    if(NOT DEFINED CCE_MEMORY_MODEL)
        set(CCE_MEMORY_MODEL "REGISTER_BASE")
    endif()

    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${CCE_AICORE_ARCH} -D${CCE_MEMORY_MODEL} -std=c++17
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init)

    set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
    if(NOT EXISTS "${PTO_TOOLKIT_DIR}")
        set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
    endif()

    set(PTO_ASCENDC_KERNEL_INCLUDE "")
    if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
        set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
    elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
        set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
    endif()

    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        ${PTO_TOOLKIT_DIR}/asc
        ${PTO_TOOLKIT_DIR}/asc/include
        ${PTO_TOOLKIT_DIR}/ascendc/include
        ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api
        ${PTO_TOOLKIT_DIR}/include/ascendc
        ${PTO_TOOLKIT_DIR}/include/ascendc/basic_api
        ${PTO_TOOLKIT_DIR}/asc/impl/basic_api
        ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api/impl
        ${PTO_TOOLKIT_DIR}/asc/include/interface
        ${PTO_TOOLKIT_DIR}/asc/include/utils
        ${PTO_ASCENDC_KERNEL_INCLUDE}
    )
    target_link_directories(${NAME}_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(${NAME}_kernel PRIVATE hcomm runtime)
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    # --- host (main.cc, extra: hccl + hcomm CCU headers) ---
    # Probe hcomm CCU include directory
    find_path(_HCOMM_CCU_PROBE
        NAMES hcomm/ccu/ccu_kernel.h
        PATHS
            ${ASCEND_HOME_PATH}/pkg_inc
            ${ASCEND_HOME_PATH}/include
            ${PROJECT_SOURCE_DIR}/../../../../../../include
        NO_DEFAULT_PATH
    )

    add_executable(${NAME} main.cc)
    # libhcomm.so / libhccl.so ship pre-C++11 std ABI (no `_ZNSt7__cxx11`
    # symbols).  HcclCcuKernelRegister takes a CcuKernelArg* with an embedded
    # std::vector<ChannelHandle>; if the caller uses the new ABI the layout
    # mismatch corrupts ref counters and segfaults in
    # __gnu_cxx::__exchange_and_add.  Force old ABI here.  Requires linking an
    # old-ABI libgtest as well — point GTEST_ROOT to a user-built old-ABI
    # GoogleTest (do NOT use Anaconda's gtest which is new-ABI).
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS}
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init
        -D_GLIBCXX_USE_CXX11_ABI=0)
    target_compile_options(${NAME}_kernel PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../../common
        ${_HCOMM_CCU_PROBE}
    )
    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )
    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl hccl hcomm m c_sec dl GTest::gtest pthread
    )
endfunction()


# Test case list - add new test directories here
# Aligned with SPEC: docs/isa/comm/README.md
set(ALL_TESTCASES
# Point-to-Point Communication (Synchronous)
tput
tget
# Point-to-Point Communication (Asynchronous)
tget_async
tput_async
tput_async_urma
tget_async_urma
# Signal-Based Synchronization
tnotify
twait
ttest
# Collective Communication
tgather
tscatter
treduce
tbroadcast
# Collective Communication (CCU backend)
treduce_ccu
tscatter_ccu
tbroadcast_ccu
tgather_ccu
)

if(TEST_CASE IN_LIST ALL_TESTCASES)
message(STATUS "run : ${TEST_CASE}")
else()
message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES}")
endif()


foreach(TESTCASE ${ALL_TESTCASES})
if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE))
    add_subdirectory(${TESTCASE})
endif()
endforeach()
