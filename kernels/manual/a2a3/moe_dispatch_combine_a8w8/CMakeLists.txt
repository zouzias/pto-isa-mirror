# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_moe_dispatch_combine_a8w8)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()

set(FUSED_KERNEL_ARCH "dav-c220-vec" CACHE STRING "A2/A3 dispatch/combine vector kernel architecture.")

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A2A3)

add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=true"
)

if(DEBUG_MODE)
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/host
    ${PROJECT_SOURCE_DIR}/kernel
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
)

set(KERNEL_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernel
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)

set(HOST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/host
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
)

set(HOST_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

add_library(moe_dispatch_combine_a8w8_kernel SHARED kernel/moe_dispatch_combine_a8w8_kernel.cpp)
target_compile_options(moe_dispatch_combine_a8w8_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=${FUSED_KERNEL_ARCH}
    -DMEMORY_BASE -std=c++17
)
target_include_directories(moe_dispatch_combine_a8w8_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(moe_dispatch_combine_a8w8_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(moe_dispatch_combine_a8w8_kernel PRIVATE runtime)
target_link_options(moe_dispatch_combine_a8w8_kernel PRIVATE --cce-fatobj-link)

add_library(moe_dispatch_combine_a8w8_gmm_kernel SHARED kernel/moe_dispatch_combine_a8w8_gmm_kernel.cpp)
target_compile_options(moe_dispatch_combine_a8w8_gmm_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-cube
    -DMEMORY_BASE -std=c++17
)
target_include_directories(moe_dispatch_combine_a8w8_gmm_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(moe_dispatch_combine_a8w8_gmm_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(moe_dispatch_combine_a8w8_gmm_kernel PRIVATE runtime)
target_link_options(moe_dispatch_combine_a8w8_gmm_kernel PRIVATE --cce-fatobj-link)

function(pto_m2_mixed_spike_kernel TARGET SOURCE)
    get_filename_component(SOURCE_NAME ${SOURCE} NAME)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE_NAME}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE_NAME}.o)
    set(REGISTER_AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir/${SOURCE_NAME}.o)
    set(REGISTER_AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir/${SOURCE_NAME}.o)
    set(REGISTER_ELF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_register.elf)
    set(MIX_REGISTER_SCRIPT ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/src/st/testcase/syncall/make_mix_register_elf.py)
    set(M2_MIX_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=true
    )
    set(M2_MIX_INCLUDE_OPTIONS
        -I${PROJECT_SOURCE_DIR}/../../../../include
        -I${PROJECT_SOURCE_DIR}/include
        -I${PROJECT_SOURCE_DIR}/kernel
        -I${ASCEND_HOME_PATH}/include
        -I${ASCEND_DRIVER_PATH}/kernel/inc
        -I${ASCEND_HOME_PATH}/pkg_inc/
        -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${M2_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 -D__MIX_CORE_AIV_RATIO__=1
                -DM2_MIXED_SPIKE_BUILD_AIC -DM2_MIXED_SPIKE_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                ${M2_MIX_INCLUDE_OPTIONS} -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${M2_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 -D__MIX_CORE_AIV_RATIO__=1
                -DM2_MIXED_SPIKE_BUILD_AIV -DM2_MIXED_SPIKE_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                ${M2_MIX_INCLUDE_OPTIONS} -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${REGISTER_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${M2_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 -D__MIX_CORE_AIV_RATIO__=1
                -DM2_MIXED_SPIKE_BUILD_AIC -DM2_MIXED_SPIKE_REGISTER_BUILD
                ${M2_MIX_INCLUDE_OPTIONS} -xcce -o ${REGISTER_AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${CMAKE_CXX_COMPILER} ${M2_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 -D__MIX_CORE_AIV_RATIO__=1
                -DM2_MIXED_SPIKE_BUILD_AIV -DM2_MIXED_SPIKE_REGISTER_BUILD
                ${M2_MIX_INCLUDE_OPTIONS} -xcce -o ${REGISTER_AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${MIX_REGISTER_SCRIPT} ${REGISTER_AIC_OBJ} ${REGISTER_AIV_OBJ} ${REGISTER_ELF}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} ${MIX_REGISTER_SCRIPT}
        VERBATIM
    )
    add_custom_target(${TARGET}_register_obj DEPENDS ${REGISTER_ELF})
    add_library(${TARGET} SHARED ${AIC_OBJ} ${AIV_OBJ})
    add_dependencies(${TARGET} ${TARGET}_register_obj)
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${AIC_OBJ} ${AIV_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_include_directories(${TARGET} PRIVATE ${KERNEL_INCLUDE_DIRS})
    target_link_directories(${TARGET} PRIVATE ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(${TARGET} PRIVATE runtime)
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link -Wl,--strip-debug)
endfunction()

pto_m2_mixed_spike_kernel(moe_dispatch_combine_a8w8_mixed_spike
                          kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp)

add_executable(moe_dispatch_combine_a8w8 host/main.cpp)
target_compile_options(moe_dispatch_combine_a8w8 PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(moe_dispatch_combine_a8w8 PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(moe_dispatch_combine_a8w8 PUBLIC ${HOST_LINK_DIRS})
target_link_libraries(moe_dispatch_combine_a8w8 PRIVATE
    moe_dispatch_combine_a8w8_kernel
    moe_dispatch_combine_a8w8_gmm_kernel
    moe_dispatch_combine_a8w8_mixed_spike
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
