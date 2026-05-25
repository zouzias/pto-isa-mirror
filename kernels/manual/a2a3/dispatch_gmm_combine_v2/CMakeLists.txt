cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(dispatch_combine_moe_v2 C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_C_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
get_filename_component(PTO_ROOT_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(PTO_ROOT "${PTO_ROOT_DEFAULT}" CACHE PATH "Path to PTO ISA repository")
set(SOC_VERSION "ascend910_93" CACHE STRING "Ascend SoC version")
set(RUN_MODE "npu" CACHE STRING "Run mode")

if(SOC_VERSION STREQUAL "ascend910_93")
    set(ASCENDC_SOC_VERSION "ascend910_9391")
elseif(SOC_VERSION STREQUAL "ascend910_9391" OR
       SOC_VERSION STREQUAL "ascend910_9381" OR
       SOC_VERSION STREQUAL "ascend910_9372" OR
       SOC_VERSION STREQUAL "ascend910_9392" OR
       SOC_VERSION STREQUAL "ascend910_9382" OR
       SOC_VERSION STREQUAL "ascend910_9362")
    set(ASCENDC_SOC_VERSION "${SOC_VERSION}")
else()
    message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}")
endif()
set(SOC_VERSION ${ASCENDC_SOC_VERSION} CACHE STRING "AscendC exact SoC version" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check CANN installation.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(V4_COMMON_CPP_OPTIONS
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h")

set(V4_DEVICE_COMPILE_OPTIONS
    -xcce
    -Xhost-start
    -Xhost-end
    -DPTO_NPU_ARCH_A2A3
    -DMEMORY_BASE
    -std=c++17
    -Wno-ignored-attributes
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h")

set(V4_DEVICE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/protocol
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/dispatch
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/routing
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/compute
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/compute_vec
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/combine
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/output
    ${PTO_ROOT}/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl)

function(add_v4_device_kernel TARGET_NAME TARGET_ARCH)
    add_library(${TARGET_NAME} SHARED ${ARGN})
    target_compile_options(${TARGET_NAME} PRIVATE
        ${V4_DEVICE_COMPILE_OPTIONS}
        --cce-aicore-arch=${TARGET_ARCH})
    target_include_directories(${TARGET_NAME} PRIVATE ${V4_DEVICE_INCLUDE_DIRS})
    target_link_directories(${TARGET_NAME} PRIVATE ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(${TARGET_NAME} PRIVATE hcomm runtime)
    target_link_options(${TARGET_NAME} PRIVATE --cce-fatobj-link)
endfunction()

add_v4_device_kernel(dispatch_combine_moe_v2_dispatch_combine_kernel
    dav-c220-vec
    op_kernel/dispatch_ffn_combine.cpp)
add_v4_device_kernel(dispatch_combine_moe_v2_comm_vec_queue_kernel
    dav-c220-vec
    op_kernel/compute_vec/comm_vec_queue_kernel.cpp)
add_v4_device_kernel(dispatch_combine_moe_v2_compute_cube_queue_kernel
    dav-c220-cube
    op_kernel/compute/compute_cube_queue_kernel.cpp)

add_executable(dispatch_combine_moe_v2
    main.cpp
    runtime_context.cpp
    tiling_builder.cpp)

target_compile_options(dispatch_combine_moe_v2 PRIVATE ${V4_COMMON_CPP_OPTIONS})
target_include_directories(dispatch_combine_moe_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/protocol
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl
    ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    ${PTO_ROOT}/include)

target_link_directories(dispatch_combine_moe_v2 PRIVATE
    ${ASCEND_HOME_PATH}/lib64)

target_link_libraries(dispatch_combine_moe_v2 PRIVATE
    dispatch_combine_moe_v2_dispatch_combine_kernel
    dispatch_combine_moe_v2_comm_vec_queue_kernel
    dispatch_combine_moe_v2_compute_cube_queue_kernel
    ascendcl hcomm runtime nnopbase tiling_api platform c_sec dl pthread stdc++ m)
set_target_properties(dispatch_combine_moe_v2 PROPERTIES
    BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    INSTALL_RPATH "$ORIGIN/lib")

