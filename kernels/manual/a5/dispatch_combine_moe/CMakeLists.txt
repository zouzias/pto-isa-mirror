cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(dispatch_combine_moe_a5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for IDE navigation" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
get_filename_component(PTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(PTO_ROOT "${PTO_ROOT}" CACHE PATH "Path to PTO ISA repository" FORCE)

if(NOT EXISTS ${PTO_ROOT}/include/pto/pto-inst.hpp)
    message(FATAL_ERROR "PTO headers not found under PTO_ROOT=${PTO_ROOT}")
endif()

if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()

if(EXISTS ${ASCEND_HOME_PATH}/x86_64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/aarch64-linux)
else()
    message(FATAL_ERROR "Cannot find Ascend kernel include root under ${ASCEND_HOME_PATH}/x86_64-linux or aarch64-linux")
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A5)

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS ${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print)
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

set(KERNEL_INCLUDE_DIRS
    ${PTO_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/token_reorder/routing
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
)

set(HOST_INCLUDE_DIRS
    ${PTO_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/token_reorder/routing
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api)
endif()

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND KERNEL_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api)
endif()

set(HOST_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
)

add_library(dispatch_combine_moe_kernel SHARED op_kernel/dispatch_ffn_combine.cpp)
target_compile_options(dispatch_combine_moe_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310
    -DMEMORY_BASE -std=c++17
    -DDTYPE_W1=int8_t
    -DDTYPE_OUT=half
)
target_include_directories(dispatch_combine_moe_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(dispatch_combine_moe_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(dispatch_combine_moe_kernel PRIVATE runtime hcomm nnopbase)
target_link_options(dispatch_combine_moe_kernel PRIVATE --cce-fatobj-link)

add_executable(dispatch_combine_moe
    main.cpp
    op_host/runtime_context.cpp
    op_host/tiling_builder.cpp
    op_host/data_utils.cpp
)
target_compile_options(dispatch_combine_moe PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(dispatch_combine_moe PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(dispatch_combine_moe PUBLIC ${HOST_LINK_DIRS})
target_link_libraries(dispatch_combine_moe PRIVATE
    dispatch_combine_moe_kernel
    runtime
    stdc++ ascendcl hcomm m tiling_api unified_dlog register platform c_sec dl nnopbase pthread
)
