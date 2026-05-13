# --------------------------------------------------------------------------------
# mani_moe - auto-mode A3 prototype.
#
# Local harness copied from the existing auto-mode MoE/topk prototypes, adapted
# for this directory's source names:
#   kernel TU : moe_kernel.cpp
#   executable: mani_moe
#
# Builds cube and vec kernel TUs separately:
#   moe_kernel.cpp      : router GEMM, cube target
#   moe_topk_kernel.cpp : router TopK, vec target
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(pto_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

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
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

add_library(mani_moe_kernel SHARED moe_kernel.cpp)
target_compile_options(mani_moe_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-cube
    --cce-enable-pto-passes
    -std=c++17
)
target_include_directories(mani_moe_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
)
target_link_options(mani_moe_kernel PRIVATE --cce-fatobj-link)

add_library(mani_moe_topk_kernel SHARED moe_topk_kernel.cpp)
target_compile_options(mani_moe_topk_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-vec
    --cce-enable-pto-passes
    -std=c++17
)
target_include_directories(mani_moe_topk_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
)
target_link_options(mani_moe_topk_kernel PRIVATE --cce-fatobj-link)

add_executable(mani_moe main.cpp)
target_compile_options(mani_moe PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(mani_moe PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
)

target_link_directories(mani_moe PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(mani_moe PRIVATE
    mani_moe_kernel
    mani_moe_topk_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
