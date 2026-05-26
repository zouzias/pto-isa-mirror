# --------------------------------------------------------------------------------
# full_moe_combined - auto-mode A3 prototype (end-to-end MoE; one host wrapper
# launcher fires six kernels on the same stream with only one final sync).
#
# Builds SIX kernel shared libraries from the local copies under ./kernels/
# and one host executable linking against all six.
#
#   kernels/router_matmul_kernel.cpp     -> dav-c220-cube
#   kernels/moe_topk_padded_kernel.cpp   -> dav-c220-vec
#   kernels/outval_pad_kernel.cpp        -> dav-c220-vec   (combined-only bridge)
#   kernels/scatter_kernel.cpp           -> dav-c220-vec
#   kernels/expert_ffn_kernel.cpp        -> dav-c220-cube
#   kernels/gather_kernel.cpp            -> dav-c220-vec
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

set(CMAKE_COMPILER bisheng CACHE STRING "Compiler used for auto-mode kernel builds")
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

function(pto_kernel_lib LIBNAME SRC ARCH_FLAG)
    add_library(${LIBNAME} SHARED ${SRC})
    target_compile_options(${LIBNAME} PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        ${ARCH_FLAG}
        --cce-enable-pto-passes
        -std=c++17
    )
    target_include_directories(${LIBNAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include/
        ${PROJECT_SOURCE_DIR}/../build
    )
    target_link_options(${LIBNAME} PRIVATE --cce-fatobj-link)
endfunction()

pto_kernel_lib(router_matmul_kernel
    ${PROJECT_SOURCE_DIR}/kernels/router_matmul_kernel.cpp
    --cce-aicore-arch=dav-c220-cube)

pto_kernel_lib(moe_topk_padded_kernel
    ${PROJECT_SOURCE_DIR}/kernels/moe_topk_padded_kernel.cpp
    --cce-aicore-arch=dav-c220-vec)

pto_kernel_lib(outval_pad_kernel
    ${PROJECT_SOURCE_DIR}/kernels/outval_pad_kernel.cpp
    --cce-aicore-arch=dav-c220-vec)

pto_kernel_lib(scatter_kernel
    ${PROJECT_SOURCE_DIR}/kernels/scatter_kernel.cpp
    --cce-aicore-arch=dav-c220-vec)

pto_kernel_lib(expert_ffn_kernel
    ${PROJECT_SOURCE_DIR}/kernels/expert_ffn_kernel.cpp
    --cce-aicore-arch=dav-c220-cube)

pto_kernel_lib(gather_kernel
    ${PROJECT_SOURCE_DIR}/kernels/gather_kernel.cpp
    --cce-aicore-arch=dav-c220-vec)

add_executable(full_moe_combined main.cpp)
target_compile_options(full_moe_combined PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(full_moe_combined PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../../include/
        ${PROJECT_SOURCE_DIR}/../build
    ${PROJECT_SOURCE_DIR}/../../../../../tests/common
)
target_link_directories(full_moe_combined PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)
target_link_libraries(full_moe_combined PRIVATE
    router_matmul_kernel
    moe_topk_padded_kernel
    outval_pad_kernel
    scatter_kernel
    expert_ffn_kernel
    gather_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
