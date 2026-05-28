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
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

function(pto_kernel_arch_auto NAME ARCH)
    add_library(${NAME} SHARED ${NAME}.cpp)
    target_compile_options(${NAME} PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=${ARCH}
        --cce-enable-pto-passes
        -std=c++17
    )
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../include/
        ${PROJECT_BINARY_DIR}
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
endfunction()

# GEMM: cube arch (matrix multiply)
pto_kernel_arch_auto(gemm                          dav-c220)
# Biased Softmax: vec arch (element-wise bias add + softmax reduction)
pto_kernel_arch_auto(biased_softmax_compression    dav-c220)
# Weighted Sum: vec arch (element-wise multiply + reduction along ratio dim)
pto_kernel_arch_auto(weighted_sum                  dav-c220)

# ---- Host executable ---------------------------------------------------------
add_executable(hca_compressor main.cpp)
target_compile_options(hca_compressor PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(hca_compressor PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${PROJECT_BINARY_DIR}
)

target_link_directories(hca_compressor PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(hca_compressor PRIVATE
    gemm
    biased_softmax_compression
    weighted_sum
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
