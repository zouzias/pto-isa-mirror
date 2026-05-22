cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng CACHE STRING "Compiler used for auto-mode kernel builds")
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(sparse_flash_attention_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE npu)
endif()

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910B1)
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(
    --cce-enable-pto-passes
    -D_FORTIFY_SOURCE=2
    -g
    -O2
    -std=c++17
    -Wno-macro-redefined
    -Wno-ignored-attributes
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
    "-Rpass=hiipu-pto-mem-alloc"
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=false"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    -g
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    /home/m50048152/pto-isa/include  # pto-isa
    /home/m50048152/pto-isa/include/pto
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

add_library(sparse_flash_attention SHARED sparse_flash_attention.cpp)
target_compile_options(sparse_flash_attention PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -std=c++17
)
target_link_options(sparse_flash_attention PRIVATE --cce-fatobj-link)

add_executable(main main.cpp)
target_compile_options(main PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})

target_link_directories(main PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib
)

target_link_libraries(main PRIVATE
    sparse_flash_attention
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
