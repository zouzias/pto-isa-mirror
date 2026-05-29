# --------------------------------------------------------------------------------
# mla_basic - auto-mode A3 prototype, Multi-Head Latent Attention.
#
# Builds TWO kernel SHARED libraries with different `--cce-aicore-arch` flags:
#   mla_basic_cube_kernel  : `dav-c220-cube` for the 5 cube __global__ entries
#                            (Q proj, KV compress, KV reconstruct, attn QK,
#                            attn PV).
#   mla_basic_vec_kernel   : `dav-c220-vec`  for the 2 vec __global__ entries
#                            (cache store, attn softmax).
#
# Both libraries get `--cce-enable-pto-passes` to enable PTO auto mode.
# The host executable `mla_basic` links against both. ACL stream-order
# provides cross-kernel sync (§A18 pattern from
# docs_for_ai/known_good_kernel_examples.md).
#
# Shape of this file is a generalisation of
# kernels/automode/a2a3/moe_segmented_ffn_top1/CMakeLists.txt; the only delta
# is that we expose two example-creation functions (`pto_kernel_cube_auto`
# and `pto_kernel_vec_auto`) and build the executable separately.
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

# Compiler: honour `-DCMAKE_COMPILER=/path/to/compiler` from the CLI / run.sh.
# Fall back to plain `bisheng` (resolved from PATH) if nothing was passed.
if(NOT DEFINED CMAKE_COMPILER OR CMAKE_COMPILER STREQUAL "")
    set(CMAKE_COMPILER bisheng)
endif()
message(STATUS "Using compiler: ${CMAKE_COMPILER}")
set(CMAKE_C_COMPILER   ${CMAKE_COMPILER})
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

# ---- Helper: build one kernel SHARED library targeting a given arch -------
# NAME : library / source basename. Expects <NAME>.cpp to exist.
# ARCH : --cce-aicore-arch value (e.g. dav-c220-cube, dav-c220-vec).
function(pto_kernel_arch_auto NAME ARCH)
    add_library(${NAME} SHARED ${NAME}.cpp)
    target_compile_options(${NAME} PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=${ARCH}
        --cce-enable-pto-passes
        -std=c++17
    )
    # ${CMAKE_CURRENT_BINARY_DIR} == build/  — picks up generated_cases.h
    # emitted by scripts/generate_cases.py before cmake configure.
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../include/
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
endfunction()

pto_kernel_arch_auto(mla_basic_cube_kernel dav-c220-cube)
pto_kernel_arch_auto(mla_basic_vec_kernel  dav-c220-vec)

# ---- Host executable -----------------------------------------------------
add_executable(mla_basic main.cpp)
target_compile_options(mla_basic PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(mla_basic PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_directories(mla_basic PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(mla_basic PRIVATE
    mla_basic_cube_kernel
    mla_basic_vec_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
