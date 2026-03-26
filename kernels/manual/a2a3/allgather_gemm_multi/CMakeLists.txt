cmake_minimum_required(VERSION 3.16)
project(allgather_gemm_demo)

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

if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please export SHMEM_HOME_PATH to shmem install root.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
    message(STATUS "SHMEM_HOME_PATH: ${SHMEM_HOME_PATH}")
endif()

# Auto-detect SHMEM backend
if(DEFINED ENV{SHMEM_BACKEND})
    set(SHMEM_BACKEND $ENV{SHMEM_BACKEND})
endif()
if(NOT DEFINED SHMEM_BACKEND)
    if(EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem.h")
        set(SHMEM_BACKEND "CANN_SHMEM")
    elseif(EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem_api.h")
        set(SHMEM_BACKEND "ASCEND_SHMEM")
    else()
        message(FATAL_ERROR "Cannot find shmem headers in ${SHMEM_HOME_PATH}/shmem/include/")
    endif()
endif()
message(STATUS "SHMEM_BACKEND: ${SHMEM_BACKEND}")

# =============================================================================
# GEMM Size Configuration
# =============================================================================
if(NOT DEFINED G_M)
    set(G_M 2048)
endif()
if(NOT DEFINED G_K)
    set(G_K 2048)
endif()
if(NOT DEFINED G_N)
    set(G_N 1024)
endif()
if(NOT DEFINED SIZE_NAME)
    set(SIZE_NAME "large")
endif()
if(NOT DEFINED COMPUTE_BLOCK_NUM)
    set(COMPUTE_BLOCK_NUM 24)  # Max AIC blocks on Ascend 910B
endif()
if(NOT DEFINED COMM_BLOCK_NUM)
    set(COMM_BLOCK_NUM 48)     # Max AIV blocks on Ascend 910B (2 AIV per AICORE)
endif()
message(STATUS "GEMM Size: M=${G_M}, K=${G_K}, N=${G_N}, Size Name: ${SIZE_NAME}")
message(STATUS "Block Configuration: COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}, COMM_BLOCK_NUM=${COMM_BLOCK_NUM}")

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
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

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
    ${PROJECT_SOURCE_DIR}/../../../include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# SHMEM include paths and definitions
if(SHMEM_BACKEND STREQUAL "ASCEND_SHMEM")
    add_compile_definitions(ASCEND_SHMEM)
    include_directories(
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    )
else()
    add_compile_definitions(CANN_SHMEM)
    include_directories(
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/shmem/include/device
        ${SHMEM_HOME_PATH}/../src/device
    )
endif()

# =============================================================================
# 1. Compute Kernel (Cube arch) - GEMM consumer
# =============================================================================
add_library(allgather_gemm_compute_kernel SHARED allgather_gemm_compute_kernel.cpp)
target_compile_options(allgather_gemm_compute_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-cube
    -DMEMORY_BASE -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
)
target_include_directories(allgather_gemm_compute_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include/
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase/
)
target_link_options(allgather_gemm_compute_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Communication Kernel (Vec arch + shmem) - AllGather producer
# =============================================================================
add_library(allgather_gemm_comm_kernel SHARED allgather_gemm_comm_kernel.cpp)
target_compile_options(allgather_gemm_comm_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_SIZE_NAME="${SIZE_NAME}"
    -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
    -DCONFIG_COMM_BLOCK_NUM=${COMM_BLOCK_NUM}
)
target_include_directories(allgather_gemm_comm_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include/
    ${PROJECT_SOURCE_DIR}/../../../tests/npu/a2a3/comm/st/testcase/
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
)
target_link_directories(allgather_gemm_comm_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/lib
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/lib64
)
# 显式依赖 compute_kernel，以便解析 launchAllGatherGemmComputeStreaming 等符号
target_link_libraries(allgather_gemm_comm_kernel PRIVATE
    allgather_gemm_compute_kernel
    shmem
    runtime
)
target_link_options(allgather_gemm_comm_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 3. Executable
# =============================================================================
add_executable(allgather_gemm main.cpp)
target_compile_options(allgather_gemm PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(allgather_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include/
    ${PROJECT_SOURCE_DIR}/../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../tests/comm/st/testcase/
)
target_compile_definitions(allgather_gemm PRIVATE
    CONFIG_SIZE_NAME="${SIZE_NAME}"
)

target_link_directories(allgather_gemm PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${SHMEM_HOME_PATH}/shmem/lib
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(allgather_gemm PRIVATE
    allgather_gemm_compute_kernel
    allgather_gemm_comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase shmem pthread
)
