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

# HCCL backend (replaces SHMEM)
message(STATUS "Using HCCL backend for communication")

# =============================================================================
# Architecture Selection (SOC_VERSION driven)
# =============================================================================
if(SOC_VERSION MATCHES "^Ascend950")
    set(CCE_AICORE_ARCH_CUBE "dav-c310-cube")
    set(CCE_AICORE_ARCH_VEC  "dav-c310-vec")
    set(CCE_MEMORY_MODEL     "REGISTER_BASE")
    set(COMPUTE_BLOCK_NUM_DEFAULT 32)
    set(COMM_BLOCK_NUM_DEFAULT 64)
    message(STATUS "A5 (Ascend 950): arch=${CCE_AICORE_ARCH_CUBE}/${CCE_AICORE_ARCH_VEC}, memory=${CCE_MEMORY_MODEL}")
else()
    set(CCE_AICORE_ARCH_CUBE "dav-c220-cube")
    set(CCE_AICORE_ARCH_VEC  "dav-c220-vec")
    set(CCE_MEMORY_MODEL     "MEMORY_BASE")
    set(COMPUTE_BLOCK_NUM_DEFAULT 24)
    set(COMM_BLOCK_NUM_DEFAULT 48)
    message(STATUS "A2/A3: arch=${CCE_AICORE_ARCH_CUBE}/${CCE_AICORE_ARCH_VEC}, memory=${CCE_MEMORY_MODEL}")
endif()

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
    set(COMPUTE_BLOCK_NUM ${COMPUTE_BLOCK_NUM_DEFAULT})
endif()
if(NOT DEFINED COMM_BLOCK_NUM)
    set(COMM_BLOCK_NUM ${COMM_BLOCK_NUM_DEFAULT})
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
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# =============================================================================
# 1. Compute Kernel (Cube arch) - GEMM consumer
# =============================================================================
add_library(allgather_gemm_compute_kernel SHARED allgather_gemm_compute_kernel.cpp)
target_compile_options(allgather_gemm_compute_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=${CCE_AICORE_ARCH_CUBE}
    -D${CCE_MEMORY_MODEL} -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
)
target_include_directories(allgather_gemm_compute_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase/
)
target_link_options(allgather_gemm_compute_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Communication Kernel (Vec arch + shmem) - AllGather producer
# =============================================================================
add_library(allgather_gemm_comm_kernel SHARED allgather_gemm_comm_kernel.cpp)
target_compile_options(allgather_gemm_comm_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=${CCE_AICORE_ARCH_VEC}
    -D${CCE_MEMORY_MODEL} -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_SIZE_NAME="${SIZE_NAME}"
    -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
    -DCONFIG_COMM_BLOCK_NUM=${COMM_BLOCK_NUM}
)
# Host-arch aware include root: x86_64 first, fallback aarch64.
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

target_include_directories(allgather_gemm_comm_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
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
target_link_directories(allgather_gemm_comm_kernel PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(allgather_gemm_comm_kernel PRIVATE
    allgather_gemm_compute_kernel
    hccl hcomm runtime nnopbase
)
target_link_options(allgather_gemm_comm_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 3. Executable
# =============================================================================
add_executable(allgather_gemm main.cpp)
target_compile_options(allgather_gemm PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(allgather_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase/
)
target_compile_definitions(allgather_gemm PRIVATE
    CONFIG_SIZE_NAME="${SIZE_NAME}"
)

target_link_directories(allgather_gemm PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(allgather_gemm PRIVATE
    allgather_gemm_compute_kernel
    allgather_gemm_comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hccl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
