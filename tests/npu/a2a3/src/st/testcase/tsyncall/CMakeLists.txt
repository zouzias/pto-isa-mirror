# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tsyncall)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(tsyncall_soft_kernel SHARED tsyncall_soft_kernel.cpp)
target_compile_options(tsyncall_soft_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                                                     -DMEMORY_BASE -std=c++17)
target_include_directories(tsyncall_soft_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(tsyncall_soft_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(tsyncall PRIVATE tsyncall_soft_kernel)

function(pto_tsyncall_mix_kernel TARGET SOURCE RATIO_DEF)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(REGISTER_AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir/${SOURCE}.o)
    set(REGISTER_AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir/${SOURCE}.o)
    set(REGISTER_ELF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_register.elf)
    set(TSYNCALL_MIX_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    )
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${TSYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 ${RATIO_DEF} -DTSYNCALL_MIX_BUILD_AIC
                -DTSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${TSYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 ${RATIO_DEF} -DTSYNCALL_MIX_BUILD_AIV
                -DTSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${REGISTER_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${TSYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 ${RATIO_DEF} -DTSYNCALL_MIX_BUILD_AIC
                -DTSYNCALL_MIX_REGISTER_BUILD
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${REGISTER_AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${CMAKE_CXX_COMPILER} ${TSYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=24 ${RATIO_DEF} -DTSYNCALL_MIX_BUILD_AIV
                -DTSYNCALL_MIX_REGISTER_BUILD
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${REGISTER_AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_mix_register_elf.py
                ${REGISTER_AIC_OBJ} ${REGISTER_AIV_OBJ} ${REGISTER_ELF}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/make_mix_register_elf.py
        VERBATIM
    )
    add_custom_target(${TARGET}_register_obj DEPENDS ${REGISTER_ELF})
    add_library(${TARGET} SHARED ${AIC_OBJ} ${AIV_OBJ})
    add_dependencies(${TARGET} ${TARGET}_register_obj)
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${AIC_OBJ} ${AIV_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link -Wl,--strip-debug)
    target_link_libraries(tsyncall PRIVATE ${TARGET})
endfunction()

pto_tsyncall_mix_kernel(tsyncall_mix_1_1_kernel tsyncall_mix_1_1_kernel.cpp -D__MIX_CORE_AIV_RATION__=1)
pto_tsyncall_mix_kernel(tsyncall_mix_1_2_kernel tsyncall_mix_1_2_kernel.cpp -D__MIX_CORE_AIV_RATION__=2)
