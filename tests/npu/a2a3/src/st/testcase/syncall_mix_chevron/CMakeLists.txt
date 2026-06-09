# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_mix_st(syncall_mix_chevron)

# NOTE: hard and soft SYNCALL kernels must NOT share one .so. A soft kernel in
# the same fatobj corrupts the hard kernel's FFTS hardware-barrier config and
# causes hard AIV-only / AIC-only cases to hang. Keep each kernel in its own .so.
function(pto_syncall_chevron_kernel TARGET SOURCE ARCH)
    add_library(${TARGET} SHARED ${SOURCE})
    target_compile_options(${TARGET} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${ARCH} -DMEMORY_BASE -std=c++17)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link)
    target_link_libraries(syncall_mix_chevron PRIVATE ${TARGET})
endfunction()

pto_syncall_chevron_kernel(syncall_mix_chevron_aiv_hard_kernel syncall_aiv_hard_kernel.cpp dav-c220-vec)
pto_syncall_chevron_kernel(syncall_mix_chevron_aiv_soft_kernel syncall_aiv_soft_kernel.cpp dav-c220-vec)
# Hard AIC-only is built as a MIX kernel (dav-c220 auto-split, AIC does the work,
# AIV is an empty stub). AIC-only hard sync needs the MIX FFTS context to barrier
# correctly; a pure dav-c220-cube chevron launch hangs.
pto_syncall_chevron_kernel(syncall_mix_chevron_aic_hard_kernel syncall_aic_hard_kernel.cpp dav-c220)
pto_syncall_chevron_kernel(syncall_mix_chevron_aic_soft_kernel syncall_aic_soft_kernel.cpp dav-c220-cube)

# Soft MIX 1:1 needs 20 AIC + 20 AIV launched on two streams (dav-c220 auto-split
# is fixed at 1:2, so 1:1 cannot use auto-split). Compile one source for both
# arches (cube + vec objects) into a single .so; the host launcher issues two
# chevron launches on two streams. No register ELF is needed (soft sync polls GM,
# it does not rely on a unified MIX FFTS context).
function(pto_syncall_chevron_mix11_kernel TARGET SOURCE)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(MIX11_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    )
    set(MIX11_INCLUDES
        -I${PROJECT_SOURCE_DIR}/../../../../../include
        -I${ASCEND_HOME_PATH}/include
        -I${ASCEND_DRIVER_PATH}/kernel/inc
        -I${ASCEND_HOME_PATH}/pkg_inc/
        -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        -I${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIV_RATIO__=1 -DSYNCALL_MIX_BUILD_AIC
                ${MIX11_INCLUDES}
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIV_RATIO__=1 -DSYNCALL_MIX_BUILD_AIV
                ${MIX11_INCLUDES}
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_library(${TARGET} SHARED ${AIC_OBJ} ${AIV_OBJ})
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${AIC_OBJ} ${AIV_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link -Wl,--strip-debug)
    target_link_libraries(syncall_mix_chevron PRIVATE ${TARGET})
endfunction()

pto_syncall_chevron_mix11_kernel(syncall_mix_chevron_mix11_soft_kernel syncall_mix11_soft_kernel.cpp)

# Hard MIX 1:1 cannot use dav-c220 auto-split: ccec/bisheng GetTaskRation() is
# always 2, so auto-split is physically fixed at 1:2 and a meta(1,1) override is
# ignored (the hard 1:1 auto-split experiment produced all-zero output). CANN's
# own 1:1 mix operators only get a real 1:1 on the opc compile path. We therefore
# reproduce the original register-ELF path: compile the kernel for cube and vec,
# build a compact registration ELF from the nested device ELFs
# (make_mix_register_elf.py), and launch with rtKernelLaunchWithHandleV2 so the
# runtime schedules 20 AIC + 20 AIV under one MIX FFTS context.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(pto_syncall_chevron_mix11_hard_kernel TARGET SOURCE)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(REGISTER_AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir/${SOURCE}.o)
    set(REGISTER_AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir/${SOURCE}.o)
    set(REGISTER_ELF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_register.elf)
    set(MIX11_HARD_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    )
    set(MIX11_HARD_INCLUDES
        -I${PROJECT_SOURCE_DIR}/../../../../../include
        -I${ASCEND_HOME_PATH}/include
        -I${ASCEND_DRIVER_PATH}/kernel/inc
        -I${ASCEND_HOME_PATH}/pkg_inc/
        -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        -I${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_HARD_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=20 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIC -DSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                ${MIX11_HARD_INCLUDES}
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_HARD_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=20 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIV -DSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                ${MIX11_HARD_INCLUDES}
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${REGISTER_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_HARD_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=20 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIC -DSYNCALL_MIX_REGISTER_BUILD
                ${MIX11_HARD_INCLUDES}
                -xcce -o ${REGISTER_AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_HARD_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=20 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIV -DSYNCALL_MIX_REGISTER_BUILD
                ${MIX11_HARD_INCLUDES}
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
    target_link_libraries(syncall_mix_chevron PRIVATE ${TARGET})
endfunction()

pto_syncall_chevron_mix11_hard_kernel(syncall_mix_chevron_mix11_hard_kernel syncall_mix11_hard_kernel.cpp)
