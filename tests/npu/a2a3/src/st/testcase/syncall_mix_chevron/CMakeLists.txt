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

add_library(syncall_mix_chevron_aiv_kernel SHARED syncall_aiv_kernel.cpp)
target_compile_options(syncall_mix_chevron_aiv_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17)
target_include_directories(syncall_mix_chevron_aiv_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_mix_chevron_aiv_kernel PRIVATE --cce-fatobj-link)

add_library(syncall_mix_chevron_aic_kernel SHARED syncall_aic_kernel.cpp)
target_compile_options(syncall_mix_chevron_aic_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube -DMEMORY_BASE -std=c++17)
target_include_directories(syncall_mix_chevron_aic_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_mix_chevron_aic_kernel PRIVATE --cce-fatobj-link)

target_link_libraries(syncall_mix_chevron PRIVATE syncall_mix_chevron_aiv_kernel syncall_mix_chevron_aic_kernel)
