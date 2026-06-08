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
