# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_cube_st(tmatmul_mx)
# The same Runtime entry points are available in runtime and runtime_camodel.
target_include_directories(tmatmul_mx PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
)
# ACL pulls in the hardware Runtime even in sim mode; this host only uses Runtime APIs.
get_target_property(tmatmul_mx_libraries tmatmul_mx LINK_LIBRARIES)
list(REMOVE_ITEM tmatmul_mx_libraries ascendcl nnopbase tiling_api platform)
set_target_properties(tmatmul_mx PROPERTIES LINK_LIBRARIES "${tmatmul_mx_libraries}")
# Drop unused ACL helpers defined in test_common.h.
target_compile_options(tmatmul_mx PRIVATE -ffunction-sections -fdata-sections)
target_link_options(tmatmul_mx PRIVATE -Wl,--gc-sections)
