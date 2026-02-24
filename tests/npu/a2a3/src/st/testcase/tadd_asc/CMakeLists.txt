# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tadd_asc)

# AscendC kernel headers
# kernel_operator.h -> kernel_tpipe_impl.h -> kernel_tpipe.h
# kernel_operator_layout.h -> include/utils/std/tuple.h

target_include_directories(tadd_asc_kernel PRIVATE
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/asc
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/asc/include
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/asc/include/basic_api
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/asc/include/interface
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/asc/impl/basic_api
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/ascendc/include
  /usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/ascendc/include/basic_api/impl
)
