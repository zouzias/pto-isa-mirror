/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/npu/comm/async/ccu/ccu_trigger_intrin.hpp"

__global__ __aicore__ void tscatter_ccu_trigger_kernel(
    uint64_t ckeVA, uint32_t mask)
{
    if (get_block_idx() != 0) return;
    pto::comm::ccu::__ccu_trigger_gate(
        reinterpret_cast<__gm__ uint64_t *>(ckeVA), mask);
}

extern "C" __attribute__((visibility("default"))) int
tscatter_ccu_trigger_launch(void *stream, uint64_t ckeVA, uint32_t mask)
{
    tscatter_ccu_trigger_kernel<<<1, nullptr, stream>>>(ckeVA, mask);
    return 0;
}
