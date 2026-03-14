/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#if (__CHECK_FEATURE_AT_PRECOMPILE) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__) || \
    defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)

#include "moe_grouped_ffn_projection_common.h"

using namespace moe_grouped_ffn_projection;

extern "C" __global__ AICORE void moe_grouped_dual_proj_custom(GM_ADDR x, GM_ADDR gate_weight_dn, GM_ADDR up_weight_dn,
                                                               GM_ADDR gate_proj, GM_ADDR up_proj,
                                                               GM_ADDR group_offsets, GM_ADDR tile_offsets,
                                                               uint32_t num_experts, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
    (void)workspace;
    auto *xPtr = reinterpret_cast<__gm__ bfloat16_t *>(x);
    auto *groupOffsetsPtr = reinterpret_cast<__gm__ int32_t *>(group_offsets);
    auto *tileOffsetsPtr = reinterpret_cast<__gm__ int32_t *>(tile_offsets);

    RunProjectionKernelCompact(xPtr, reinterpret_cast<__gm__ bfloat16_t *>(gate_weight_dn),
                               reinterpret_cast<__gm__ float *>(gate_proj), groupOffsetsPtr, tileOffsetsPtr,
                               num_experts);
    RunProjectionKernelCompact(xPtr, reinterpret_cast<__gm__ bfloat16_t *>(up_weight_dn),
                               reinterpret_cast<__gm__ float *>(up_proj), groupOffsetsPtr, tileOffsetsPtr,
                               num_experts);
#endif

    pipe_barrier(PIPE_ALL);
}

#endif
#endif
