/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file dispatch_combine_moe.cpp
 * \brief
 */
#include <pto/pto-inst.hpp>
#if defined(__CCE_AICORE__)
#include "lib/matmul_intf.h"
#include "dispatch_combine_moe_tiling.h"
#include "dispatch_combine_moe.h"
#endif
#include "../kernel_launch.hpp"

#if defined(__CCE_AICORE__)
using DispatchCombineMoeImpl::DispatchCombineMoe;
#endif

PTO_SYNCALL_MIX_AIC_KERNEL_META(dispatch_combine_moe, 1, 2);

extern "C" __global__ AICORE void dispatch_combine_moe(__gm__ uint8_t *x, __gm__ uint8_t *w1, __gm__ uint8_t *w2,
                                                       __gm__ uint8_t *expertId, __gm__ uint8_t *scale1,
                                                       __gm__ uint8_t *scale2, __gm__ uint8_t *probs,
                                                       __gm__ uint8_t *xActiveMask, __gm__ uint8_t *c,
                                                       __gm__ uint8_t *expertTokenNums, __gm__ uint8_t *workspaceGM,
                                                       __gm__ uint8_t *tilingGM)
{
#if defined(__CCE_AICORE__)
    const __gm__ DispatchCombineMoeTilingData *tilingData =
        reinterpret_cast<__gm__ DispatchCombineMoeTilingData *>(tilingGM);
    DispatchCombineMoe<int8_t, DTYPE_W1, DTYPE_OUT, false, true> op;
    op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM, tilingData);
    op.Process();
#else
    (void)x;
    (void)w1;
    (void)w2;
    (void)expertId;
    (void)scale1;
    (void)scale2;
    (void)probs;
    (void)xActiveMask;
    (void)c;
    (void)expertTokenNums;
    (void)workspaceGM;
    (void)tilingGM;
#endif
}

void launchDispatchCombineMoe(const DispatchCombineMoeLaunchArgs &args, void *stream)
{
    dispatch_combine_moe<<<args.block_dim, nullptr, stream>>>(
        static_cast<uint8_t *>(args.x), static_cast<uint8_t *>(args.weight1), static_cast<uint8_t *>(args.weight2),
        static_cast<uint8_t *>(args.expert_idx), static_cast<uint8_t *>(args.scale1),
        static_cast<uint8_t *>(args.scale2), static_cast<uint8_t *>(args.probs),
        static_cast<uint8_t *>(args.x_active_mask), static_cast<uint8_t *>(args.out),
        static_cast<uint8_t *>(args.expert_token_nums), static_cast<uint8_t *>(args.workspace),
        static_cast<uint8_t *>(args.tiling));
}
