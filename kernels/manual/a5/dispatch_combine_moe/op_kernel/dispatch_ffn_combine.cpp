/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dispatch_ffn_combine.cpp
 * \brief
 */
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine.h"
#endif
#include "../kernel_launch.hpp"

#if defined(__CCE_AICORE__)
using namespace AscendC;
using namespace DispatchFFNCombineImpl;
#endif

extern "C" __global__ __aicore__ void dispatch_ffn_combine(__gm__ uint8_t *x, __gm__ uint8_t *w1, __gm__ uint8_t *w2,
                                                           __gm__ uint8_t *expertId, __gm__ uint8_t *scale1,
                                                           __gm__ uint8_t *scale2, __gm__ uint8_t *probs,
                                                           __gm__ uint8_t *xActiveMask, __gm__ uint8_t *c,
                                                           __gm__ uint8_t *expertTokenNums, __gm__ uint8_t *workspaceGM,
                                                           __gm__ uint8_t *tilingGM)
{
#if defined(__CCE_AICORE__)
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        const __gm__ DispatchFFNCombineTilingData *tilingData =
            reinterpret_cast<__gm__ DispatchFFNCombineTilingData *>(tilingGM);
        DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.Init(x, w1, w2, expertId, scale1, scale2, probs, xActiveMask, c, expertTokenNums, workspaceGM, tilingData);
        op.Process();
    }
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

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream)
{
    dispatch_ffn_combine<<<args.block_dim, nullptr, stream>>>(
        static_cast<uint8_t *>(args.x), static_cast<uint8_t *>(args.weight1), static_cast<uint8_t *>(args.weight2),
        static_cast<uint8_t *>(args.expert_idx), static_cast<uint8_t *>(args.scale1),
        static_cast<uint8_t *>(args.scale2), static_cast<uint8_t *>(args.probs),
        static_cast<uint8_t *>(args.x_active_mask), static_cast<uint8_t *>(args.out),
        static_cast<uint8_t *>(args.expert_token_nums), static_cast<uint8_t *>(args.workspace),
        static_cast<uint8_t *>(args.tiling));
}
