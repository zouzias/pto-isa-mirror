/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_token_permute_body.hpp"
#include "acl/acl.h"
#include "runtime/rt.h"

using namespace moe_token_permute;

// 真 MIX kernel：不手工注入 .ascend.meta（避免与编译器自动生成的
// launch_kernel_mix_aic/_mix_aiv（KTYPE=MIX_AIC_MAIN, ratio 1:2）冲突），
// 对齐 ascendc 后端（其 KERNEL_TASK_TYPE_DEFAULT 在普通编译下为空，依赖编译器自动 meta）。

extern "C" __global__ AICORE void launch_kernel(__gm__ half *tokensGm, __gm__ int32_t *indicesGm,
                                                  __gm__ half *permOutGm, __gm__ int32_t *sioOutGm,
                                                  __gm__ int32_t *workspaceGm, uint64_t fftsAddr)
{
    RunMoeTokenPermuteBody(tokensGm, indicesGm, permOutGm, sioOutGm, workspaceGm, fftsAddr);
}

void LaunchMoeTokenPermuteChevron(aclFloat16 *tokensGm, int32_t *indicesGm, aclFloat16 *permOutGm, int32_t *sioOutGm,
                                  int32_t *workspaceGm, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<kMoeNumCores, nullptr, stream>>>(reinterpret_cast<half *>(tokensGm), indicesGm,
                                                      reinterpret_cast<half *>(permOutGm), sioOutGm, workspaceGm,
                                                      fftsAddr);
}
