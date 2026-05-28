/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "a3_int8_backend.hpp"
#include "control_metadata.hpp"
#include "moe_dispatch_combine_a8w8_types.hpp"
#include "protocol_core.hpp"

namespace moe_dispatch_combine_a8w8 {
namespace {

__global__ AICORE void MoeDispatchCombineA8W8Kernel(ShapeConfig shape, RankConfig rank, GM_ADDR workspace,
                                                    GM_ADDR peerWindow, GM_ADDR hcclCtx)
{
    (void)shape;
    (void)rank;
    (void)workspace;
    (void)peerWindow;
    (void)hcclCtx;
}

} // namespace

void LaunchMoeDispatchCombineA8W8(ShapeConfig shape, RankConfig rank, uint8_t *workspace, uint8_t *peerWindow,
                                  uint8_t *hcclCtx, void *stream, uint32_t launchBlockCount)
{
    MoeDispatchCombineA8W8Kernel<<<launchBlockCount, nullptr, stream>>>(shape, rank, workspace, peerWindow, hcclCtx);
}

} // namespace moe_dispatch_combine_a8w8
