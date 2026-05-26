/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_KERNEL_LAUNCHERS_H_
#define MOE_DISPATCH_KERNEL_LAUNCHERS_H_

#include "common.h"

#include <cstdint>

namespace moe_dispatch {

// Fixed host launcher ABI. Task 1 adds the device launch expressions without
// changing this order: shape, rank, data pointers, peerWindow, hcclCtx,
// workspace, stream, launch block count.
void LaunchMoeDispatchKernel(MoeDispatchShape shape, uint32_t myRank, uint8_t *inputA, uint8_t *expertIdx,
                             uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace, void *stream,
                             uint32_t launchBlockCount);

} // namespace moe_dispatch

#endif // MOE_DISPATCH_KERNEL_LAUNCHERS_H_
