/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASYNC_POST_STABILITY_KERNEL_H_
#define ASYNC_POST_STABILITY_KERNEL_H_

#include <cstdint>

enum class AsyncTransferKind : uint32_t {
    TGet = 0,
    TPut = 1,
};

enum class AsyncCheckMode : uint32_t {
    Immediate = 0,
    Deferred = 1,
    LastWaitOnly = 2,
};

bool IsAsyncPostStabilityDeviceRangeAvailable(int nRanks, int firstDeviceId);

bool RunAsyncPostStability(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, AsyncTransferKind transferKind,
    AsyncCheckMode checkMode, uint32_t postCount, uint32_t rounds, uint32_t queueNum);

#endif
