/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#define ENABLE_DEBUG_PRINT 1

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

template <typename T>
__global__ AICORE void WindowMemCopyIn(__gm__ T *winDst, __gm__ T *devSrc, int count)
{
    for (int i = 0; i < count; ++i) {
        winDst[i] = devSrc[i];
    }
    pipe_barrier(PIPE_ALL);
}
