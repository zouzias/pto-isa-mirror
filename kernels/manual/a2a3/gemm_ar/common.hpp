// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
//
// HCCL communication wrapper header for GEMM AllReduce.
// Uses HCCL windows for inter-rank communication.
// MUST be included BEFORE <iostream> to avoid kernel_operator.h <-> std::dec collision.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef MEMORY_BASE
#include "kernel_operator.h"
using AscendC::GlobalTensor;
#endif

#include "hccl_context.h"

// ============================================================================
// Device-only wrappers
// ============================================================================
#ifdef MEMORY_BASE

template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

#endif // MEMORY_BASE
