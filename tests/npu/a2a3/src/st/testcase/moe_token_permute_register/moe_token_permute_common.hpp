/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_TOKEN_PERMUTE_REGISTER_COMMON_HPP
#define MOE_TOKEN_PERMUTE_REGISTER_COMMON_HPP

#include <cstdint>

constexpr int32_t kMoeNumTokens = 16;
constexpr int32_t kMoeTopK = 4;
constexpr int32_t kMoeE = kMoeNumTokens * kMoeTopK;
constexpr int32_t kMoeNumExperts = 4;
constexpr int32_t kMoeNumCores = 16;
constexpr int32_t kMoeChunkSize = 4;
constexpr int32_t kMoeHiddenSize = 8;
constexpr int32_t kMoeOutLen = 64;
// 每核 histogram 独占一条 32B cache line（8 个 int32），避免相邻核共享 cache line 导致跨核写丢失。
constexpr int32_t kMoeWsSlotStride = 8;
constexpr int32_t kMoeWsTotal = kMoeNumCores * kMoeWsSlotStride;
constexpr int32_t kMoeTokensPerCore = 1;
constexpr int32_t kMoeHalfH = kMoeHiddenSize / 2;
constexpr int32_t kMoeRegisterTilingKey = 2101;

constexpr int32_t kBurstBytes = 32;
constexpr int32_t kMoeBurstIdx = (kMoeChunkSize * 4 + kBurstBytes - 1) / kBurstBytes; // 1
constexpr int32_t kMoeBurstWs = (kMoeWsTotal * 4 + kBurstBytes - 1) / kBurstBytes;    // 16
constexpr int32_t kMoeBurstHalfRow = (kMoeHalfH * 2 + kBurstBytes - 1) / kBurstBytes; // 1

#endif
