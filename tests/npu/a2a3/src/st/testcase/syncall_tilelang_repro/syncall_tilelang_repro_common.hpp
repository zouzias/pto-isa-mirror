/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYNCALL_TILELANG_REPRO_COMMON_HPP
#define SYNCALL_TILELANG_REPRO_COMMON_HPP

#include <cstdint>

enum class ReproVariant : int32_t
{
    Baseline = 0,
    FixCacheLine = 1,
    FixSyncMix = 2,
    FixAivOnly = 3,
    FixCacheLineSyncMix = 4,
};

constexpr int32_t kReproNumCores = 16;
constexpr int32_t kReproNumExperts = 4;
constexpr int32_t kReproBaselineWsSize = kReproNumCores * kReproNumExperts;
constexpr int32_t kReproCacheLineWsSize = kReproNumCores * 8;

inline int32_t GetReproWorkspaceElems(ReproVariant variant)
{
    switch (variant) {
        case ReproVariant::FixCacheLine:
        case ReproVariant::FixCacheLineSyncMix:
            return kReproCacheLineWsSize;
        default:
            return kReproBaselineWsSize;
    }
}

#endif
