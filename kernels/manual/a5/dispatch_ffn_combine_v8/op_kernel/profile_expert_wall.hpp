/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_PROFILE_EXPERT_WALL_H
#define DISPATCH_FFN_COMBINE_V8_PROFILE_EXPERT_WALL_H

#include "kernel_launch.hpp"
#include "profile_debug_config.h"

namespace dispatch_ffn_combine_v8 {

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
AICORE inline void RecordExpertWallClock(volatile __gm__ uint64_t *profileEntry, size_t base, uint32_t index,
                                         uint32_t maxIndex, uint64_t start, uint64_t end)
{
    if (profileEntry == nullptr || start == 0U || end < start || index >= maxIndex) {
        return;
    }
    const size_t slot = base + static_cast<size_t>(index) * kDispatchFFNCombineProfileExpertWallFieldCount;
    if (slot + 1U >= kDispatchFFNCombineProfileEntryU64Count) {
        return;
    }
    if (profileEntry[slot] == 0U) {
        profileEntry[slot] = start;
    }
    profileEntry[slot + 1U] = end;
}
#else
AICORE inline void RecordExpertWallClock(volatile __gm__ uint64_t *, size_t, uint32_t, uint32_t, uint64_t, uint64_t)
{}
#endif

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_PROFILE_EXPERT_WALL_H
