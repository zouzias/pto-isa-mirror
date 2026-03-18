/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_TYPES_HPP
#define PTO_COSTMODEL_TYPES_HPP

#include <cstdint>

namespace pto {

struct CostModelStats {
    int total_repeats = 0;
    int masked_repeats = 0;
    int strided_repeats = 0;
    int count_mode_calls = 0;

    void AddRepeat(unsigned repeats, bool isMasked = false, bool isStrided = false)
    {
        total_repeats += static_cast<int>(repeats);
        if (isMasked) {
            masked_repeats += static_cast<int>(repeats);
        }
        if (isStrided) {
            strided_repeats += static_cast<int>(repeats);
        }
    }

    void AddCountModeCall()
    {
        ++count_mode_calls;
    }
};

struct CostModelParams {
    float startup_cycles = 0.0f;
    float completion_cycles = 0.0f;
    float per_repeat_cycles = 0.0f;
    float interval_cycles = 0.0f;
    float mask_effect = 1.0f;
    float bank_conflict_cycles = 0.0f;
};

inline void RecordCountMode(CostModelStats &stats)
{
    stats.AddCountModeCall();
}

inline void RecordRepeat(CostModelStats &stats, unsigned repeats, bool isMasked = false, bool isStrided = false)
{
    stats.AddRepeat(repeats, isMasked, isStrided);
}

} // namespace pto

#endif
