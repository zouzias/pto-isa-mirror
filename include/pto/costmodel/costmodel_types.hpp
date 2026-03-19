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
#include <iostream>

namespace pto {

struct CostModelStats {
    int total_repeats = 0;
    int masked_repeats = 0;
    int strided_repeats = 0;
    int count_mode_calls = 0;

    std::string cceInstName;
    int repeats = 0;
    int maskCount = 0;
    int maskmode = 0;
    int dstBlockStride;
    int src0BlockStride;
    int src1BlockStride;
    int dstRepeatStride;
    int src0RepeatStride;
    int src1RepeatStride;
    int order;  // vcmax/vcmin专用

    int sid = 0;
    int nBurst;
    int lenBurst;
    int srcGap;
    int dstGap;

    void setCceInstName(std::string cceInstName_)
    {
        cceInstName = cceInstName_;
    }

    // BinOp
    CostModelStats(const std::string cceInstName_, int repeats_, int dstBlockStride_, int src0BlockStride_,
                        int src1BlockStride_, int dstRepeatStride_, int src0RepeatStride_, int src1RepeatStride_,
                        int maskCount_, int order_ = 2) :
    cceInstName(cceInstName_), repeats(repeats_),
    dstBlockStride(dstBlockStride_), src0BlockStride(src0BlockStride_), src1BlockStride(src1BlockStride_),
    dstRepeatStride(dstRepeatStride_), src0RepeatStride(src0RepeatStride_), src1RepeatStride(src1RepeatStride_),
    maskCount(maskCount_), order(order_) {}

    // pipe_barrier
    CostModelStats(const std::string cceInstName_) :
    cceInstName(cceInstName_) {}

    // move
    CostModelStats(const std::string cceInstName_, int nBurst_, int lenBurst_, int srcGap_, int dstGap_) :
    cceInstName(cceInstName_), nBurst(nBurst_), lenBurst(lenBurst_), srcGap(srcGap_), dstGap(dstGap_) {}

    // BinSOp UnaryOp
    CostModelStats(const std::string cceInstName_, int repeats_, int dstBlockStride_, int srcBlockStride_,
                   int dstRepeatStride_, int srcRepeatStride_, int maskCount_) :
    cceInstName(cceInstName_), repeats(repeats_), dstBlockStride(dstBlockStride_), src0BlockStride(srcBlockStride_),
    dstRepeatStride(dstRepeatStride_), src0RepeatStride(srcRepeatStride_), maskCount(maskCount_) {}

    // GroupOp
    CostModelStats(const std::string cceInstName_, int repeats_, int dstRepeatStride_, int srcBlockStride_,
                    int srcRepeatStride_, int maskCount_) :
    cceInstName(cceInstName_), repeats(repeats_), dstRepeatStride(dstRepeatStride_), src0BlockStride(srcBlockStride_),
    src0RepeatStride(srcRepeatStride_), maskCount(maskCount_) {}

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
