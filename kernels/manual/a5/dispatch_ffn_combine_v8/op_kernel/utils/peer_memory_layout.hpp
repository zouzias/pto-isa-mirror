/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_FFN_COMBINE_V8_PEER_MEMORY_LAYOUT_HPP
#define DISPATCH_FFN_COMBINE_V8_PEER_MEMORY_LAYOUT_HPP

#include "common_helpers.hpp"
#include "const_args.hpp"
#include "hccl_window.hpp"

namespace dispatch_ffn_combine_v8 {

struct PeerMemoryLayout {
    int64_t offsetA = 0;
    int64_t offsetScale2 = 0;
    int64_t offsetPeerTokenPerExpert = 0;
    int64_t offsetD = 0;

    AICORE inline void Init(const PtoRemoteWindow &remoteWindow)
    {
        offsetA = 0;
        offsetScale2 = offsetA + alignUp(remoteWindow.SegmentSize() / 3, 512);
        offsetD = offsetScale2 + MB_SIZE;
        offsetPeerTokenPerExpert = remoteWindow.SegmentSize() - 2 * MB_SIZE;
    }
};

struct PeerRowMapper {
    __gm__ int32_t *cumsumMM = nullptr;
    __gm__ int32_t *preSumBeforeRank = nullptr;
    __gm__ int32_t *tokenPerExpert = nullptr;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t expertPerRank = 0;
    uint32_t expertNumAligned = 0;
    uint32_t maxOutputSize = 0;

    AICORE inline void Init(__gm__ int32_t *cumsum, __gm__ int32_t *preSum, __gm__ int32_t *tokenCounts,
                            uint32_t localRank, uint32_t worldSize, uint32_t localExpertPerRank,
                            uint32_t alignedExpertNum, uint32_t maxRows)
    {
        cumsumMM = cumsum;
        preSumBeforeRank = preSum;
        tokenPerExpert = tokenCounts;
        rank = localRank;
        rankSize = worldSize;
        expertPerRank = localExpertPerRank;
        expertNumAligned = alignedExpertNum;
        maxOutputSize = maxRows;
    }

    AICORE inline uint32_t GlobalExpert(uint32_t groupIdx) const
    {
        return rank * expertPerRank + groupIdx;
    }

    AICORE inline uint32_t CurrentM(uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(cumsumMM[static_cast<uint64_t>(rankSize - 1U) * expertPerRank + groupIdx]);
    }

    AICORE inline uint32_t ClipCurrentM(uint32_t currentMRaw, uint32_t groupBase) const
    {
        if (groupBase >= maxOutputSize) {
            return 0U;
        }
        const uint32_t remaining = maxOutputSize - groupBase;
        return currentMRaw > remaining ? remaining : currentMRaw;
    }

    AICORE inline uint32_t CumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        if (srcRank == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(cumsumMM[static_cast<uint64_t>(srcRank - 1U) * expertPerRank + groupIdx]);
    }

    AICORE inline uint32_t RowsRaw(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(
            tokenPerExpert[static_cast<uint64_t>(srcRank) * expertNumAligned + GlobalExpert(groupIdx)]);
    }

    AICORE inline uint32_t RowsClipped(uint32_t srcRowOffset, uint32_t rowsRaw) const
    {
        if (srcRowOffset >= maxOutputSize) {
            return 0U;
        }
        const uint32_t remaining = maxOutputSize - srcRowOffset;
        return rowsRaw > remaining ? remaining : rowsRaw;
    }

    AICORE inline uint32_t SourceRowOffset(uint32_t groupBase, uint32_t srcRank, uint32_t groupIdx) const
    {
        return groupBase + CumsumBeforeSource(srcRank, groupIdx);
    }

    AICORE inline uint32_t SourceRows(uint32_t groupBase, uint32_t srcRank, uint32_t groupIdx) const
    {
        return RowsClipped(SourceRowOffset(groupBase, srcRank, groupIdx), RowsRaw(srcRank, groupIdx));
    }

    AICORE inline uint32_t DstRowOffset(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(preSumBeforeRank[static_cast<uint64_t>(srcRank) * expertPerRank + groupIdx]);
    }
};

} // namespace dispatch_ffn_combine_v8

#endif // DISPATCH_FFN_COMBINE_V8_PEER_MEMORY_LAYOUT_HPP
