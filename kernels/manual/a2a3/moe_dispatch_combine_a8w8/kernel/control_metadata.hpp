/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_

#include <pto/common/type.hpp>

#include "moe_dispatch_combine_a8w8_types.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

namespace moe_dispatch_combine_a8w8 {

template <typename T>
struct GmArrayView {
    __gm__ T *ptr = nullptr;

    AICORE inline __gm__ T *At(uint64_t index) const
    {
        return ptr + index;
    }
};

struct TokenPerExpertView {
    __gm__ int32_t *ptr = nullptr;
    uint32_t rankNum = 0;
    uint32_t expertPerRank = 0;

    AICORE inline uint64_t Index(uint32_t tokenOwnerRank, uint32_t expertOwnerRank, uint32_t localExpert) const
    {
        return (static_cast<uint64_t>(tokenOwnerRank) * rankNum + expertOwnerRank) * expertPerRank + localExpert;
    }

    AICORE inline __gm__ int32_t *At(uint32_t tokenOwnerRank, uint32_t expertOwnerRank,
                                     uint32_t localExpert) const
    {
        return ptr + Index(tokenOwnerRank, expertOwnerRank, localExpert);
    }
};

struct RankExpertView {
    __gm__ int32_t *ptr = nullptr;
    uint32_t expertPerRank = 0;

    AICORE inline uint64_t Index(uint32_t tokenOwnerRank, uint32_t localExpert) const
    {
        return static_cast<uint64_t>(tokenOwnerRank) * expertPerRank + localExpert;
    }

    AICORE inline __gm__ int32_t *At(uint32_t tokenOwnerRank, uint32_t localExpert) const
    {
        return ptr + Index(tokenOwnerRank, localExpert);
    }
};

struct ControlMetadataView {
    TokenPerExpertView tokenPerExpert;
    GmArrayView<int32_t> blockTokenPerExpert;
    GmArrayView<int32_t> blockPrefixPerExpert;
    GmArrayView<int32_t> expandedRowIdx;
    GmArrayView<int32_t> dispatchOffset;
    RankExpertView cumsumMM;
    RankExpertView preSumBeforeRank;
    GmArrayView<int32_t> expertTokenNums;
    GmArrayView<int32_t> countReadySignal;
    GmArrayView<int32_t> combineDoneSignal;
    GmArrayView<int32_t> debugCounters;
};

AICORE inline __gm__ int32_t *I32Field(GM_ADDR base, const FieldLayout &field)
{
    return reinterpret_cast<__gm__ int32_t *>(base + field.offset);
}

AICORE inline ControlMetadataView MakeControlMetadataView(GM_ADDR workspace, GM_ADDR peerWindow,
                                                          const WorkspaceLayout &workspaceLayout,
                                                          const PeerWindowLayout &peerWindowLayout,
                                                          const ShapeConfig &shape)
{
    ControlMetadataView view;
    view.tokenPerExpert = TokenPerExpertView{I32Field(peerWindow, peerWindowLayout.tokenPerExpertMatrix),
                                             shape.rankNum, shape.expertPerRank};
    view.blockTokenPerExpert.ptr = I32Field(workspace, workspaceLayout.blockTokenPerExpert);
    view.blockPrefixPerExpert.ptr = I32Field(workspace, workspaceLayout.blockPrefixPerExpert);
    view.expandedRowIdx.ptr = I32Field(workspace, workspaceLayout.expandedRowIdx);
    view.dispatchOffset.ptr = I32Field(workspace, workspaceLayout.dispatchOffset);
    view.cumsumMM = RankExpertView{I32Field(workspace, workspaceLayout.cumsumMM), shape.expertPerRank};
    view.preSumBeforeRank = RankExpertView{I32Field(workspace, workspaceLayout.preSumBeforeRank),
                                           shape.expertPerRank};
    view.expertTokenNums.ptr = I32Field(workspace, workspaceLayout.expertTokenNums);
    view.countReadySignal.ptr = I32Field(peerWindow, peerWindowLayout.countReadySignal);
    view.combineDoneSignal.ptr = I32Field(peerWindow, peerWindowLayout.combineDoneSignal);
    view.debugCounters.ptr = I32Field(peerWindow, peerWindowLayout.debugCounters);
    return view;
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_KERNEL_CONTROL_METADATA_HPP_
