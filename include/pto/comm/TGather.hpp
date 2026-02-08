/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGATHER_HPP
#define PTO_COMM_TGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TGATHER_IMPL: Gather operation - root collects data from all ranks
// 
// The calling NPU is the root and gathers data from all ranks into a single
// concatenated output buffer on the root.
// ============================================================================

namespace detail {
template <typename GlobalRefData, typename GlobalBaseData>
PTO_INTERNAL GlobalRefData MakeViewWithBase(GlobalBaseData &base, GlobalRefData &ref, int64_t elemOffset)
{
    using ShapeT = typename GlobalRefData::Shape;
    using StrideT = typename GlobalRefData::Stride;

    ShapeT shape(ref.GetShape(GlobalTensorDim::DIM_0), ref.GetShape(GlobalTensorDim::DIM_1),
        ref.GetShape(GlobalTensorDim::DIM_2), ref.GetShape(GlobalTensorDim::DIM_3), ref.GetShape(GlobalTensorDim::DIM_4));
    StrideT stride(ref.GetStride(GlobalTensorDim::DIM_0), ref.GetStride(GlobalTensorDim::DIM_1),
        ref.GetStride(GlobalTensorDim::DIM_2), ref.GetStride(GlobalTensorDim::DIM_3), ref.GetStride(GlobalTensorDim::DIM_4));

    return GlobalRefData(base.data() + elemOffset, shape, stride);
}
} // namespace detail

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                               TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;
    
    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, 
        "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
        "TGATHER: TileData element type must match GlobalData element type");

    const int my_rank = parallelGroup.GetRank();
    const int nranks = parallelGroup.GetSize();

    // Check PG size 
    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");

    if (nranks == 1) {
        // Single rank: just copy local data to output
        TLOAD(stagingTileData, parallelGroup[my_rank]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobalData, stagingTileData);
        return;
    }

    PTO_ASSERT(dstGlobalData.GetShape(GlobalTensorDim::DIM_3) >= nranks,
        "TGATHER: dstGlobalData dim3 must be >= nranks");

    // Root gathers data from all ranks using staged transfer
    for (int teamRank = 0; teamRank < nranks; ++teamRank) {
        // Load remote/local data to UB
        TLOAD(stagingTileData, parallelGroup[teamRank]);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        
        // Store to appropriate position in output
        const int64_t offset = static_cast<int64_t>(teamRank) *
            static_cast<int64_t>(dstGlobalData.GetStride(GlobalTensorDim::DIM_3));
        GlobalSrcData dstView = detail::MakeViewWithBase<GlobalSrcData>(dstGlobalData, parallelGroup[teamRank], offset);
        TSTORE(dstView, stagingTileData);
        
        if (teamRank < nranks - 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGATHER_HPP
