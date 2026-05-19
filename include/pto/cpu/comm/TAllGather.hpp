/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_TALL_GATHER_HPP
#define PTO_CPU_COMM_TALL_GATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_GATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TALL_GATHER: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TALL_GATHER: tile type mismatch!");

    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0, "TALL_GATHER: nranks must be > 0");

    auto &srcRef = parallelGroup[0];
    const int perRankRows = srcRef.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcRef.GetShape(GlobalTensorDim::DIM_4);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

    using DynShape = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;

    DynShape perRankShape(srcRef.GetShape(GlobalTensorDim::DIM_0), srcRef.GetShape(GlobalTensorDim::DIM_1),
                          srcRef.GetShape(GlobalTensorDim::DIM_2), perRankRows, gShape4);
    DynStride dstViewStride(
        dstGlobalData.GetStride(GlobalTensorDim::DIM_0), dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
        dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3, dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

    for (int r = 0; r < nranks; ++r) {
        TLOAD(stagingTileData, parallelGroup[r]);
        int64_t dstOffset = static_cast<int64_t>(r) * perRankRows * dstStride3;
        DstViewT dstView(dstGlobalData.data() + dstOffset, perRankShape, dstViewStride);
        TSTORE(dstView, stagingTileData);
    }
}

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_GATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &pingTileData, TileData &pongTileData)
{
    TALL_GATHER_IMPL(parallelGroup, dstGlobalData, pingTileData);
    (void)pongTileData;
}

template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TALL_GATHER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU, "TALL_GATHER<CCU> not available on CPU simulator");
}

} // namespace comm
} // namespace pto

#endif // PTO_CPU_COMM_TALL_GATHER_HPP
