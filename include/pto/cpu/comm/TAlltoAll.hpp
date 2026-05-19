/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_TALL_TO_ALL_HPP
#define PTO_CPU_COMM_TALL_TO_ALL_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_TO_ALL_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TALL_TO_ALL: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TALL_TO_ALL: tile type mismatch!");

    const int selfIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0 && selfIdx >= 0 && selfIdx < nranks, "TALL_TO_ALL: invalid group");

    auto &srcRef = parallelGroup[0];
    const int totalSrcRows = srcRef.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = srcRef.GetShape(GlobalTensorDim::DIM_4);
    const int sliceRows = totalSrcRows / nranks;
    PTO_ASSERT(totalSrcRows == sliceRows * nranks, "TALL_TO_ALL: src rows not divisible by nranks");

    const int srcStride3 = srcRef.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

    using DynShape = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;

    DynShape sliceShape(1, 1, 1, sliceRows, gShape4);
    DynStride srcViewStride(srcRef.GetStride(GlobalTensorDim::DIM_0), srcRef.GetStride(GlobalTensorDim::DIM_1),
                            srcRef.GetStride(GlobalTensorDim::DIM_2), srcStride3,
                            srcRef.GetStride(GlobalTensorDim::DIM_4));
    DynStride dstViewStride(
        dstGlobalData.GetStride(GlobalTensorDim::DIM_0), dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
        dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3, dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

    for (int r = 0; r < nranks; ++r) {
        int64_t srcOffset = static_cast<int64_t>(selfIdx) * sliceRows * srcStride3;
        SrcViewT srcView(parallelGroup[r].data() + srcOffset, sliceShape, srcViewStride);
        TLOAD(stagingTileData, srcView);
        int64_t dstOffset = static_cast<int64_t>(r) * sliceRows * dstStride3;
        DstViewT dstView(dstGlobalData.data() + dstOffset, sliceShape, dstViewStride);
        TSTORE(dstView, stagingTileData);
    }
}

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_TO_ALL_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &pingTileData, TileData &pongTileData)
{
    TALL_TO_ALL_IMPL(parallelGroup, dstGlobalData, pingTileData);
    (void)pongTileData;
}

template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TALL_TO_ALL_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU, "TALL_TO_ALL<CCU> not available on CPU simulator");
}

} // namespace comm
} // namespace pto

#endif // PTO_CPU_COMM_TALL_TO_ALL_HPP
