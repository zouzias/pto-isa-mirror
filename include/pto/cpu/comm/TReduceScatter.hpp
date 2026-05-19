/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_TREDUCE_SCATTER_HPP
#define PTO_CPU_COMM_TREDUCE_SCATTER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_SCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &recvTileData, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TREDUCE_SCATTER: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TREDUCE_SCATTER: tile type mismatch!");

    const int selfIdx = parallelGroup.GetRootIdx();
    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0 && selfIdx >= 0 && selfIdx < nranks, "TREDUCE_SCATTER: invalid group");

    auto &refTensor = parallelGroup[0];
    const int totalSrcRows = refTensor.GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = refTensor.GetShape(GlobalTensorDim::DIM_4);
    const int sliceRows = totalSrcRows / nranks;
    PTO_ASSERT(totalSrcRows == sliceRows * nranks, "TREDUCE_SCATTER: src rows not divisible by nranks");

    const int srcStride3 = refTensor.GetStride(GlobalTensorDim::DIM_3);
    const int64_t sliceOffset = static_cast<int64_t>(selfIdx) * sliceRows * srcStride3;

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;

    DynShape sliceShape(1, 1, 1, sliceRows, gShape4);
    DynStride srcStride(refTensor.GetStride(GlobalTensorDim::DIM_0), refTensor.GetStride(GlobalTensorDim::DIM_1),
                        refTensor.GetStride(GlobalTensorDim::DIM_2), srcStride3,
                        refTensor.GetStride(GlobalTensorDim::DIM_4));

    SrcViewT firstView(parallelGroup[0].data() + sliceOffset, sliceShape, srcStride);
    TLOAD(accTileData, firstView);

    for (int r = 1; r < nranks; ++r) {
        SrcViewT remoteView(parallelGroup[r].data() + sliceOffset, sliceShape, srcStride);
        TLOAD(recvTileData, remoteView);
        detail::ReduceTiles(accTileData, recvTileData, op);
    }

    TSTORE(dstGlobalData, accTileData);
}

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TREDUCE_SCATTER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &pingTile, TileData &pongTile, ReduceOp op)
{
    TREDUCE_SCATTER_IMPL(parallelGroup, dstGlobalData, accTileData, pingTile, op);
    (void)pongTile;
}

template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TREDUCE_SCATTER_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU, "TREDUCE_SCATTER<CCU> not available on CPU simulator");
}

} // namespace comm
} // namespace pto

#endif // PTO_CPU_COMM_TREDUCE_SCATTER_HPP
