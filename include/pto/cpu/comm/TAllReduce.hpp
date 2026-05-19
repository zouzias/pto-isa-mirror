/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_TALL_REDUCE_HPP
#define PTO_CPU_COMM_TALL_REDUCE_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_REDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &recvTileData, ReduceOp op)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TALL_REDUCE: type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>, "TALL_REDUCE: tile type mismatch!");

    const int nranks = parallelGroup.GetSize();
    PTO_ASSERT(nranks > 0, "TALL_REDUCE: nranks must be > 0");

    TLOAD(accTileData, parallelGroup[0]);

    for (int r = 1; r < nranks; ++r) {
        TLOAD(recvTileData, parallelGroup[r]);
        detail::ReduceTiles(accTileData, recvTileData, op);
    }

    TSTORE(dstGlobalData, accTileData);
}

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TALL_REDUCE_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &pingTile, TileData &pongTile, ReduceOp op)
{
    TALL_REDUCE_IMPL(parallelGroup, dstGlobalData, accTileData, pingTile, op);
    (void)pongTile;
}

template <CollEngine engine = CollEngine::CCU, typename... Args>
PTO_INTERNAL void TALL_REDUCE_CCU_IMPL(Args &&...)
{
    static_assert(engine != CollEngine::CCU, "TALL_REDUCE<CCU> not available on CPU simulator");
}

} // namespace comm
} // namespace pto

#endif // PTO_CPU_COMM_TALL_REDUCE_HPP
