/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TALL_REDUCE_HPP
#define PTO_COMM_A5_TALL_REDUCE_HPP

#define PTO_COMM_A5_TALL_REDUCE_PROVIDED 1

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

// TALL_REDUCE_CCU_IMPL — A5 CCU AllReduce.
//
// AIV trigger side. The CCU micro-code (ccu_all_reduce_kernel.hpp) does
// ReduceScatter + AllGather internally in a single kernel launch.
//
//   ctx.inputSource == HostManaged
//     Host has already written this rank's input. AIV only doorbells CKE.
//
//   ctx.inputSource == AivStored
//     accTileData holds this rank's partial. IMPL TSTOREs it into
//     parallelGroup[selfIdx] (= input HBM) before CKE doorbell.

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TALL_REDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &recvTileData, ReduceOp op,
                                       const CcuTriggerContext &ctx, WaitEvents &...events)
{
    WaitAllEvents(events...);

    if (ctx.inputSource == CcuInputSource::AivStored) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(parallelGroup[static_cast<int>(ctx.selfIdx)], accTileData);
        pipe_barrier(PIPE_MTE3);
    }

    pto::comm::ccu::CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, accTileData);

    (void)dstGlobalData;
    (void)recvTileData;
    (void)op;
}

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TALL_REDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                       TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                                       ReduceOp op, const CcuTriggerContext &ctx, WaitEvents &...events)
{
    WaitAllEvents(events...);

    if (ctx.inputSource == CcuInputSource::AivStored) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(parallelGroup[static_cast<int>(ctx.selfIdx)], accTileData);
        pipe_barrier(PIPE_MTE3);
    }

    pto::comm::ccu::CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, accTileData);

    (void)dstGlobalData;
    (void)pingTileData;
    (void)pongTileData;
    (void)op;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TALL_REDUCE_HPP
