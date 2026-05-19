/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TREDUCE_SCATTER_HPP
#define PTO_COMM_A5_TREDUCE_SCATTER_HPP

#define PTO_COMM_A5_TREDUCE_SCATTER_PROVIDED 1

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

// TREDUCE_SCATTER_CCU_IMPL — A5 CCU ReduceScatter.
//
// Data-flow contract (mirrors a5/TReduce.hpp for the AIV trigger side;
// the CCU micro-code half lives in ccu_reduce_scatter_kernel.hpp):
//
//   ctx.inputSource == HostManaged
//     The host has already written this rank's full input into the input HBM
//     buffer.  AIV does NOT touch accTileData; only writes the CKE gate.
//
//   ctx.inputSource == AivStored
//     accTileData holds this rank's partial data (produced by an upstream AIV
//     stage).  The IMPL TSTOREs it into parallelGroup[selfIdx]+offset (the
//     input HBM slice that the CCU will reduce), pipe_barrier(PIPE_MTE3)s to
//     drain, then writes the CKE gate.
//
// ReduceScatter semantics: each rank's input is size N*nranks. After the op,
// each rank r holds the reduced slice [r*sliceSize .. (r+1)*sliceSize).
// The CCU kernel reads from input[r] + offset (per-rank), reduces, and writes
// to output[r].

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TREDUCE_SCATTER_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
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
PTO_INTERNAL void TREDUCE_SCATTER_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
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

#endif // PTO_COMM_A5_TREDUCE_SCATTER_HPP
