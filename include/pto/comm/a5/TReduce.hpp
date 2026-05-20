/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TREDUCE_HPP
#define PTO_COMM_A5_TREDUCE_HPP

// Suppress the a2a3-side deferred-fail TREDUCE_CCU_IMPL stub when we are
// building for A5 — the real overload lives below. Without this guard the
// `Args&&...` stub competes with our typed signature in the overload set and
// can win under partial ordering on some compilers (observed on bisheng).
// Mirrors a5/TBroadCast.hpp / a5/TScatter.hpp / a5/TGather.hpp.
#define PTO_COMM_A5_TREDUCE_PROVIDED 1

// AIV path - shared with a2a3, forwarded to avoid code duplication.
#include "pto/comm/a2a3/TReduce.hpp"

// CCU path - A5-only implementation used by TREDUCE<CollEngine::CCU>.
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

// Leading `CollEngine` placeholder mirrors the TPUT_ASYNC_IMPL<engine> pattern:
// callers write `TREDUCE_CCU_IMPL<engine>(...)`, which turns the qualified
// template-id into a dependent name so the discarded `if constexpr (engine ==
// CCU)` branch in pto_comm_inst.hpp performs no lookup on non-A5 builds.
//
// Data-flow contract (this IMPL only handles the AIV trigger side; the CCU
// micro-code half lives in pto/npu/comm/async/ccu/ccu_reduce_kernel.hpp):
//
//   ctx.inputSource == HostManaged
//     parallelGroup[selfIdx] is assumed already populated by the host (or by
//     a prior op). AIV does NOT touch accTileData; the IMPL only writes the
//     CKE gate. recvTileData / dstGlobalData / op are unused here. This is
//     the legacy doorbell-only behaviour.
//
//   ctx.inputSource == AivStored
//     accTileData holds this rank's partial (produced by an upstream AIV
//     stage, e.g. a GEMM). The IMPL TSTOREs it into parallelGroup[selfIdx]
//     (whose VA must equal the host-side CcuReduceTaskArg::inputAddr),
//     pipe_barrier(PIPE_MTE3)s so the write drains, then writes the CKE
//     gate. recvTileData is reserved for a downstream AIV stage that wants
//     to TLOAD dstGlobalData after CCU is done (caller must aclrtSynchronize
//     ccuStream before launching that stage — device-side wait-on-CCU-done
//     is not yet exposed by PTO).
//
//   op is forwarded by the caller for signature symmetry with the AIV path,
//   but is already baked into the CCU micro-code at HcclCcuKernelRegister
//   time via CcuReduceKernelArg::reduceOp. The IMPL itself does not consume
//   it; callers MUST keep the two in sync.
template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TREDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &recvTileData, ReduceOp op,
                                   const CcuTriggerContext &ctx, WaitEvents &...events)
{
    WaitAllEvents(events...);

    if (ctx.inputSource == CcuInputSource::AivStored) {
        // Use explicit set_flag/wait_flag to annotate the V→MTE3 data
        // dependency for the CCE compiler's instruction scheduler.
        // pipe_barrier(PIPE_ALL) alone is a hardware fence but does NOT
        // provide the compiler-level dependency edge that CCE requires to
        // correctly schedule the MTE3 read (TSTORE) after V-pipe writes
        // (TEXPANDS/TADD) to UB.
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(parallelGroup[static_cast<int>(ctx.selfIdx)], accTileData);
        pipe_barrier(PIPE_MTE3);
    }

    CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, accTileData);

    (void)dstGlobalData;
    (void)recvTileData;
    (void)op;
}

template <CollEngine = CollEngine::CCU, typename ParallelGroupType, typename GlobalDstData, typename TileData,
          typename... WaitEvents>
PTO_INTERNAL void TREDUCE_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                   TileData &accTileData, TileData &pingTileData, TileData &pongTileData, ReduceOp op,
                                   const CcuTriggerContext &ctx, WaitEvents &...events)
{
    WaitAllEvents(events...);

    // CCU micro-code does its own staging across ranks; AIV-side ping/pong
    // is meaningful only for the post-reduce consume stage. For the publish
    // step we only need to land accTileData once.
    if (ctx.inputSource == CcuInputSource::AivStored) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(parallelGroup[static_cast<int>(ctx.selfIdx)], accTileData);
        pipe_barrier(PIPE_MTE3);
    }

    CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, accTileData);

    (void)dstGlobalData;
    (void)pingTileData;
    (void)pongTileData;
    (void)op;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TREDUCE_HPP
