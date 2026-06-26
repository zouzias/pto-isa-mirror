/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_POSTSORT_IMPL_HPP
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_POSTSORT_IMPL_HPP

AICORE inline void FrontStoreExpertTokenOutCountSlice(const FrontReorderCommonState &op, int32_t firstExpertId,
                                                      uint32_t copyLength, uint64_t countUb, uint64_t storeScratchUb)
{
    if (copyLength == 0U) {
        return;
    }
    const uint32_t first = static_cast<uint32_t>(firstExpertId);
    const uint32_t alignedFirst = first / kFrontInt32OneBlockElems * kFrontInt32OneBlockElems;
    const uint32_t headElems = first - alignedFirst;
    const uint32_t alignedLength = static_cast<uint32_t>(alignUp(headElems + copyLength, kFrontInt32OneBlockElems));

    PtoFillUb<int32_t>(storeScratchUb, 0, alignedLength);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (uint32_t idx = 0U; idx < copyLength; ++idx) {
        PtoSetValue<int32_t>(storeScratchUb, headElems + idx, PtoGetValue<int32_t>(countUb, idx));
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreAtomicAddVector<int32_t>(op.localTokenPerExpertPtr_ + alignedFirst, storeScratchUb, alignedLength);
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
}

AICORE inline void FrontUpdateExpertTokenOutCount(uint64_t countUb, int32_t curExpertId, int32_t &lastExpertId,
                                                  int32_t &tokenCount)
{
    tokenCount++;
    if (lastExpertId < curExpertId) {
        PtoSetValue<int32_t>(countUb, static_cast<uint32_t>(lastExpertId), tokenCount - 1);
        tokenCount = 1;
        lastExpertId = curExpertId;
    }
}

AICORE inline void FrontBuildExpertTokenOut(const FrontReorderCommonState &op)
{
    const FrontSrcToDstTiling tiling = FrontBuildSrcToDstTiling(op.routeElems_, op.coreNum_);
    const uint32_t expertNumUbAlign = FrontExpertTokenOutExpertNumUbAlign(op.expertNum_);
    const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
    if (!FrontExpertTokenOutInputsValid(op.routeElems_, expertNumUbAlign) || !coreLoop.active) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        return;
    }
    const FrontExpertTokenOutUbPlan ubPlan(op.expertNum_, coreLoop.perLoopRows);
    if (!FrontRowLoopUbFits(coreLoop, ubPlan.requiredBytes)) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        return;
    }

    PtoFillUb<int32_t>(ubPlan.countUb, 0, expertNumUbAlign);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
    int32_t tokenCount = 0;
    int32_t lastExpertId = -1;
    for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
        const uint32_t currentLoopRows = loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
        PtoLoadVector<int32_t>(ubPlan.chunkUb,
                               op.frontExpandedExpertPtr_ + coreLoop.coreBase + loop * coreLoop.perLoopRows,
                               currentLoopRows);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        for (uint32_t idx = 0U; idx < currentLoopRows; ++idx) {
            const int32_t expertId = PtoGetValue<int32_t>(ubPlan.chunkUb, idx);
            if (lastExpertId == -1) {
                lastExpertId = expertId;
            }
            FrontUpdateExpertTokenOutCount(ubPlan.countUb, expertId, lastExpertId, tokenCount);
        }
        if (lastExpertId >= 0) {
            PtoSetValue<int32_t>(ubPlan.countUb, static_cast<uint32_t>(lastExpertId), tokenCount);
        }
    }
    if (lastExpertId >= 0) {
        FrontStoreExpertTokenOutCountSlice(op, 0, expertNumUbAlign, ubPlan.countUb, ubPlan.scratchUb);
    }
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

AICORE inline void FrontPrepareSrcToDstAssist(const FrontReorderCommonState &op, uint64_t assistUb,
                                              const FrontSrcToDstTiling &tiling)
{
    using AssistTile = PtoVecTile<int32_t, kFrontSrcToDstAssistElems>;
    AssistTile assistTile(1, kFrontSrcToDstAssistElems);
    pto::TASSIGN(assistTile, assistUb);
    const int32_t baseRow = static_cast<int32_t>(op.coreIdx_ * tiling.perCoreRows);
    for (uint32_t idx = 0U; idx < kFrontSrcToDstAssistElems; ++idx) {
        int32_t rowOffset = 0;
        if (idx % kFrontInt32OneBlockElems == 0U) {
            rowOffset = baseRow + static_cast<int32_t>(idx / kFrontInt32OneBlockElems);
        }
        assistTile.SetValue(idx, rowOffset);
    }
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

AICORE inline void FrontComputeSrcToDstRows(uint32_t progress, uint32_t perLoopRows, uint32_t currentLoopRows,
                                            uint64_t outputUb, uint64_t assistUb)
{
    const uint32_t loops = (currentLoopRows + kFrontSrcToDstAssistIndexElems - 1U) / kFrontSrcToDstAssistIndexElems;
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        PtoAddScalarUb<int32_t, kFrontSrcToDstAssistElems>(
            outputUb + static_cast<uint64_t>(loop) * kFrontSrcToDstAssistElems * sizeof(int32_t), assistUb,
            kFrontSrcToDstAssistElems,
            static_cast<int32_t>(progress * perLoopRows + loop * kFrontSrcToDstAssistIndexElems));
    }
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}

AICORE inline void FrontCopyOutSrcToDstRows(const FrontReorderCommonState &op, uint32_t currentLoopRows,
                                            uint64_t inputUb, uint64_t outputUb, bool hasNextLoop)
{
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    for (uint32_t idx = 0U; idx < currentLoopRows; ++idx) {
        const uint32_t srcRoute = static_cast<uint32_t>(PtoGetValue<int32_t>(inputUb, idx));
        PtoStoreVector<int32_t>(op.expandedRowIdxPtr_ + srcRoute,
                                outputUb + static_cast<uint64_t>(idx) * kFrontInt32OneBlockElems * sizeof(int32_t), 1);
    }
    if (hasNextLoop) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
}
/*
inputUb  : 存 srcRoute 列表，大小约 perLoopRows * 4B
outputUb : 存 dstRow 列表，但每个 dstRow 占 8-int block，大小约 ceil(perLoopRows/256) * 256 * 8 *  4B
assistUb : 32 个 8-int block 模板，用来批量生成 dstRow，固定 1024B
8-int对齐是为了写GM的时候，由于是按照srcRoute乱序写，这么做能做到32B对齐
assistUb 是为了expandedRowIdx[srcRoute] = dstRow的时候， dstRow的生成32个元素依次加偏移性能差，可以用aiv批量加偏移
 */
AICORE inline void FrontBuildSrcToDst(const FrontReorderCommonState &op)
{
    const FrontSrcToDstTiling tiling = FrontBuildSrcToDstTiling(op.routeElems_, op.coreNum_);
    if (FrontSrcToDstInputsValid(op.routeElems_, tiling)) {
        const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
        if (coreLoop.active) {
            const FrontSrcToDstUbPlan ubPlan(coreLoop.perLoopRows);
            if (FrontRowLoopUbFits(coreLoop, ubPlan.requiredBytes)) {
                FrontPrepareSrcToDstAssist(op, ubPlan.assistUb, tiling);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
                    const uint32_t currentLoopRows =
                        loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
                    const bool hasNextLoop = loop + 1U < coreLoop.rowLoops;
                    PtoLoadVector<int32_t>(ubPlan.inputUb,
                                           op.frontExpandDstToSrcPtr_ + coreLoop.coreBase + loop * coreLoop.perLoopRows,
                                           currentLoopRows);
                    FrontComputeSrcToDstRows(loop, coreLoop.perLoopRows, currentLoopRows, ubPlan.outputUb,
                                             ubPlan.assistUb);
                    FrontCopyOutSrcToDstRows(op, currentLoopRows, ubPlan.inputUb, ubPlan.outputUb, hasNextLoop);
                }
            }
        }
    }
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

template <typename InputElement>
AICORE inline void FrontCopyInGatherExpandedRowIdx(const FrontReorderCommonState &op, uint32_t coreBase,
                                                   uint32_t progress, uint32_t perLoopRows, uint32_t currentLoopRows,
                                                   uint64_t indexUb)
{
    PtoLoadVector<int32_t>(indexUb, op.expandedRowIdxPtr_ + coreBase + progress * perLoopRows, currentLoopRows);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
}

template <typename InputElement>
AICORE inline void FrontRunGatherQuantLoop(const FrontReorderCommonState &op)
{
    const FrontGatherQuantTiling tiling = FrontBuildGatherQuantTiling(op.routeElems_, op.coreNum_, op.problemK_);
    const FrontCoreRowLoopView coreLoop = FrontGetCoreRowLoopView(op, tiling);
    if (FrontPostSortGatherQuantEnabled(tiling, op.problemK_) && coreLoop.active) {
        const FrontGatherQuantUbPlan<InputElement> ubPlan(op.problemK_, coreLoop.perLoopRows);
        if (FrontGatherQuantCoreLoopReady(coreLoop, ubPlan.requiredBytes)) {
            for (uint32_t loop = 0U; loop < coreLoop.rowLoops; ++loop) {
                uint32_t currentLoopRows =
                    loop == coreLoop.rowLoops - 1U ? coreLoop.lastLoopRows : coreLoop.perLoopRows;
                if (currentLoopRows == 0U && loop != coreLoop.rowLoops - 1U) {
                    currentLoopRows = coreLoop.perLoopRows;
                }
                FrontCopyInGatherExpandedRowIdx<InputElement>(op, coreLoop.coreBase, loop, coreLoop.perLoopRows,
                                                              currentLoopRows, ubPlan.indexUb);
                const uint32_t routeRowStart = coreLoop.coreBase + coreLoop.perLoopRows * loop;
                FrontQuantAndScatterPackedRows<InputElement>(op, routeRowStart, currentLoopRows, ubPlan.indexUb,
                                                             op.routeElems_, true, ubPlan.rawUb, ubPlan.fp32Ub,
                                                             ubPlan.tmpUb, ubPlan.outUb, ubPlan.scaleUb);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_POSTSORT_IMPL_HPP
