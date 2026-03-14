/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#if (__CHECK_FEATURE_AT_PRECOMPILE) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../../../../../include/pto/pto-inst.hpp"
#include "../../../../../include/pto/npu/a2a3/custom/TSyncCVID.hpp"
#include "../../../../../include/pto/npu/a2a3/custom/TSync_Custom.hpp"
#include "moe_grouped_ffn_custom.h"

using namespace pto;

namespace {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
// Cross-core sync flags: shared across all AIV subblocks.
// AIC broadcast is received by ALL subblocks, so we use one flag pair per projection
// rather than per-subblock flags (per-subblock flags cause unconsumed FFTS counter
// accumulation from broadcast signals that no subblock waits on).
constexpr uint16_t kGateReadyFlag = 5;
constexpr uint16_t kUpReadyFlag = 7;
constexpr uint16_t kBlockEndFlag = 11;
// Each AIV subblock must signal when it has loaded a projection tile into UB so AIC can safely
// reuse the FIFO slot. Using distinct flags avoids the "double-consume" pitfall where a single
// flag counter can be satisfied by two frees from the same subblock.
constexpr uint16_t kGateFreeFlag0 = 6;
constexpr uint16_t kGateFreeFlag1 = 9;
constexpr uint16_t kUpFreeFlag0 = 8;
constexpr uint16_t kUpFreeFlag1 = 10;
constexpr uint32_t kGateUbOffset = 0x0;
constexpr uint32_t kAuxUbOffset = kGateUbOffset + kMoeVecSubTileBytes;
constexpr uint32_t kRecipUbOffset = kAuxUbOffset + kMoeVecSubTileBytes;
constexpr uint32_t kFusedGateUbOffset = 0x0;
constexpr uint32_t kFusedAuxUbOffset = kFusedGateUbOffset + kMoeVecTileBytes;
constexpr std::size_t kMoeFusedVecResidentBytes = 2 * kMoeVecTileBytes;
constexpr event_t kVecStoreEvent = EVENT_ID5;
constexpr event_t kVecGateLoadEvent = EVENT_ID6;
constexpr event_t kVecUpLoadEvent = EVENT_ID7;

using CrossCoreSync = TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD>;

static_assert(kMoeCvCommSlotBytes == static_cast<uint32_t>(pto::kCvCommSlotBytes),
              "moe workspace CV slot size must match TSYNC_CVID");
static_assert(kMoeCvMaxSlots == static_cast<uint32_t>(pto::kCvMaxCores),
              "moe workspace CV slot count must match TSYNC_CVID");
static_assert(kMoeFusedVecResidentBytes <= kMoeMaxVecUbBytes, "Fused vec tile UB allocation exceeds 192KB");

struct MoeWorkspaceLayout {
    __gm__ float *gateTiles;
    __gm__ float *upTiles;
    __gm__ uint8_t *cvCommBuf;
};

AICORE inline uint32_t GetCommSlotCountDevice(uint32_t blockDim)
{
    // Stable-first policy: use one projection slot per logical block and avoid wrapped slot reuse.
    return blockDim;
}

AICORE inline bool UseMoeCvCommDevice(uint32_t blockDim)
{
    (void)blockDim;
    return false;
}

AICORE inline std::size_t GetProjectionBufferBytesDevice(uint32_t blockDim)
{
    const std::size_t unalignedBytes =
        static_cast<std::size_t>(GetCommSlotCountDevice(blockDim)) * kMoeFifoDepth * kMoeVecTileBytes;
    return ((unalignedBytes + kMoeWorkspaceAlignBytes - 1) / kMoeWorkspaceAlignBytes) * kMoeWorkspaceAlignBytes;
}

AICORE inline MoeWorkspaceLayout GetMoeWorkspaceLayout(__gm__ uint8_t *workspace, uint32_t blockDim)
{
    const std::size_t projectionBytes = GetProjectionBufferBytesDevice(blockDim);
    MoeWorkspaceLayout layout;
    layout.gateTiles = reinterpret_cast<__gm__ float *>(workspace);
    layout.upTiles = reinterpret_cast<__gm__ float *>(workspace + projectionBytes);
    layout.cvCommBuf = workspace + 2 * projectionBytes;
    return layout;
}

AICORE inline int GetCommSlot(uint32_t workIdx, uint32_t blockDim, __gm__ uint8_t *cvCommBuf)
{
    if (!UseMoeCvCommDevice(blockDim)) {
        return static_cast<int>(workIdx);
    }
    return pto::TSYNC_CVID(static_cast<int>(workIdx), cvCommBuf);
}

AICORE inline __gm__ float *GetProjectionTile(__gm__ float *base, int commSlot)
{
    return base + static_cast<std::size_t>(commSlot) * kMoeVecTileElems;
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id)
{
    set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id)
{
    wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t kIter)
{
    if (kIter == 0) {
        TMATMUL(cTile, aTile, bTile);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);
    }
}

template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void InitBuffers(TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM], LeftTile aTile[BUFFER_NUM],
                               RightTile bTile[BUFFER_NUM], ResTile &cTile)
{
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * sizeof(bfloat16_t));
    TASSIGN(bMatTile[0], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * BUFFER_NUM * sizeof(bfloat16_t));
    TASSIGN(bMatTile[1], 0x0 + kMoeBaseM * kMoeBaseK * kMoeStepKa * BUFFER_NUM * sizeof(bfloat16_t) +
                               kMoeBaseK * kMoeBaseN * kMoeStepKb * sizeof(bfloat16_t));

    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
    TASSIGN(cTile, 0x0);
}

AICORE inline void InitSyncFlags()
{
    SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
    SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);
}

AICORE inline void WaitSyncFlags()
{
    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}

AICORE inline void WaitVecSubBlocksFree(uint16_t flag0, uint16_t flag1)
{
    wait_flag_dev(flag0);
    wait_flag_dev(flag1);
}

AICORE inline void SignalVecSubBlockFree(uint32_t subBlockIdx, uint16_t flag0, uint16_t flag1)
{
    const uint16_t flagId = (subBlockIdx == 0) ? flag0 : flag1;
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, flagId));
}

template <typename ResTile>
AICORE inline void StoreProjectionToOutput(ResTile &projTile, __gm__ float *currentDst, uint32_t rowCount)
{
    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeInterSize, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(currentDst, CValidShape(rowCount, kMoeBaseN));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, projTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void ProcessKIteration(uint32_t kIter, __gm__ bfloat16_t *currentSrc0, __gm__ bfloat16_t *currentSrc1,
                                     TileMatA aMatTile[BUFFER_NUM], TileMatB bMatTile[BUFFER_NUM],
                                     LeftTile aTile[BUFFER_NUM], RightTile bTile[BUFFER_NUM], ResTile &cTile,
                                     uint8_t &mte2DBFlag, uint8_t &mte1DBFlag, uint32_t currentM)
{
    using AValidShape = TileShape2D<bfloat16_t, -1, -1, Layout::ND>;
    using ABaseShape = BaseShape2D<bfloat16_t, kMoeBaseM, kMoeHiddenSize, Layout::ND>;
    using AGlobal = GlobalTensor<bfloat16_t, AValidShape, ABaseShape, Layout::ND>;

    using BValidShape = TileShape2D<bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, Layout::DN>;
    using BBaseShape = BaseShape2D<bfloat16_t, kMoeHiddenSize, kMoeInterSize, Layout::DN>;
    using BGlobal = GlobalTensor<bfloat16_t, BValidShape, BBaseShape, Layout::DN>;

    const uint32_t kModStepKa = kIter % kMoeStepKa;

    if (kModStepKa == 0) {
        AGlobal gmA(currentSrc0 + kIter * kMoeBaseK, AValidShape(currentM, kMoeBaseK * kMoeStepKa));
        BGlobal gmB(currentSrc1 + kIter * kMoeBaseK);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;

    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    }
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * kMoeBaseK);

    if (kModStepKa == 0) {
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    }
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % kMoeStepKb) * kMoeBaseK, 0);

    if ((kIter + 1) % kMoeStepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void RunGroupedProjectionToGM(__gm__ bfloat16_t *x, __gm__ bfloat16_t *weightDn,
                                            __gm__ int32_t *expertIds, __gm__ int32_t *rowOffsets, __gm__ int32_t *validRows,
                                            uint32_t colOffset,
                                            __gm__ float *tileDst)
{
    const uint32_t workIdx = get_block_idx();
    const uint32_t currentM = static_cast<uint32_t>(validRows[workIdx]);
    if (currentM == 0) {
        return;
    }

    const uint32_t expertId = static_cast<uint32_t>(expertIds[workIdx]);
    const uint32_t rowOffset = static_cast<uint32_t>(rowOffsets[workIdx]);

    __gm__ bfloat16_t *currentSrc0 = x + rowOffset * kMoeHiddenSize;
    __gm__ bfloat16_t *currentSrc1 = weightDn + (expertId * kMoeInterSize + colOffset) * kMoeHiddenSize;

    TileMatA aMatTile[BUFFER_NUM] = {TileMatA(currentM, kMoeBaseK * kMoeStepKa), TileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    TileMatB bMatTile[BUFFER_NUM];
    LeftTile aTile[BUFFER_NUM] = {LeftTile(currentM, kMoeBaseK), LeftTile(currentM, kMoeBaseK)};
    RightTile bTile[BUFFER_NUM] = {RightTile(kMoeBaseK, kMoeBaseN), RightTile(kMoeBaseK, kMoeBaseN)};
    ResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIteration(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                          mte1DBFlag, currentM);
    }

    WaitSyncFlags();

    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeBaseN, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(tileDst, CValidShape(currentM, kMoeBaseN));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <typename TileMatA, typename TileMatB, typename LeftTile, typename RightTile, typename ResTile>
AICORE inline void RunGroupedProjectionToGMWithOutput(__gm__ bfloat16_t *x, __gm__ bfloat16_t *weightDn,
                                                      __gm__ int32_t *expertIds, __gm__ int32_t *rowOffsets,
                                                      __gm__ int32_t *validRows, uint32_t colOffset,
                                                      __gm__ float *tileDst, __gm__ float *globalDst)
{
    const uint32_t workIdx = get_block_idx();
    const uint32_t currentM = static_cast<uint32_t>(validRows[workIdx]);
    if (currentM == 0) {
        return;
    }

    const uint32_t expertId = static_cast<uint32_t>(expertIds[workIdx]);
    const uint32_t rowOffset = static_cast<uint32_t>(rowOffsets[workIdx]);

    __gm__ bfloat16_t *currentSrc0 = x + rowOffset * kMoeHiddenSize;
    __gm__ bfloat16_t *currentSrc1 = weightDn + (expertId * kMoeInterSize + colOffset) * kMoeHiddenSize;

    TileMatA aMatTile[BUFFER_NUM] = {TileMatA(currentM, kMoeBaseK * kMoeStepKa), TileMatA(currentM, kMoeBaseK * kMoeStepKa)};
    TileMatB bMatTile[BUFFER_NUM];
    LeftTile aTile[BUFFER_NUM] = {LeftTile(currentM, kMoeBaseK), LeftTile(currentM, kMoeBaseK)};
    RightTile bTile[BUFFER_NUM] = {RightTile(kMoeBaseK, kMoeBaseN), RightTile(kMoeBaseK, kMoeBaseN)};
    ResTile cTile(currentM, kMoeBaseN);

    InitBuffers(aMatTile, bMatTile, aTile, bTile, cTile);
    InitSyncFlags();

    uint8_t mte2DBFlag = 0;
    uint8_t mte1DBFlag = 0;
    constexpr uint32_t kLoop = kMoeHiddenSize / kMoeBaseK;

    for (uint32_t kIter = 0; kIter < kLoop; ++kIter) {
        ProcessKIteration(kIter, currentSrc0, currentSrc1, aMatTile, bMatTile, aTile, bTile, cTile, mte2DBFlag,
                          mte1DBFlag, currentM);
    }

    WaitSyncFlags();

    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeBaseN, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(tileDst, CValidShape(currentM, kMoeBaseN));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

    StoreProjectionToOutput(cTile, globalDst, currentM);
}
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
template <typename VecTile>
AICORE inline void LoadProjectionFromGM(__gm__ float *currentSrc, uint32_t rowOffset, uint32_t rowCount, VecTile &dstTile)
{
    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeBaseN, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal srcGlobal(currentSrc + static_cast<std::size_t>(rowOffset) * kMoeBaseN, CValidShape(rowCount, kMoeBaseN));
    TLOAD(dstTile, srcGlobal);
}

template <typename VecTile>
AICORE inline void ApplySilu(VecTile &gateTile, VecTile &tmpTile, VecTile &recipTile)
{
    TNEG(tmpTile, gateTile);
    TEXP(tmpTile, tmpTile);
    TADDS(tmpTile, tmpTile, 1.0f);
    // A3: TRECIP does not support in-place src/dst, so use a dedicated recip tile.
    TRECIP(recipTile, tmpTile);
    TMUL(gateTile, gateTile, recipTile);
}

template <typename VecTile>
AICORE inline void StoreResult(VecTile &outTile, __gm__ float *currentDst, uint32_t rowOffset, uint32_t rowCount)
{
    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeInterSize, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal dstGlobal(currentDst + static_cast<std::size_t>(rowOffset) * kMoeInterSize,
                      CValidShape(rowCount, kMoeBaseN));
    TSTORE(dstGlobal, outTile);
}

template <typename VecTile>
AICORE inline void LoadProjectionFromOutput(__gm__ float *currentSrc, uint32_t rowCount, VecTile &dstTile)
{
    using CValidShape = TileShape2D<float, -1, -1, Layout::ND>;
    using CBaseShape = BaseShape2D<float, kMoeBaseM, kMoeInterSize, Layout::ND>;
    using CGlobal = GlobalTensor<float, CValidShape, CBaseShape, Layout::ND>;

    CGlobal srcGlobal(currentSrc, CValidShape(rowCount, kMoeBaseN));
    TLOAD(dstTile, srcGlobal);
}

AICORE inline void GetVecSubBlockRange(uint32_t currentM, uint32_t &rowOffset, uint32_t &rowCount)
{
    const uint32_t subBlockIdx = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
    const uint32_t subBlockNum = static_cast<uint32_t>(AscendC::GetSubBlockNum());
    const uint32_t rowsPerSubBlock = (currentM + subBlockNum - 1U) / subBlockNum;
    rowOffset = subBlockIdx * rowsPerSubBlock;
    if (currentM <= rowOffset) {
        rowCount = 0;
        return;
    }

    const uint32_t rowsRemaining = currentM - rowOffset;
    rowCount = (rowsRemaining < rowsPerSubBlock) ? rowsRemaining : rowsPerSubBlock;
}
#endif

} // namespace

extern "C" __global__ AICORE void moe_grouped_gemm_custom(GM_ADDR x, GM_ADDR gate_weight_dn, GM_ADDR up_weight_dn,
                                                          GM_ADDR out, GM_ADDR expert_ids, GM_ADDR row_offsets,
                                                          GM_ADDR valid_rows, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    __gm__ bfloat16_t *xPtr = reinterpret_cast<__gm__ bfloat16_t *>(x);
    __gm__ bfloat16_t *gateWeightPtr = reinterpret_cast<__gm__ bfloat16_t *>(gate_weight_dn);
    __gm__ bfloat16_t *upWeightPtr = reinterpret_cast<__gm__ bfloat16_t *>(up_weight_dn);
    __gm__ float *outPtr = reinterpret_cast<__gm__ float *>(out);
    __gm__ int32_t *expertIdsPtr = reinterpret_cast<__gm__ int32_t *>(expert_ids);
    __gm__ int32_t *rowOffsetsPtr = reinterpret_cast<__gm__ int32_t *>(row_offsets);
    __gm__ int32_t *validRowsPtr = reinterpret_cast<__gm__ int32_t *>(valid_rows);

    const uint32_t workIdx = get_block_idx();
    const uint32_t blockDim = static_cast<uint32_t>(get_block_num());
    const uint32_t currentM = static_cast<uint32_t>(validRowsPtr[workIdx]);
    if (currentM == 0) {
        return;
    }

    auto workspaceLayout = GetMoeWorkspaceLayout(reinterpret_cast<__gm__ uint8_t *>(workspace), blockDim);
    const int commSlot = GetCommSlot(workIdx, blockDim, workspaceLayout.cvCommBuf);

    const uint32_t rowOffset = static_cast<uint32_t>(rowOffsetsPtr[workIdx]);
    __gm__ float *currentDst = outPtr + rowOffset * kMoeInterSize;



#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
    using TileMatA =
        Tile<TileType::Mat, bfloat16_t, kMoeBaseM, kMoeBaseK * kMoeStepKa, BLayout::ColMajor, -1, -1,
             SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, BLayout::RowMajor,
                          kMoeBaseK * kMoeStepKb, kMoeBaseN, SLayout::ColMajor>;
    using LeftTile = TileLeftCompact<bfloat16_t, kMoeBaseM, kMoeBaseK, -1, -1>;
    using RightTile = TileRightCompact<bfloat16_t, kMoeBaseK, kMoeBaseN, -1, -1>;
    using ResTile = TileAcc<float, kMoeBaseM, kMoeBaseN, -1, -1>;

    CrossCoreSync gateSync{kGateReadyFlag};
    CrossCoreSync upSync{kUpReadyFlag};

    for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
        const uint32_t colOffset = nTile * static_cast<uint32_t>(kMoeBaseN);
        const uint32_t bufSlot = nTile % static_cast<uint32_t>(kMoeFifoDepth);

        // Wait for VECTOR to free this buffer slot (skip for first kMoeFifoDepth tiles)
        if (nTile >= static_cast<uint32_t>(kMoeFifoDepth)) {
            WaitVecSubBlocksFree(kGateFreeFlag0, kGateFreeFlag1);
        }
        __gm__ float *curGateDst = GetProjectionTile(
            workspaceLayout.gateTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
        RunGroupedProjectionToGM<TileMatA, TileMatB, LeftTile, RightTile, ResTile>(
            xPtr, gateWeightPtr, expertIdsPtr, rowOffsetsPtr, validRowsPtr, colOffset, curGateDst);
        gateSync.record();

        if (nTile >= static_cast<uint32_t>(kMoeFifoDepth)) {
            WaitVecSubBlocksFree(kUpFreeFlag0, kUpFreeFlag1);
        }
        __gm__ float *curUpDst = GetProjectionTile(
            workspaceLayout.upTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
        RunGroupedProjectionToGM<TileMatA, TileMatB, LeftTile, RightTile, ResTile>(
            xPtr, upWeightPtr, expertIdsPtr, rowOffsetsPtr, validRowsPtr, colOffset, curUpDst);
        upSync.record();
    }

    // Drain: wait for VECTOR to finish with the last kMoeFifoDepth buffer slots
    for (uint32_t d = 0; d < static_cast<uint32_t>(kMoeFifoDepth); ++d) {
        WaitVecSubBlocksFree(kGateFreeFlag0, kGateFreeFlag1);
        WaitVecSubBlocksFree(kUpFreeFlag0, kUpFreeFlag1);
    }

#elif defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
    // Initialize VEC state explicitly. Running the mix kernel directly skips the auto-generated
    // wrapper path that would normally sanitize AIV state for matmul-based kernels.
    AscendC::SetAtomicNone();
    AscendC::SetVectorMask<uint64_t, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
    AscendC::SetMaskNorm();

    using VecTile = Tile<TileType::Vec, float, kMoeVecSubBlockRows, kMoeBaseN, BLayout::RowMajor, -1, -1>;

    CrossCoreSync gateSync{kGateReadyFlag};
    CrossCoreSync upSync{kUpReadyFlag};

    const uint32_t subBlockIdx = static_cast<uint32_t>(get_subblockid());
    const uint32_t subBlockRowOffset = subBlockIdx * static_cast<uint32_t>(kMoeVecSubBlockRows);
    uint32_t subBlockRows = 0;
    if (currentM > subBlockRowOffset) {
        const uint32_t rowsRemaining = currentM - subBlockRowOffset;
        subBlockRows = (rowsRemaining < static_cast<uint32_t>(kMoeVecSubBlockRows)) ?
                           rowsRemaining :
                           static_cast<uint32_t>(kMoeVecSubBlockRows);
    }

    VecTile gateTile(subBlockRows, kMoeBaseN);
    VecTile auxTile(subBlockRows, kMoeBaseN);
    VecTile recipTile(subBlockRows, kMoeBaseN);

    TASSIGN(gateTile, kGateUbOffset);
    TASSIGN(auxTile, kAuxUbOffset);
    TASSIGN(recipTile, kRecipUbOffset);

    if (subBlockRows == 0) {
        // Empty subblock: participate in sync protocol without doing real work
        for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
            gateSync.wait();
            SignalVecSubBlockFree(subBlockIdx, kGateFreeFlag0, kGateFreeFlag1);
            upSync.wait();
            SignalVecSubBlockFree(subBlockIdx, kUpFreeFlag0, kUpFreeFlag1);
        }
    } else {
        SetFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);

        for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
            const uint32_t colOffset = nTile * static_cast<uint32_t>(kMoeBaseN);
            const uint32_t bufSlot = nTile % static_cast<uint32_t>(kMoeFifoDepth);

            __gm__ float *curGateSrc = GetProjectionTile(
                workspaceLayout.gateTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
            __gm__ float *curUpSrc = GetProjectionTile(
                workspaceLayout.upTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));

            WaitFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);

            gateSync.wait();
            LoadProjectionFromGM(curGateSrc, subBlockRowOffset, subBlockRows, gateTile);
            SetFlag<PIPE_MTE2, PIPE_V>(kVecGateLoadEvent);
            WaitFlag<PIPE_MTE2, PIPE_V>(kVecGateLoadEvent);
            SignalVecSubBlockFree(subBlockIdx, kGateFreeFlag0, kGateFreeFlag1);
            ApplySilu(gateTile, auxTile, recipTile);

            upSync.wait();
            LoadProjectionFromGM(curUpSrc, subBlockRowOffset, subBlockRows, auxTile);
            SetFlag<PIPE_MTE2, PIPE_V>(kVecUpLoadEvent);
            WaitFlag<PIPE_MTE2, PIPE_V>(kVecUpLoadEvent);
            SignalVecSubBlockFree(subBlockIdx, kUpFreeFlag0, kUpFreeFlag1);
            TMUL(gateTile, gateTile, auxTile);

            SetFlag<PIPE_V, PIPE_MTE3>(kVecStoreEvent);
            WaitFlag<PIPE_V, PIPE_MTE3>(kVecStoreEvent);
            StoreResult(gateTile, currentDst + colOffset, subBlockRowOffset, subBlockRows);
            SetFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);
        }

        WaitFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);
    }

#ifdef __DAV_C220_VEC__
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, kBlockEndFlag));
#endif
#endif

#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
#ifdef __DAV_C220_CUBE__
    // Each AIV subblock signals completion; drain all to avoid counter accumulation.
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(kMoeVecSubBlockCount); ++idx) {
        wait_flag_dev(kBlockEndFlag);
    }
#endif
#endif

    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ AICORE void moe_grouped_gemm_with_intermediates_custom(
    GM_ADDR x,
    GM_ADDR gate_weight_dn,
    GM_ADDR up_weight_dn,
    GM_ADDR out,
    GM_ADDR gate_proj,
    GM_ADDR up_proj,
    GM_ADDR expert_ids,
    GM_ADDR row_offsets,
    GM_ADDR valid_rows,
    GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    __gm__ bfloat16_t *xPtr = reinterpret_cast<__gm__ bfloat16_t *>(x);
    __gm__ bfloat16_t *gateWeightPtr = reinterpret_cast<__gm__ bfloat16_t *>(gate_weight_dn);
    __gm__ bfloat16_t *upWeightPtr = reinterpret_cast<__gm__ bfloat16_t *>(up_weight_dn);
    __gm__ float *outPtr = reinterpret_cast<__gm__ float *>(out);
    __gm__ float *gateProjPtr = reinterpret_cast<__gm__ float *>(gate_proj);
    __gm__ float *upProjPtr = reinterpret_cast<__gm__ float *>(up_proj);
    __gm__ int32_t *expertIdsPtr = reinterpret_cast<__gm__ int32_t *>(expert_ids);
    __gm__ int32_t *rowOffsetsPtr = reinterpret_cast<__gm__ int32_t *>(row_offsets);
    __gm__ int32_t *validRowsPtr = reinterpret_cast<__gm__ int32_t *>(valid_rows);

    const uint32_t workIdx = get_block_idx();
    const uint32_t blockDim = static_cast<uint32_t>(get_block_num());
    const uint32_t currentM = static_cast<uint32_t>(validRowsPtr[workIdx]);
    if (currentM == 0) {
        return;
    }

    auto workspaceLayout = GetMoeWorkspaceLayout(reinterpret_cast<__gm__ uint8_t *>(workspace), blockDim);
    const int commSlot = GetCommSlot(workIdx, blockDim, workspaceLayout.cvCommBuf);

    const uint32_t rowOffset = static_cast<uint32_t>(rowOffsetsPtr[workIdx]);
    __gm__ float *currentDst = outPtr + rowOffset * kMoeInterSize;
    __gm__ float *currentGateProjDst = gateProjPtr + rowOffset * kMoeInterSize;
    __gm__ float *currentUpProjDst = upProjPtr + rowOffset * kMoeInterSize;

#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
    using TileMatA =
        Tile<TileType::Mat, bfloat16_t, kMoeBaseM, kMoeBaseK * kMoeStepKa, BLayout::ColMajor, -1, -1,
             SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, kMoeBaseK * kMoeStepKb, kMoeBaseN, BLayout::RowMajor,
                          kMoeBaseK * kMoeStepKb, kMoeBaseN, SLayout::ColMajor>;
    using LeftTile = TileLeftCompact<bfloat16_t, kMoeBaseM, kMoeBaseK, -1, -1>;
    using RightTile = TileRightCompact<bfloat16_t, kMoeBaseK, kMoeBaseN, -1, -1>;
    using ResTile = TileAcc<float, kMoeBaseM, kMoeBaseN, -1, -1>;

    CrossCoreSync gateSync{kGateReadyFlag};
    CrossCoreSync upSync{kUpReadyFlag};

    for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
        const uint32_t colOffset = nTile * static_cast<uint32_t>(kMoeBaseN);
        const uint32_t bufSlot = nTile % static_cast<uint32_t>(kMoeFifoDepth);

        // Wait for VECTOR to free this buffer slot (skip for first kMoeFifoDepth tiles)
        if (nTile >= static_cast<uint32_t>(kMoeFifoDepth)) {
            WaitVecSubBlocksFree(kGateFreeFlag0, kGateFreeFlag1);
        }
        __gm__ float *curGateDst = GetProjectionTile(
            workspaceLayout.gateTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
        RunGroupedProjectionToGMWithOutput<TileMatA, TileMatB, LeftTile, RightTile, ResTile>(
            xPtr,
            gateWeightPtr,
            expertIdsPtr,
            rowOffsetsPtr,
            validRowsPtr,
            colOffset,
            curGateDst,
            currentGateProjDst + colOffset);
        gateSync.record();

        if (nTile >= static_cast<uint32_t>(kMoeFifoDepth)) {
            WaitVecSubBlocksFree(kUpFreeFlag0, kUpFreeFlag1);
        }
        __gm__ float *curUpDst = GetProjectionTile(
            workspaceLayout.upTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
        RunGroupedProjectionToGMWithOutput<TileMatA, TileMatB, LeftTile, RightTile, ResTile>(
            xPtr,
            upWeightPtr,
            expertIdsPtr,
            rowOffsetsPtr,
            validRowsPtr,
            colOffset,
            curUpDst,
            currentUpProjDst + colOffset);
        upSync.record();
    }

    // Drain: wait for VECTOR to finish with the last kMoeFifoDepth buffer slots
    for (uint32_t d = 0; d < static_cast<uint32_t>(kMoeFifoDepth); ++d) {
        WaitVecSubBlocksFree(kGateFreeFlag0, kGateFreeFlag1);
        WaitVecSubBlocksFree(kUpFreeFlag0, kUpFreeFlag1);
    }

#elif defined(__DAV_C220_VEC__) || defined(__DAV_VEC__)
    // Initialize VEC state explicitly. Running the mix kernel directly skips the auto-generated
    // wrapper path that would normally sanitize AIV state for matmul-based kernels.
    AscendC::SetAtomicNone();
    AscendC::SetVectorMask<uint64_t, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
    AscendC::SetMaskNorm();

    using VecTile = Tile<TileType::Vec, float, kMoeVecSubBlockRows, kMoeBaseN, BLayout::RowMajor, -1, -1>;

    CrossCoreSync gateSync{kGateReadyFlag};
    CrossCoreSync upSync{kUpReadyFlag};

    const uint32_t subBlockIdx = static_cast<uint32_t>(get_subblockid());
    const uint32_t subBlockRowOffset = subBlockIdx * static_cast<uint32_t>(kMoeVecSubBlockRows);
    uint32_t subBlockRows = 0;
    if (currentM > subBlockRowOffset) {
        const uint32_t rowsRemaining = currentM - subBlockRowOffset;
        subBlockRows = (rowsRemaining < static_cast<uint32_t>(kMoeVecSubBlockRows)) ?
                           rowsRemaining :
                           static_cast<uint32_t>(kMoeVecSubBlockRows);
    }

    VecTile gateTile(subBlockRows, kMoeBaseN);
    VecTile auxTile(subBlockRows, kMoeBaseN);
    VecTile recipTile(subBlockRows, kMoeBaseN);

    TASSIGN(gateTile, kGateUbOffset);
    TASSIGN(auxTile, kAuxUbOffset);
    TASSIGN(recipTile, kRecipUbOffset);

    if (subBlockRows == 0) {
        // Empty subblock: participate in sync protocol without doing real work
        for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
            gateSync.wait();
            SignalVecSubBlockFree(subBlockIdx, kGateFreeFlag0, kGateFreeFlag1);
            upSync.wait();
            SignalVecSubBlockFree(subBlockIdx, kUpFreeFlag0, kUpFreeFlag1);
        }
    } else {
        SetFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);

        for (uint32_t nTile = 0; nTile < static_cast<uint32_t>(kMoeNumNTiles); ++nTile) {
            const uint32_t colOffset = nTile * static_cast<uint32_t>(kMoeBaseN);
            const uint32_t bufSlot = nTile % static_cast<uint32_t>(kMoeFifoDepth);

            __gm__ float *curGateSrc = GetProjectionTile(
                workspaceLayout.gateTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));
            __gm__ float *curUpSrc = GetProjectionTile(
                workspaceLayout.upTiles, commSlot * kMoeFifoDepth + static_cast<int>(bufSlot));

            WaitFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);

            gateSync.wait();
            LoadProjectionFromGM(curGateSrc, subBlockRowOffset, subBlockRows, gateTile);
            SetFlag<PIPE_MTE2, PIPE_V>(kVecGateLoadEvent);
            WaitFlag<PIPE_MTE2, PIPE_V>(kVecGateLoadEvent);
            SignalVecSubBlockFree(subBlockIdx, kGateFreeFlag0, kGateFreeFlag1);
            ApplySilu(gateTile, auxTile, recipTile);

            upSync.wait();
            LoadProjectionFromGM(curUpSrc, subBlockRowOffset, subBlockRows, auxTile);
            SetFlag<PIPE_MTE2, PIPE_V>(kVecUpLoadEvent);
            WaitFlag<PIPE_MTE2, PIPE_V>(kVecUpLoadEvent);
            SignalVecSubBlockFree(subBlockIdx, kUpFreeFlag0, kUpFreeFlag1);
            TMUL(gateTile, gateTile, auxTile);

            SetFlag<PIPE_V, PIPE_MTE3>(kVecStoreEvent);
            WaitFlag<PIPE_V, PIPE_MTE3>(kVecStoreEvent);
            StoreResult(gateTile, currentDst + colOffset, subBlockRowOffset, subBlockRows);
            SetFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);
        }

        WaitFlag<PIPE_MTE3, PIPE_V>(kVecStoreEvent);
    }

#ifdef __DAV_C220_VEC__
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, kBlockEndFlag));
#endif
#endif

#if defined(__DAV_C220_CUBE__) || defined(__DAV_CUBE__)
#ifdef __DAV_C220_CUBE__
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(kMoeVecSubBlockCount); ++idx) {
        wait_flag_dev(kBlockEndFlag);
    }
#endif
#endif

    pipe_barrier(PIPE_ALL);
}

#endif
