/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP
#define PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP

#include "moe_pto_utils.hpp"
#include "dispatch_policy_custom.hpp"
#include "pto_mmad_ops.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

namespace pto_ext::Gemm::Block {
template <pto_ext::PtoHardEvent event>
__aicore__ inline void SyncFlagFunc(int32_t eventID)
{
    pto_ext::PtoSetFlag<event>(eventID);
    pto_ext::PtoWaitFlag<event>(eventID);
}

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_, class L1TileShape_, class L0TileShape_,
          class AType_, class BType_, class CType_, class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<MmadAtlasA5PreloadAsyncFixpipe<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                                ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
                 L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA5PreloadAsyncFixpipe<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_,
                                                          L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using MatmulShell = detail::MatmulShell<ArchTag, TileCopy_, AType_, BType_, CType_>;
    using CopyL1ToFPTraits = typename MatmulShell::CopyL1ToFPTraits;
    using CopyL1ToL0ATraits = typename MatmulShell::CopyL1ToL0ATraits;
    using CopyL1ToL0BTraits = typename MatmulShell::CopyL1ToL0BTraits;
    using ElementAccumulator = typename MatmulShell::ElementAccumulator;
    using CopyL0CToGmTraits = typename MatmulShell::CopyL0CToGmTraits;
    using LayoutAInL1 = typename MatmulShell::LayoutAInL1;
    using LayoutBInL1 = typename MatmulShell::LayoutBInL1;
    using LayoutAInL0 = typename MatmulShell::LayoutAInL0;
    using LayoutBInL0 = typename MatmulShell::LayoutBInL0;
    using LayoutCInL0 = layout::Zn;

    using L1AAlignHelper = typename MatmulShell::L1AAlignHelper;
    using L1BAlignHelper = typename MatmulShell::L1BAlignHelper;

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L1S_TILE_SIZE = L1TileShape::N * sizeof(int64_t);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::ND>, "LayoutC only supports ND.");

    // Check L1TileShape
    static_assert((std::is_same_v<ElementA, int8_t> ?
                       (L1A_TILE_SIZE + L1B_TILE_SIZE + L1S_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE :
                       (L1A_TILE_SIZE + L1B_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE),
                  "L1TileShape exceeding the L1 space for the given data type");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
                  "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    __forceinline__ __aicore__ static LayoutAInL1 MakeL1ALayout()
    {
        return LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
    }

    __forceinline__ __aicore__ static LayoutBInL1 MakeL1BLayout()
    {
        return LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
    }

    __forceinline__ __aicore__ BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0,
                                         uint32_t FpAddrStart = 0)
    {
        syncGroupIdx = 0;
        InitL1(resource, l1BufAddrStart);
        InitFpBuf(resource, FpAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
    }

    __forceinline__ __aicore__ ~BlockMmad()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1AEventList[i]);
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::FIX_M>(l0CEventList[i]);
        }
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::FIX_MTE2>(0);
        }
    }

    __forceinline__ __aicore__ void operator()(__gm__ ElementA *gmBlockAPtr, LayoutA const &layoutA,
                                               __gm__ ElementB *gmBlockBPtr, LayoutB const &layoutB,
                                               __gm__ ElementC *gmBlockCPtr, LayoutC const &layoutC,
                                               __gm__ uint64_t *gmBlockSPtr, layout::VectorLayout const &layoutScale,
                                               PtoShape3D const &actualShape, int32_t syncLoopIdx = -1,
                                               int32_t flag = 0)
    {
        uint32_t actualM = static_cast<uint32_t>(actualShape.shape[0]);
        uint32_t actualN = static_cast<uint32_t>(actualShape.shape[1]);
        uint32_t actualK = static_cast<uint32_t>(actualShape.shape[2]);
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualK);
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);
        uint32_t startTileIdx = GetStartTileIdx(kTileCount);

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = GetKTileIdx(startTileIdx, kLoopIdx, kTileCount);
            uint32_t kActual = (kTileIdx < kTileCount - 1) ? L1TileShape::K : (actualK - kTileIdx * L1TileShape::K);
            LoadL1Tiles(gmBlockAPtr, layoutA, gmBlockBPtr, layoutB, actualM, actualN, kTileIdx, kActual);
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }
            RecordPreloadParams(layoutC, layoutScale, actualShape, gmBlockCPtr, gmBlockSPtr, syncLoopIdx, flag, mRound,
                                nRound, kActual, kLoopIdx, kTileCount);
            AdvancePreloadState();
        }
    }

    __forceinline__ __aicore__ void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

    __forceinline__ __aicore__ void Finalize(int32_t target, int32_t flag = 0)
    {
        for (; syncGroupIdx <= target; syncGroupIdx++) {
            int32_t flagId = syncGroupIdx / 15 + flag;
            pto_ext::PtoCrossCoreRecord<pto::SyncOpType::TSTORE_C2GM, pto::SyncOpType::TLOAD>(flagId);
        }
    }

private:
    struct L1TileMmadParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        bool isKLoopFirst;
        bool isKLoopLast;
        __gm__ ElementC *gmBlockC;
        __gm__ uint64_t *gmBlockS;
        LayoutC layoutCInGm;
        layout::VectorLayout layoutScale;
        int32_t syncLoopIdx;
        int32_t flag;
        __forceinline__ __aicore__ L1TileMmadParams() = default;
    };

    __forceinline__ __aicore__ uint32_t GetStartTileIdx(uint32_t kTileCount)
    {
        if constexpr (ENABLE_SHUFFLE_K) {
            return pto_ext::PtoAicLogicalIdx() % kTileCount;
        }
        return 0;
    }

    __forceinline__ __aicore__ uint32_t GetKTileIdx(uint32_t startTileIdx, uint32_t kLoopIdx, uint32_t kTileCount)
    {
        uint32_t tileIdx = startTileIdx + kLoopIdx;
        return (tileIdx < kTileCount) ? tileIdx : (tileIdx - kTileCount);
    }

    __forceinline__ __aicore__ void LoadL1Tiles(__gm__ ElementA *gmBlockAPtr, LayoutA const &layoutA,
                                                __gm__ ElementB *gmBlockBPtr, LayoutB const &layoutB, uint32_t actualM,
                                                uint32_t actualN, uint32_t kTileIdx, uint32_t kActual)
    {
        auto gmTileAOffset = PtoCoord2D(0, kTileIdx * L1TileShape::K);
        auto gmTileBOffset = PtoCoord2D(kTileIdx * L1TileShape::K, 0);
        __gm__ ElementA *gmTileA = gmBlockAPtr + layoutA.GetOffset(gmTileAOffset);
        __gm__ ElementB *gmTileB = gmBlockBPtr + layoutB.GetOffset(gmTileBOffset);

        pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        auto layoutTileA = layoutA.GetTileLayout(PtoCoord2D(actualM, kActual));
        detail::PtoLoadNdGmToNzL1<ElementA, L1TileShape::M, L1TileShape::K>(l1AOffsetList[l1ListId], gmTileA,
                                                                            layoutTileA);
        pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileB = layoutB.GetTileLayout(PtoCoord2D(kActual, actualN));
        detail::PtoLoadNzGmToNzL1<ElementB, L1TileShape::K, L1TileShape::N>(l1BOffsetList[l1ListId], gmTileB,
                                                                            MakeL1BLayout(), layoutTileB);
        pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
    }

    __forceinline__ __aicore__ void RecordPreloadParams(LayoutC const &layoutC, layout::VectorLayout const &layoutScale,
                                                        PtoShape3D const &actualShape, __gm__ ElementC *gmBlockCPtr,
                                                        __gm__ uint64_t *gmBlockSPtr, int32_t syncLoopIdx, int32_t flag,
                                                        uint32_t mRound, uint32_t nRound, uint32_t kActual,
                                                        uint32_t kLoopIdx, uint32_t kTileCount)
    {
        uint32_t preloadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES) ?
                                       (l1TileMmadParamsId + preloadCount) :
                                       (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
        auto &params = l1TileMmadParamsList[preloadParamsId];
        params.l1ListId = l1ListId;
        params.mRound = mRound;
        params.nRound = nRound;
        params.kActual = kActual;
        params.isKLoopFirst = (kLoopIdx == 0);
        params.isKLoopLast = (kLoopIdx == kTileCount - 1);
        params.flag = flag;
        if (params.isKLoopLast) {
            params.gmBlockC = gmBlockCPtr;
            params.gmBlockS = gmBlockSPtr;
            params.layoutCInGm = layoutC.GetTileLayout(PtoShape2D(actualShape.shape[0], actualShape.shape[1]));
            params.layoutScale = layoutScale;
            params.syncLoopIdx = syncLoopIdx;
        }
    }

    __forceinline__ __aicore__ void AdvancePreloadState()
    {
        if (preloadCount < PRELOAD_STAGES) {
            ++preloadCount;
        } else {
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
        }
        l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
    }

    __forceinline__ __aicore__ void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;

        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1AOffsetList[i] = resource.l1Buf.GetBufferAddrByByte(l1AOffset + L1A_TILE_SIZE * i);
            l1BOffsetList[i] = resource.l1Buf.GetBufferAddrByByte(l1BOffset + L1B_TILE_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1AEventList[i]);
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        uint32_t l1SOffset = l1BOffset + L1B_TILE_SIZE * L1_STAGES;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            l1SBaseOffset = resource.l1Buf.GetBufferAddrByByte(l1SOffset);
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::FIX_MTE2>(0);
        }
    }

    __forceinline__ __aicore__ void InitFpBuf(Arch::Resource<ArchTag> &resource, uint32_t FpAddrStart)
    {
        uint32_t FpOffset = FpAddrStart;
        fixpipeBaseOffset = resource.fpBuf.GetBufferAddrByByte(FpOffset);
    }

    __forceinline__ __aicore__ void InitL0A(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0AOffsetList[i] = resource.l0ABuf.GetBufferAddrByByte(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    __forceinline__ __aicore__ void InitL0B(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BOffsetList[i] = resource.l0BBuf.GetBufferAddrByByte(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    __forceinline__ __aicore__ void InitL0C(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0COffsetList[i] = resource.l0CBuf.GetBufferAddrByByte(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    __forceinline__ __aicore__ void MoveL1ATileToL0(const L1TileMmadParams &params, uint32_t mPartIdx,
                                                    uint32_t kPartIdx, uint32_t mPartLoop, uint32_t kPartLoop,
                                                    uint32_t mPartActual, uint32_t kPartActual,
                                                    const LayoutAInL0 &layoutAInL0)
    {
        PtoCoord2D l1AOffset(mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K);
        const uint64_t l1AOffsetBytes = l1AOffsetList[params.l1ListId] +
                                        static_cast<uint64_t>(MakeL1ALayout().GetOffset(l1AOffset)) * sizeof(ElementA);
        const uint64_t l0AStagingOffsetBytes =
            l0AOffsetList[l0AListId] +
            static_cast<uint64_t>(layoutAInL0.GetOffset(PtoCoord2D(0, 0))) * sizeof(ElementA);

        pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::M_MTE1>(l0AEventList[l0AListId]);
        if ((mPartIdx == 0) && (kPartIdx == 0)) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
        }
        detail::PtoMoveL1ToL0A<ElementA, L0TileShape>(l0AStagingOffsetBytes, l1AOffsetBytes, mPartActual, kPartActual);
        if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
        }
    }

    __forceinline__ __aicore__ void MoveL1BTileToL0(const L1TileMmadParams &params, uint32_t kPartIdx,
                                                    uint32_t nPartIdx, uint32_t kPartLoop, uint32_t nPartLoop,
                                                    uint32_t kPartActual, uint32_t nPartActual,
                                                    const LayoutBInL0 &layoutBInL0)
    {
        PtoCoord2D l1BOffset(kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N);
        const uint64_t l1BOffsetBytes = l1BOffsetList[params.l1ListId] +
                                        static_cast<uint64_t>(MakeL1BLayout().GetOffset(l1BOffset)) * sizeof(ElementB);
        const uint64_t l0BStagingOffsetBytes =
            l0BOffsetList[l0BListId] +
            static_cast<uint64_t>(layoutBInL0.GetOffset(PtoCoord2D(0, 0))) * sizeof(ElementB);

        pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if ((kPartIdx == 0) && (nPartIdx == 0)) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
        }
        detail::PtoMoveL1ToL0B<ElementB, L0TileShape>(l0BStagingOffsetBytes, l1BOffsetBytes, kPartActual, nPartActual);
        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
        }
    }

    __forceinline__ __aicore__ uint8_t GetUnitFlag(const L1TileMmadParams &params, uint32_t mPartIdx, uint32_t kPartIdx,
                                                   uint32_t nPartIdx, uint32_t mPartLoop, uint32_t kPartLoop,
                                                   uint32_t nPartLoop)
    {
        uint8_t unitFlag = 0b00;
        if constexpr (ENABLE_UNIT_FLAG) {
            if (params.isKLoopLast && (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) &&
                (nPartIdx == nPartLoop - 1)) {
                unitFlag = 0b11;
            } else {
                unitFlag = 0b10;
            }
        }
        return unitFlag;
    }

    __forceinline__ __aicore__ void StoreFinalAccumulator(const L1TileMmadParams &params)
    {
        auto layoutCInGm = params.layoutCInGm;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::FIX_MTE2>(0);
            detail::StagePerChannelScale<L1TileShape::N>(l1SBaseOffset, fixpipeBaseOffset, params.gmBlockS,
                                                         params.layoutScale, layoutCInGm.shape(1));
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_FIX>(0);
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_FIX>(0);
            pto_ext::PtoPipeBarrier<PIPE_FIX>();
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::M_FIX>(l0CEventList[l0CListId]);
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::M_FIX>(l0CEventList[l0CListId]);
            detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                params.gmBlockC, l0COffsetList[l0CListId], fixpipeBaseOffset, layoutCInGm);
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::FIX_M>(l0CEventList[l0CListId]);
        } else {
            detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                params.gmBlockC, l0COffsetList[l0CListId], fixpipeBaseOffset, layoutCInGm, 0b11);
        }
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::FIX_MTE2>(0);
        }
    }

    __forceinline__ __aicore__ void RunNPartLoop(const L1TileMmadParams &params, const LayoutCInL0 &layoutCInL0,
                                                 const LayoutAInL0 &layoutAInL0, uint32_t mPartIdx, uint32_t kPartIdx,
                                                 uint32_t mPartLoop, uint32_t kPartLoop, uint32_t nPartLoop,
                                                 uint32_t mPartActual, uint32_t kPartActual)
    {
        for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
            uint32_t nPartActual =
                (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (params.nRound - nPartIdx * L0TileShape::N);
            auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
            MoveL1BTileToL0(params, kPartIdx, nPartIdx, kPartLoop, nPartLoop, kPartActual, nPartActual, layoutBInL0);
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::MTE1_M>(EVENT_ID0);
            PtoCoord2D l0COffset(mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N);
            pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::MTE1_M>(EVENT_ID0);
            bool initC = (params.isKLoopFirst && (kPartIdx == 0));
            uint8_t unitFlag = GetUnitFlag(params, mPartIdx, kPartIdx, nPartIdx, mPartLoop, kPartLoop, nPartLoop);
            const uint64_t l0AOffsetBytes =
                l0AOffsetList[l0AListId] +
                static_cast<uint64_t>(layoutAInL0.GetOffset(PtoCoord2D(0, 0))) * sizeof(ElementA);
            const uint64_t l0BOffsetBytes =
                l0BOffsetList[l0BListId] +
                static_cast<uint64_t>(layoutBInL0.GetOffset(PtoCoord2D(0, 0))) * sizeof(ElementB);
            const uint64_t l0COffsetBytes =
                l0COffsetList[l0CListId] +
                static_cast<uint64_t>(layoutCInL0.GetOffset(l0COffset)) * sizeof(ElementAccumulator);
            detail::PtoTileMmad<ElementAccumulator, ElementA, ElementB, L0TileShape>(
                l0COffsetBytes, l0AOffsetBytes, l0BOffsetBytes, mPartActual, nPartActual, kPartActual, initC, unitFlag);
            pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::M_MTE1>(l0BEventList[l0BListId]);
            l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
        }
    }

    __forceinline__ __aicore__ void L1TileMmad(L1TileMmadParams const &params)
    {
        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(params.mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(params.nRound);
        uint32_t kPartLoop = CeilDiv<L0TileShape::K>(params.kActual);
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(PtoCoord2D(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                pto_ext::PtoWaitFlag<pto_ext::PtoHardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual =
                (mPartIdx < mPartLoop - 1) ? L0TileShape::M : (params.mRound - mPartIdx * L0TileShape::M);

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual =
                    (kPartIdx < kPartLoop - 1) ? L0TileShape::K : (params.kActual - kPartIdx * L0TileShape::K);

                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                MoveL1ATileToL0(params, mPartIdx, kPartIdx, mPartLoop, kPartLoop, mPartActual, kPartActual,
                                layoutAInL0);

                RunNPartLoop(params, layoutCInL0, layoutAInL0, mPartIdx, kPartIdx, mPartLoop, kPartLoop, nPartLoop,
                             mPartActual, kPartActual);
                pto_ext::PtoSetFlag<pto_ext::PtoHardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            StoreFinalAccumulator(params);
#ifdef __TILE_SYNC__
            if (params.flag > 0) {
                int32_t flagId = params.flag + params.syncLoopIdx / 8;
                pto_ext::PtoCrossCoreRecord<pto::SyncOpType::TSTORE_C2GM, pto::SyncOpType::TLOAD>(flagId);
            }
#else
            Finalize(params.syncLoopIdx, params.flag);
#endif
        }
    }

    uint64_t fixpipeBaseOffset{0};

    uint64_t l1AOffsetList[L1_STAGES];
    uint64_t l1BOffsetList[L1_STAGES];
    uint64_t l1SBaseOffset{0};
    int32_t syncGroupIdx;
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    uint64_t l0AOffsetList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    uint64_t l0BOffsetList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    uint64_t l0COffsetList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    TileMmad tileMmad;
};

} // namespace pto_ext::Gemm::Block

#endif // PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP
