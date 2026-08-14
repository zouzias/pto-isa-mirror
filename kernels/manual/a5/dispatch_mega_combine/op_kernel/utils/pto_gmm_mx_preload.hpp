/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP
#define DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP

#include "const_args.hpp"
#include "../gmm1_swiglu_cv_pipe.h"
#include "../gmm2_combine_cv_pipe.h"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

template <uint32_t L1_M_, uint32_t L1_N_, uint32_t L1_K_, uint32_t L0_M_, uint32_t L0_N_, uint32_t L0_K_,
          uint32_t SCALE_PREFETCH_K_, typename ElementA_ = float8_e4m3_t, typename ElementB_ = float8_e4m3_t,
          typename ScaleElement_ = float8_e8m0_t, typename ElementC_ = bfloat16_t>
class PtoGmmMxPreloadPipeline {
public:
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ScaleElement = ScaleElement_;
    using ElementC = ElementC_;

    static constexpr uint32_t L1_M = L1_M_;
    static constexpr uint32_t L1_N = L1_N_;
    static constexpr uint32_t L1_K = L1_K_;
    static constexpr uint32_t L0_M = L0_M_;
    static constexpr uint32_t L0_N = L0_N_;
    static constexpr uint32_t L0_K = L0_K_;
    static constexpr uint32_t SCALE_PREFETCH_K = SCALE_PREFETCH_K_;
    static constexpr uint32_t SCALE_GROUP_SIZE = kMegaMoeMxGroupSize;
    static constexpr uint32_t L0_SCALE_K = L0_K / SCALE_GROUP_SIZE;
    static constexpr uint32_t L1_SCALE_K = SCALE_PREFETCH_K / SCALE_GROUP_SIZE;
    static constexpr uint32_t BUFFER_NUM = 2U;
    static constexpr uint32_t L0_PINGPONG_BYTES = 32U * 1024U;
    static constexpr uint32_t L0_C_TILE_BYTES = L0_M * L0_N * sizeof(float);

    static constexpr uint32_t L1_A_DATA_BYTES = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1_B_DATA_BYTES = L1_K * L1_N * sizeof(ElementB);
    static constexpr uint32_t L1_A_SCALE_BYTES = L1_M * L1_SCALE_K * sizeof(ScaleElement);
    static constexpr uint32_t L1_B_SCALE_BYTES = L1_SCALE_K * L1_N * sizeof(ScaleElement);
    static constexpr uint32_t L1_TOTAL_BYTES =
        BUFFER_NUM * (L1_A_DATA_BYTES + L1_B_DATA_BYTES + L1_A_SCALE_BYTES + L1_B_SCALE_BYTES);

    static_assert(L1_M == L0_M && L1_N == L0_N, "MX GMM requires identical L1/L0 M/N tiles");
    static_assert(L1_K % L0_K == 0U, "L1 K must be divisible by L0 K");
    static_assert(SCALE_PREFETCH_K % L1_K == 0U, "scale prefetch K must contain complete L1 panels");
    static_assert(L0_K % 64U == 0U, "TMATMUL_MX reduction K must be a multiple of 64");
    static_assert(SCALE_PREFETCH_K % SCALE_GROUP_SIZE == 0U, "invalid MX scale prefetch shape");
    static_assert(L1_TOTAL_BYTES <= AtlasA5::L1_SIZE, "MX data/scale ping-pong exceeds L1");
    static_assert(L0_M * L0_K * sizeof(ElementA) <= L0_PINGPONG_BYTES, "MX A tile exceeds one L0A slot");
    static_assert(L0_K * L0_N * sizeof(ElementB) <= L0_PINGPONG_BYTES, "MX B tile exceeds one L0B slot");
    static_assert(BUFFER_NUM * L0_PINGPONG_BYTES <= AtlasA5::L0A_SIZE, "MX A ping-pong exceeds L0A");
    static_assert(BUFFER_NUM * L0_PINGPONG_BYTES <= AtlasA5::L0B_SIZE, "MX B ping-pong exceeds L0B");
    static_assert(L0_C_TILE_BYTES <= AtlasA5::L0C_SIZE, "MX accumulator exceeds L0C");
    static_assert(2U * L0_C_TILE_BYTES <= AtlasA5::L0C_SIZE, "MX pair accumulators exceed L0C");

    AICORE inline explicit PtoGmmMxPreloadPipeline(uint32_t l1BufAddrStart = 0U)
    {
        InitL1Offsets(l1BufAddrStart);
        InitEvents();
    }

    AICORE inline ~PtoGmmMxPreloadPipeline()
    {
        SynchronizeBlock();
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        WaitFlag<PIPE_FIX, PIPE_M>(kGateCReuseEvent);
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            WaitFlag<PIPE_M, PIPE_MTE1>(L0AEvent(slot));
            WaitFlag<PIPE_M, PIPE_MTE1>(L0BEvent(slot));
        }
    }

    AICORE inline void operator()(__gm__ ElementA *gmBlockA, __gm__ ScaleElement *gmBlockAScale,
                                  __gm__ ElementB *gmBlockB, __gm__ ScaleElement *gmBlockBScale,
                                  __gm__ ElementC *gmBlockC, uint32_t actualM, uint32_t actualN, uint32_t actualK,
                                  uint32_t aLeadingDim, uint32_t aScaleLeadingDim, uint32_t bLeadingDim,
                                  uint32_t bScaleLeadingDim, uint32_t cLeadingDim)
    {
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);

        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile cTile(actualM, actualN);
        pto::TASSIGN(cTile, 0U);

        const uint32_t kTileCount = actualK / L0_K;
        for (uint32_t kTileIdx = 0U; kTileIdx < kTileCount; ++kTileIdx) {
            const uint32_t kStart = kTileIdx * L0_K;
            const uint32_t dataPanelStart = kStart / L1_K * L1_K;
            const uint32_t dataSlot = (dataPanelStart / L1_K) & 1U;
            const uint32_t scalePanelStart = kStart / SCALE_PREFETCH_K * SCALE_PREFETCH_K;
            const uint32_t scaleSlot = (scalePanelStart / SCALE_PREFETCH_K) & 1U;

            if (kStart == dataPanelStart) {
                const uint32_t panelK = MinU32(L1_K, actualK - dataPanelStart);
                LoadDataPanel(dataSlot, gmBlockA + dataPanelStart, gmBlockB + dataPanelStart, actualM, actualN, panelK,
                              aLeadingDim, bLeadingDim);
            }
            if (kStart == scalePanelStart) {
                const uint32_t panelK = MinU32(SCALE_PREFETCH_K, actualK - scalePanelStart);
                LoadScalePanel(scaleSlot, gmBlockAScale + scalePanelStart / SCALE_GROUP_SIZE,
                               gmBlockBScale + scalePanelStart / SCALE_GROUP_SIZE, actualM, actualN,
                               panelK / SCALE_GROUP_SIZE, aScaleLeadingDim, bScaleLeadingDim);
            }

            const uint32_t l0Slot = kTileIdx & 1U;
            const uint32_t dataKOffset = kStart - dataPanelStart;
            const uint32_t scaleKOffset = (kStart - scalePanelStart) / SCALE_GROUP_SIZE;
            ExtractL0Tiles(dataSlot, scaleSlot, l0Slot, actualM, actualN, dataKOffset, scaleKOffset,
                           kStart + L0_K == MinU32(dataPanelStart + L1_K, actualK),
                           kStart + L0_K == MinU32(scalePanelStart + SCALE_PREFETCH_K, actualK));
            LaunchMatmul<true>(cTile, l0Slot, actualM, actualN, kTileIdx == 0U, kTileIdx + 1U == kTileCount);
        }

        StoreResult(cTile, gmBlockC, actualM, actualN, cLeadingDim);
    }

    AICORE inline void RunPairDirect(Gmm1SwigluCvPipe &cvPipe,
                                     __gm__ ElementA *gmBlockA, __gm__ ScaleElement *gmBlockAScale,
                                     __gm__ ElementB *gmBlockBX, __gm__ ScaleElement *gmBlockBScaleX,
                                     __gm__ ElementB *gmBlockBGate, __gm__ ScaleElement *gmBlockBScaleGate,
                                     uint32_t actualM, uint32_t actualPairN, uint32_t actualK, uint32_t aLeadingDim,
                                     uint32_t aScaleLeadingDim, uint32_t bLeadingDim, uint32_t bScaleLeadingDim)
    {
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        WaitFlag<PIPE_FIX, PIPE_M>(kGateCReuseEvent);

        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile xTile(actualM, actualPairN);
        AccTile gateTile(actualM, actualPairN);
        pto::TASSIGN(xTile, 0U);
        pto::TASSIGN(gateTile, L0_C_TILE_BYTES);

        ComputeResult<false>(xTile, gmBlockA, gmBlockAScale, gmBlockBX, gmBlockBScaleX, actualM, actualPairN, actualK,
                             aLeadingDim, aScaleLeadingDim, bLeadingDim, bScaleLeadingDim);
        ComputeResult<false>(gateTile, gmBlockA, gmBlockAScale, gmBlockBGate, gmBlockBScaleGate, actualM,
                             actualPairN, actualK, aLeadingDim, aScaleLeadingDim, bLeadingDim, bScaleLeadingDim);

        SetFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        SetFlag<PIPE_M, PIPE_FIX>(kGateStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kGateStoreReadyEvent);
        PushPairTileToAiv1(cvPipe, xTile, gateTile, actualM, actualPairN);
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        SetFlag<PIPE_FIX, PIPE_M>(kGateCReuseEvent);
    }

    AICORE inline void RunDirect(Gmm2CombineCvPipe &cvPipe,
                                 __gm__ ElementA *gmBlockA, __gm__ ScaleElement *gmBlockAScale,
                                 __gm__ ElementB *gmBlockB, __gm__ ScaleElement *gmBlockBScale, uint32_t actualM,
                                 uint32_t actualN, uint32_t actualK, uint32_t aLeadingDim,
                                 uint32_t aScaleLeadingDim, uint32_t bLeadingDim, uint32_t bScaleLeadingDim)
    {
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);

        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile cTile(actualM, actualN);
        pto::TASSIGN(cTile, 0U);
        ComputeResult<true>(cTile, gmBlockA, gmBlockAScale, gmBlockB, gmBlockBScale, actualM, actualN, actualK,
                            aLeadingDim, aScaleLeadingDim, bLeadingDim, bScaleLeadingDim);

        SetFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        Gmm2CombineProducerAllocate(cvPipe);
        Gmm2CombineCvTile cvTile(actualM, actualN);
        pto::TASSIGN(cvTile, Gmm2CombineSlotOffset(cvPipe.prod.tileIndex));
        pto::TMOV<pto::STPhase::Final, Gmm2CombineCvTile, AccTile,
                  pto::AccToVecMode::SingleModeVec1>(cvTile, cTile);
        pipe_barrier(PIPE_FIX);
        Gmm2CombineProducerRecord(cvPipe);
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
    }

    AICORE inline void SynchronizeBlock()
    {
        pipe_barrier(PIPE_ALL);
    }

private:
    static constexpr uint32_t kCReuseEvent = 0U;
    static constexpr uint32_t kGateCReuseEvent = 1U;
    static constexpr uint32_t kStoreReadyEvent = 0U;
    static constexpr uint32_t kGateStoreReadyEvent = 1U;
    static constexpr uint32_t kL0ReadyEvent = 0U;

    AICORE static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    AICORE static inline uint32_t ADataEvent(uint32_t slot)
    {
        return slot;
    }

    AICORE static inline uint32_t BDataEvent(uint32_t slot)
    {
        return BUFFER_NUM + slot;
    }

    AICORE static inline uint32_t AScaleEvent(uint32_t slot)
    {
        return 2U * BUFFER_NUM + slot;
    }

    AICORE static inline uint32_t BScaleEvent(uint32_t slot)
    {
        return 3U * BUFFER_NUM + slot;
    }

    AICORE static inline uint32_t L0AEvent(uint32_t slot)
    {
        return slot;
    }

    AICORE static inline uint32_t L0BEvent(uint32_t slot)
    {
        return BUFFER_NUM + slot;
    }

    template <pipe_t SrcPipe, pipe_t DstPipe>
    AICORE static inline void SetFlag(uint32_t eventId)
    {
        set_flag(SrcPipe, DstPipe, static_cast<event_t>(eventId));
    }

    template <pipe_t SrcPipe, pipe_t DstPipe>
    AICORE static inline void WaitFlag(uint32_t eventId)
    {
        wait_flag(SrcPipe, DstPipe, static_cast<event_t>(eventId));
    }

    AICORE inline void InitL1Offsets(uint32_t start)
    {
        uint64_t next = start;
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            l1ADataOffset_[slot] = next;
            next += L1_A_DATA_BYTES;
        }
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            l1BDataOffset_[slot] = next;
            next += L1_B_DATA_BYTES;
        }
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            l1AScaleOffset_[slot] = next;
            next += L1_A_SCALE_BYTES;
        }
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            l1BScaleOffset_[slot] = next;
            next += L1_B_SCALE_BYTES;
        }
    }

    AICORE inline void InitEvents()
    {
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        SetFlag<PIPE_FIX, PIPE_M>(kGateCReuseEvent);
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            SetFlag<PIPE_M, PIPE_MTE1>(L0AEvent(slot));
            SetFlag<PIPE_M, PIPE_MTE1>(L0BEvent(slot));
        }
    }

    template <bool EnableUnitFlag, typename AccTile>
    AICORE inline void ComputeResult(AccTile &cTile, __gm__ ElementA *gmBlockA,
                                     __gm__ ScaleElement *gmBlockAScale, __gm__ ElementB *gmBlockB,
                                     __gm__ ScaleElement *gmBlockBScale, uint32_t actualM, uint32_t actualN,
                                     uint32_t actualK, uint32_t aLeadingDim, uint32_t aScaleLeadingDim,
                                     uint32_t bLeadingDim, uint32_t bScaleLeadingDim)
    {
        const uint32_t kTileCount = actualK / L0_K;
        for (uint32_t kTileIdx = 0U; kTileIdx < kTileCount; ++kTileIdx) {
            const uint32_t kStart = kTileIdx * L0_K;
            const uint32_t dataPanelStart = kStart / L1_K * L1_K;
            const uint32_t dataSlot = (dataPanelStart / L1_K) & 1U;
            const uint32_t scalePanelStart = kStart / SCALE_PREFETCH_K * SCALE_PREFETCH_K;
            const uint32_t scaleSlot = (scalePanelStart / SCALE_PREFETCH_K) & 1U;

            if (kStart == dataPanelStart) {
                const uint32_t panelK = MinU32(L1_K, actualK - dataPanelStart);
                LoadDataPanel(dataSlot, gmBlockA + dataPanelStart, gmBlockB + dataPanelStart, actualM, actualN, panelK,
                              aLeadingDim, bLeadingDim);
            }
            if (kStart == scalePanelStart) {
                const uint32_t panelK = MinU32(SCALE_PREFETCH_K, actualK - scalePanelStart);
                LoadScalePanel(scaleSlot, gmBlockAScale + scalePanelStart / SCALE_GROUP_SIZE,
                               gmBlockBScale + scalePanelStart / SCALE_GROUP_SIZE, actualM, actualN,
                               panelK / SCALE_GROUP_SIZE, aScaleLeadingDim, bScaleLeadingDim);
            }

            const uint32_t l0Slot = kTileIdx & 1U;
            const uint32_t dataKOffset = kStart - dataPanelStart;
            const uint32_t scaleKOffset = (kStart - scalePanelStart) / SCALE_GROUP_SIZE;
            ExtractL0Tiles(dataSlot, scaleSlot, l0Slot, actualM, actualN, dataKOffset, scaleKOffset,
                           kStart + L0_K == MinU32(dataPanelStart + L1_K, actualK),
                           kStart + L0_K == MinU32(scalePanelStart + SCALE_PREFETCH_K, actualK));
            LaunchMatmul<EnableUnitFlag>(cTile, l0Slot, actualM, actualN, kTileIdx == 0U,
                                         kTileIdx + 1U == kTileCount);
        }
    }

    template <typename AccTile>
    AICORE inline void PushPairTileToAiv1(Gmm1SwigluCvPipe &cvPipe, AccTile &xTile,
                                          AccTile &gateTile, uint32_t actualM, uint32_t actualPairN)
    {
        Gmm1SwigluProducerAllocate(cvPipe);
        const uint64_t slotBase =
            kGmm1SwigluCvBufferOffset +
            static_cast<uint64_t>(cvPipe.prod.tileIndex % kGmm1SwigluCvFifoDepth) *
                kGmm1SwigluCvSlotBytes;
        Gmm1SwigluCvHalfTile xCvTile(actualM, actualPairN);
        Gmm1SwigluCvHalfTile gateCvTile(actualM, actualPairN);
        pto::TASSIGN(xCvTile, slotBase + kGmm1SwigluCvXOffset);
        pto::TASSIGN(gateCvTile, slotBase + kGmm1SwigluCvGateOffset);
        pto::TMOV<Gmm1SwigluCvHalfTile, AccTile,
                  pto::AccToVecMode::SingleModeVec1>(xCvTile, xTile);
        pto::TMOV<Gmm1SwigluCvHalfTile, AccTile,
                  pto::AccToVecMode::SingleModeVec1>(gateCvTile, gateTile);
        pipe_barrier(PIPE_FIX);
        Gmm1SwigluProducerRecord(cvPipe);
    }

    AICORE inline void LoadDataPanel(uint32_t slot, __gm__ ElementA *gmA, __gm__ ElementB *gmB, uint32_t actualM,
                                     uint32_t actualN, uint32_t panelK, uint32_t aLeadingDim, uint32_t bLeadingDim)
    {
        using APanel = pto::Tile<pto::TileType::Mat, ElementA, L1_M, L1_K, pto::BLayout::ColMajor, pto::DYNAMIC,
                                 pto::DYNAMIC, pto::SLayout::RowMajor>;
        using BPanel = pto::Tile<pto::TileType::Mat, ElementB, L1_K, L1_N, pto::BLayout::RowMajor, pto::DYNAMIC,
                                 pto::DYNAMIC, pto::SLayout::ColMajor>;
        using AShape = pto::TileShape2D<ElementA, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::ND>;
        using AStride = pto::BaseShape2D<ElementA, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::ND>;
        using AGlobal = pto::GlobalTensor<ElementA, AShape, AStride, pto::Layout::ND>;
        using BShape = pto::TileShape2D<ElementB, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::DN>;
        using BStride = pto::BaseShape2D<ElementB, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::DN>;
        using BGlobal = pto::GlobalTensor<ElementB, BShape, BStride, pto::Layout::DN>;

        APanel aPanel(actualM, panelK);
        BPanel bPanel(panelK, actualN);
        pto::TASSIGN(aPanel, l1ADataOffset_[slot]);
        pto::TASSIGN(bPanel, l1BDataOffset_[slot]);

        AShape aShape(actualM, panelK);
        AStride aStride(actualM, aLeadingDim);
        BShape bShape(panelK, actualN);
        BStride bStride(bLeadingDim, actualN);
        AGlobal aGlobal(gmA, aShape, aStride);
        BGlobal bGlobal(gmB, bShape, bStride);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
        pto::TLOAD(aPanel, aGlobal);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(ADataEvent(slot));
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
        pto::TLOAD(bPanel, bGlobal);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(BDataEvent(slot));
    }

    AICORE inline void LoadScalePanel(uint32_t slot, __gm__ ScaleElement *gmAScale, __gm__ ScaleElement *gmBScale,
                                      uint32_t actualM, uint32_t actualN, uint32_t panelScaleK,
                                      uint32_t aScaleLeadingDim, uint32_t bScaleLeadingDim)
    {
        using AScalePanel = pto::Tile<pto::TileType::Mat, ScaleElement, L1_M, L1_SCALE_K, pto::BLayout::RowMajor,
                                      pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor, 32>;
        using BScalePanel = pto::Tile<pto::TileType::Mat, ScaleElement, L1_SCALE_K, L1_N, pto::BLayout::ColMajor,
                                      pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::ColMajor, 32>;
        using AShape = pto::TileShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_A_ND>;
        using AStride = pto::BaseShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_A_ND>;
        using AGlobal = pto::GlobalTensor<ScaleElement, AShape, AStride, pto::Layout::MX_A_ND>;
        using BShape = pto::TileShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_B_DN>;
        using BStride = pto::BaseShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_B_DN>;
        using BGlobal = pto::GlobalTensor<ScaleElement, BShape, BStride, pto::Layout::MX_B_DN>;

        AScalePanel aPanel(actualM, panelScaleK);
        BScalePanel bPanel(panelScaleK, actualN);
        pto::TASSIGN(aPanel, l1AScaleOffset_[slot]);
        pto::TASSIGN(bPanel, l1BScaleOffset_[slot]);

        AShape aShape(actualM, panelScaleK);
        AStride aStride(actualM, aScaleLeadingDim);
        BShape bShape(panelScaleK, actualN);
        BStride bStride(bScaleLeadingDim, actualN);
        AGlobal aGlobal(gmAScale, aShape, aStride);
        BGlobal bGlobal(gmBScale, bShape, bStride);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
        pto::TLOAD(aPanel, aGlobal);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(AScaleEvent(slot));
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
        pto::TLOAD(bPanel, bGlobal);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(BScaleEvent(slot));
    }

    AICORE inline void ExtractL0Tiles(uint32_t dataSlot, uint32_t scaleSlot, uint32_t l0Slot, uint32_t actualM,
                                      uint32_t actualN, uint32_t dataKOffset, uint32_t scaleKOffset,
                                      bool releaseDataPanel, bool releaseScalePanel)
    {
        using APanel = pto::Tile<pto::TileType::Mat, ElementA, L1_M, L1_K, pto::BLayout::ColMajor, pto::DYNAMIC,
                                 pto::DYNAMIC, pto::SLayout::RowMajor>;
        using BPanel = pto::Tile<pto::TileType::Mat, ElementB, L1_K, L1_N, pto::BLayout::RowMajor, pto::DYNAMIC,
                                 pto::DYNAMIC, pto::SLayout::ColMajor>;
        using AScalePanel = pto::Tile<pto::TileType::Mat, ScaleElement, L1_M, L1_SCALE_K, pto::BLayout::RowMajor,
                                      pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor, 32>;
        using BScalePanel = pto::Tile<pto::TileType::Mat, ScaleElement, L1_SCALE_K, L1_N, pto::BLayout::ColMajor,
                                      pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::ColMajor, 32>;
        using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, L0_K>;
        using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, L0_K, pto::DYNAMIC>;
        using LeftScaleTile = pto::TileLeftScaleCompact<ScaleElement, L0_M, L0_SCALE_K, pto::DYNAMIC, L0_SCALE_K>;
        using RightScaleTile = pto::TileRightScaleCompact<ScaleElement, L0_SCALE_K, L0_N, L0_SCALE_K, pto::DYNAMIC>;

        APanel aPanel(actualM, MinU32(L1_K, dataKOffset + L0_K));
        BPanel bPanel(MinU32(L1_K, dataKOffset + L0_K), actualN);
        AScalePanel aScalePanel(actualM, MinU32(L1_SCALE_K, scaleKOffset + L0_SCALE_K));
        BScalePanel bScalePanel(MinU32(L1_SCALE_K, scaleKOffset + L0_SCALE_K), actualN);
        pto::TASSIGN(aPanel, l1ADataOffset_[dataSlot]);
        pto::TASSIGN(bPanel, l1BDataOffset_[dataSlot]);
        pto::TASSIGN(aScalePanel, l1AScaleOffset_[scaleSlot]);
        pto::TASSIGN(bScalePanel, l1BScaleOffset_[scaleSlot]);

        LeftTile aTile(actualM);
        RightTile bTile(actualN);
        LeftScaleTile aScaleTile(actualM);
        RightScaleTile bScaleTile(actualN);
        pto::TASSIGN(aTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(bTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(aScaleTile, pto::GetScaleAddr(aTile.data()));
        pto::TASSIGN(bScaleTile, pto::GetScaleAddr(bTile.data()));

        WaitFlag<PIPE_M, PIPE_MTE1>(L0AEvent(l0Slot));
        WaitFlag<PIPE_M, PIPE_MTE1>(L0BEvent(l0Slot));
        if (dataKOffset == 0U) {
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(ADataEvent(dataSlot));
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(BDataEvent(dataSlot));
        }
        if (scaleKOffset == 0U) {
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(AScaleEvent(scaleSlot));
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(BScaleEvent(scaleSlot));
        }

        pto::TEXTRACT(aTile, aPanel, 0U, dataKOffset);
        pto::TEXTRACT(bTile, bPanel, dataKOffset, 0U);
        pto::TEXTRACT(aScaleTile, aScalePanel, 0U, scaleKOffset);
        pto::TEXTRACT(bScaleTile, bScalePanel, scaleKOffset, 0U);

        if (releaseDataPanel) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(dataSlot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(dataSlot));
        }
        if (releaseScalePanel) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(scaleSlot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(scaleSlot));
        }
        SetFlag<PIPE_MTE1, PIPE_M>(kL0ReadyEvent);
    }

    template <bool EnableUnitFlag, typename AccTile>
    AICORE inline void LaunchMatmul(AccTile &cTile, uint32_t l0Slot, uint32_t actualM, uint32_t actualN, bool firstK,
                                    bool lastK)
    {
        using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, L0_K>;
        using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, L0_K, pto::DYNAMIC>;
        using LeftScaleTile = pto::TileLeftScaleCompact<ScaleElement, L0_M, L0_SCALE_K, pto::DYNAMIC, L0_SCALE_K>;
        using RightScaleTile = pto::TileRightScaleCompact<ScaleElement, L0_SCALE_K, L0_N, L0_SCALE_K, pto::DYNAMIC>;

        LeftTile aTile(actualM);
        RightTile bTile(actualN);
        LeftScaleTile aScaleTile(actualM);
        RightScaleTile bScaleTile(actualN);
        pto::TASSIGN(aTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(bTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(aScaleTile, pto::GetScaleAddr(aTile.data()));
        pto::TASSIGN(bScaleTile, pto::GetScaleAddr(bTile.data()));

        WaitFlag<PIPE_MTE1, PIPE_M>(kL0ReadyEvent);
        if constexpr (EnableUnitFlag) {
            if (firstK) {
                if (lastK) {
                    pto::TMATMUL_MX<pto::AccPhase::Final>(cTile, aTile, aScaleTile, bTile, bScaleTile);
                } else {
                    pto::TMATMUL_MX<pto::AccPhase::Partial>(cTile, aTile, aScaleTile, bTile, bScaleTile);
                }
            } else if (lastK) {
                pto::TMATMUL_MX<pto::AccPhase::Final>(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            } else {
                pto::TMATMUL_MX<pto::AccPhase::Partial>(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            }
        } else if (firstK) {
            pto::TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
        } else {
            pto::TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
        }
        SetFlag<PIPE_M, PIPE_MTE1>(L0AEvent(l0Slot));
        SetFlag<PIPE_M, PIPE_MTE1>(L0BEvent(l0Slot));
    }

    template <typename AccTile>
    AICORE inline void StoreResult(AccTile &cTile, __gm__ ElementC *gmC, uint32_t actualM, uint32_t actualN,
                                   uint32_t cLeadingDim)
    {
        using DstShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
        using DstStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using DstGlobal = pto::GlobalTensor<ElementC, DstShape, DstStride, pto::Layout::ND>;

        DstShape dstShape(actualM, actualN);
        DstStride dstStride(static_cast<uint64_t>(actualM) * cLeadingDim, static_cast<uint64_t>(actualM) * cLeadingDim,
                            static_cast<uint64_t>(actualM) * cLeadingDim, cLeadingDim);
        DstGlobal dstGlobal(gmC, dstShape, dstStride);
        SetFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        pto::TSTORE<pto::STPhase::Final>(dstGlobal, cTile);
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
    }

    uint64_t l1ADataOffset_[BUFFER_NUM] = {0U, 0U};
    uint64_t l1BDataOffset_[BUFFER_NUM] = {0U, 0U};
    uint64_t l1AScaleOffset_[BUFFER_NUM] = {0U, 0U};
    uint64_t l1BScaleOffset_[BUFFER_NUM] = {0U, 0U};
};

#endif // DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP
