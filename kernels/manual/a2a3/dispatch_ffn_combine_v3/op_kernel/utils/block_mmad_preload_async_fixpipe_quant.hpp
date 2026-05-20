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

#include "dispatch_policy_custom.hpp"
#include "pto_global_view.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

namespace pto_ext::Gemm::Block {
namespace detail {

using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoGlobalNd;

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_DEVICE void LaunchPtoMatmul(TileAcc &cTile, TileLeft &aTile, TileRight &bTile, bool initC,
                                           uint8_t unitFlag)
{
    const bool isFinal = (unitFlag == 0b11);
    const bool isPartial = (unitFlag == 0b10);

    if (initC) {
        if (isFinal) {
            pto::TMATMUL<pto::AccPhase::Final>(cTile, aTile, bTile);
        } else if (isPartial) {
            pto::TMATMUL<pto::AccPhase::Partial>(cTile, aTile, bTile);
        } else {
            pto::TMATMUL(cTile, aTile, bTile);
        }
    } else {
        if (isFinal) {
            pto::TMATMUL_ACC<pto::AccPhase::Final>(cTile, aTile, bTile);
        } else if (isPartial) {
            pto::TMATMUL_ACC<pto::AccPhase::Partial>(cTile, aTile, bTile);
        } else {
            pto::TMATMUL_ACC(cTile, aTile, bTile);
        }
    }
}

template <typename ElementAccumulator, typename ElementA, typename ElementB, class L0TileShape>
PTO_DEVICE void PtoTileMmad(uint64_t l0COffset,
                            uint64_t l0AOffset,
                            uint64_t l0BOffset,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            bool initC = true,
                            uint8_t unitFlag = 0)
{
    using LeftTile = pto::TileLeft<ElementA, L0TileShape::M, L0TileShape::K, pto::DYNAMIC, pto::DYNAMIC>;
    using RightTile = pto::TileRight<ElementB, L0TileShape::K, L0TileShape::N, pto::DYNAMIC, pto::DYNAMIC>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, L0TileShape::M, L0TileShape::N, pto::DYNAMIC,
                                        pto::DYNAMIC>;

    LeftTile aTile(m, k);
    RightTile bTile(k, n);
    AccTile cTile(m, n);

    pto::TASSIGN(aTile, l0AOffset);
    pto::TASSIGN(bTile, l0BOffset);
    pto::TASSIGN(cTile, l0COffset);

    LaunchPtoMatmul(cTile, aTile, bTile, initC, unitFlag);

    constexpr uint32_t kPipeBarrierThreshold = 10;
    constexpr uint32_t kFractalEdge = 16;
    if ((m / kFractalEdge) * (n / kFractalEdge) < kPipeBarrierThreshold) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

template <uint32_t TileElems = FLAGSTRIDE>
PTO_DEVICE void PtoLoadSoftFlagL1(uint64_t dstOffset,
                                  __gm__ int32_t *src,
                                  uint32_t elemNum)
{
    using FlagTile = pto::Tile<pto::TileType::Mat, int32_t, 1, TileElems, pto::BLayout::RowMajor, 1,
                               pto::DYNAMIC, pto::SLayout::NoneBox>;
    using FlagShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using FlagStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using FlagGlobal = pto::GlobalTensor<int32_t, FlagShape, FlagStride, pto::Layout::ND>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        FlagShape shape(cur);
        FlagStride stride(cur, cur, cur, cur);
        FlagGlobal srcGlobal(src + offset, shape, stride);
        FlagTile tile(cur);
        pto::TASSIGN(tile, dstOffset + static_cast<uint64_t>(offset) * sizeof(int32_t));
        pto::TLOAD(tile, srcGlobal);
    }
}

template <uint32_t TileElems = FLAGSTRIDE>
PTO_DEVICE void PtoStoreSoftFlagL1(__gm__ int32_t *dst,
                                   uint64_t srcOffset,
                                   uint32_t elemNum)
{
    using FlagTile = pto::Tile<pto::TileType::Mat, int32_t, 1, TileElems, pto::BLayout::RowMajor, 1,
                               pto::DYNAMIC, pto::SLayout::NoneBox>;
    using FlagShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using FlagStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using FlagGlobal = pto::GlobalTensor<int32_t, FlagShape, FlagStride, pto::Layout::ND>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        FlagShape shape(cur);
        FlagStride stride(cur, cur, cur, cur);
        FlagGlobal dstGlobal(dst + offset, shape, stride);
        FlagTile tile(cur);
        pto::TASSIGN(tile, srcOffset + static_cast<uint64_t>(offset) * sizeof(int32_t));
        pto::TSTORE(dstGlobal, tile);
    }
}

template <typename Element, class L0TileShape>
PTO_DEVICE void PtoMoveL1ToL0A(uint64_t dstL0Offset,
                               uint64_t srcL1Offset,
                               uint32_t m,
                               uint32_t k)
{
    using SrcTile = pto::Tile<pto::TileType::Mat, Element, L0TileShape::M, L0TileShape::K, pto::BLayout::RowMajor,
                              pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::ColMajor>;
    using DstTile = pto::TileLeft<Element, L0TileShape::M, L0TileShape::K, pto::DYNAMIC, pto::DYNAMIC>;

    SrcTile srcTile(m, k);
    DstTile dstTile(m, k);
    pto::TASSIGN(srcTile, srcL1Offset);
    pto::TASSIGN(dstTile, dstL0Offset);
    pto::TMOV(dstTile, srcTile);
}

template <typename Element, class L0TileShape>
PTO_DEVICE void PtoMoveL1ToL0B(uint64_t dstL0Offset,
                               uint64_t srcL1Offset,
                               uint32_t k,
                               uint32_t n)
{
    using SrcTile = pto::Tile<pto::TileType::Mat, Element, L0TileShape::K, L0TileShape::N, pto::BLayout::RowMajor,
                              pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::ColMajor>;
    using DstTile = pto::TileRight<Element, L0TileShape::K, L0TileShape::N, pto::DYNAMIC, pto::DYNAMIC>;

    SrcTile srcTile(k, n);
    DstTile dstTile(k, n);
    pto::TASSIGN(srcTile, srcL1Offset);
    pto::TASSIGN(dstTile, dstL0Offset);
    pto::TMOV(dstTile, srcTile);
}

template <typename Element, int Rows, int Cols>
PTO_DEVICE void PtoLoadNdGmToNzL1(uint64_t dstL1Offset,
                                  __gm__ Element *src,
                                  layout::ND const &layoutSrc)
{
    using L1Tile = pto::Tile<pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor,
                             pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::ND>;

    const uint32_t rows = static_cast<uint32_t>(layoutSrc.shape(0));
    const uint32_t cols = static_cast<uint32_t>(layoutSrc.shape(1));
    const uint32_t leadingDim = static_cast<uint32_t>(layoutSrc.stride(0));

    if (leadingDim < STRIDE_LIMIT) {
        SrcShape srcShape(rows, cols);
        SrcStride srcStride(static_cast<int64_t>(rows) * leadingDim,
                            static_cast<int64_t>(rows) * leadingDim,
                            static_cast<int64_t>(rows) * leadingDim,
                            leadingDim);
        SrcGlobal srcGlobal(src, srcShape, srcStride);
        L1Tile dstTile(rows, cols);
        pto::TASSIGN(dstTile, dstL1Offset);
        pto::TLOAD(dstTile, srcGlobal);
    } else {
        for (uint32_t row = 0; row < rows; ++row) {
            SrcShape srcShape(1, cols);
            SrcStride srcStride(cols, cols, cols, cols);
            SrcGlobal srcGlobal(src + static_cast<uint64_t>(row) * leadingDim, srcShape, srcStride);
            L1Tile dstTile(1, cols);
            pto::TASSIGN(dstTile, dstL1Offset + static_cast<uint64_t>(row) * BYTE_PER_C0);
            pto::TLOAD(dstTile, srcGlobal);
        }
    }
}

template <typename Element, int Rows, int Cols>
PTO_DEVICE void PtoLoadNzGmToNzL1(uint64_t dstL1Offset,
                                  __gm__ Element *src,
                                  layout::Zn const &layoutDst,
                                  layout::Zn const &layoutSrc)
{
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using L1Tile = pto::Tile<pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor,
                             pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, pto::DYNAMIC, pto::DYNAMIC, C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, ELE_NUM_PER_C0, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::NZ>;

    const uint32_t rowBlocks = static_cast<uint32_t>(layoutSrc.shape(1));
    const uint32_t colBlocks = static_cast<uint32_t>(layoutSrc.shape(3));
    const uint32_t validRows = rowBlocks * C0_NUM_PER_FRACTAL;
    const uint32_t validCols = colBlocks * ELE_NUM_PER_C0;
    const uint32_t srcColBlockStride = static_cast<uint32_t>(layoutSrc.stride(3));
    const uint32_t dstColBlockStride = static_cast<uint32_t>(layoutDst.stride(3));
    const uint32_t rowBlockStride = static_cast<uint32_t>(layoutSrc.stride(1));

    if (srcColBlockStride / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
        SrcShape srcShape(colBlocks, rowBlocks);
        SrcStride srcStride(static_cast<int64_t>(srcColBlockStride) * colBlocks,
                            srcColBlockStride,
                            rowBlockStride);
        SrcGlobal srcGlobal(src, srcShape, srcStride);
        L1Tile dstTile(validRows, validCols);
        pto::TASSIGN(dstTile, dstL1Offset);
        pto::TLOAD(dstTile, srcGlobal);
    } else {
        for (uint32_t colBlock = 0; colBlock < colBlocks; ++colBlock) {
            SrcShape srcShape(1, rowBlocks);
            SrcStride srcStride(static_cast<int64_t>(rowBlockStride) * rowBlocks,
                                rowBlockStride * rowBlocks,
                                rowBlockStride);
            SrcGlobal srcGlobal(src + static_cast<uint64_t>(colBlock) * srcColBlockStride, srcShape, srcStride);
            L1Tile dstTile(validRows, ELE_NUM_PER_C0);
            pto::TASSIGN(dstTile, dstL1Offset + static_cast<uint64_t>(colBlock) * dstColBlockStride * sizeof(Element));
            pto::TLOAD(dstTile, srcGlobal);
        }
    }
}

template <bool ReluEnable, typename AccTile, typename GlobalDataOut>
PTO_DEVICE void PtoStoreAccTileToGm(GlobalDataOut &dstGlobal, AccTile &accTile, uint8_t unitFlag)
{
    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        if (unitFlag == 0b11) {
            pto::TSTORE<pto::STPhase::Final, AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(
                dstGlobal, accTile);
        } else if (unitFlag == 0b10) {
            pto::TSTORE<pto::STPhase::Partial, AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(
                dstGlobal, accTile);
        } else {
            pto::TSTORE<AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(dstGlobal, accTile);
        }
    } else {
        if (unitFlag == 0b11) {
            pto::TSTORE<pto::STPhase::Final, AccTile, GlobalDataOut>(dstGlobal, accTile);
        } else if (unitFlag == 0b10) {
            pto::TSTORE<pto::STPhase::Partial, AccTile, GlobalDataOut>(dstGlobal, accTile);
        } else {
            pto::TSTORE(dstGlobal, accTile);
        }
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
PTO_DEVICE void PtoStoreAccToGm(__gm__ ElementDst *dst,
                                uint64_t accOffset,
                                uint64_t scaleOffset,
                                layout::ND const &dstLayout)
{
    using GlobalDataOut = PtoGlobalNd<ElementDst>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1,
                                  pto::DYNAMIC, pto::SLayout::NoneBox>;

    const int validRow = static_cast<int>(dstLayout.shape(0));
    const int validCol = static_cast<int>(dstLayout.shape(1));
    const int64_t leadingDim = static_cast<int64_t>(dstLayout.stride(0));

    GlobalDataOut dstGlobal = pto_ext::dispatch_ffn_combine_v3::pto_bridge::MakeGlobalFromPtr(
        dst, validRow, validCol, leadingDim);
    AccTile accTile(validRow, validCol);
    ScalingTile scalingTile(validCol);

    pto::TASSIGN(accTile, accOffset);
    pto::TASSIGN(scalingTile, scaleOffset);

    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, pto::AtomicType::AtomicNone, reluMode>(
            dstGlobal, accTile, scalingTile);
    } else {
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, accTile, scalingTile);
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
PTO_DEVICE void PtoStoreAccToGm(__gm__ ElementDst *dst,
                                uint64_t accOffset,
                                layout::ND const &dstLayout,
                                uint8_t unitFlag = 0)
{
    using GlobalDataOut = PtoGlobalNd<ElementDst>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

    const int validRow = static_cast<int>(dstLayout.shape(0));
    const int validCol = static_cast<int>(dstLayout.shape(1));
    const int64_t leadingDim = static_cast<int64_t>(dstLayout.stride(0));

    GlobalDataOut dstGlobal = pto_ext::dispatch_ffn_combine_v3::pto_bridge::MakeGlobalFromPtr(
        dst, validRow, validCol, leadingDim);
    AccTile accTile(validRow, validCol);

    pto::TASSIGN(accTile, accOffset);
    PtoStoreAccTileToGm<ReluEnable>(dstGlobal, accTile, unitFlag);
}

template <int Cols>
PTO_DEVICE void StagePerChannelScale(uint64_t l1SOffset,
                                     uint64_t fixpipeOffset,
                                     __gm__ uint64_t *gmBlockS,
                                     layout::VectorLayout const &layoutScale,
                                     uint32_t cols)
{
    using ScaleMatTile = pto::Tile<pto::TileType::Mat, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1,
                                   pto::DYNAMIC, pto::SLayout::NoneBox>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1,
                                  pto::DYNAMIC, pto::SLayout::NoneBox>;

    auto layoutTileS = layoutScale.GetTileLayout(MakePtoCoord1D(cols));
    uint32_t validCols = static_cast<uint32_t>(layoutTileS.shape(0));
    using ScaleShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using ScaleStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<uint64_t, ScaleShape, ScaleStride, pto::Layout::ND>;
    ScaleShape scaleShape(validCols);
    ScaleStride scaleStride(validCols, validCols, validCols, validCols);
    ScaleGlobal gmBlockSGlobal(gmBlockS, scaleShape, scaleStride);
    ScaleMatTile scaleMatTile(validCols);
    ScalingTile scalingTile(validCols);

    pto::TASSIGN(scaleMatTile, l1SOffset);
    pto::TASSIGN(scalingTile, fixpipeOffset);
    pto::TLOAD(scaleMatTile, gmBlockSGlobal);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(0);
    pto::TMOV(scalingTile, scaleMatTile);
}

template <typename ElementA, typename ElementC, typename ElementAccumulator, int Rows, int Cols>
PTO_DEVICE void StoreAccumulator(__gm__ ElementC *dst,
                                 uint64_t accOffset,
                                 uint64_t scaleOffset,
                                 layout::ND const &dstLayout,
                                 uint8_t unitFlag = 0)
{
    if constexpr (std::is_same_v<ElementA, int8_t>) {
        PtoStoreAccToGm<ElementC, ElementAccumulator, Rows, Cols>(dst, accOffset, scaleOffset, dstLayout);
    } else if constexpr (std::is_same_v<ElementA, half>) {
        PtoStoreAccToGm<ElementC, ElementAccumulator, Rows, Cols>(dst, accOffset, dstLayout, unitFlag);
    }
}

template <class ArchTag, class TileCopy_, class AType_, class BType_, class CType_>
struct MatmulShell {
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;

    using CopyL1ToFP = typename pto_ext::Gemm::PtoQuantTileCopy<
        ArchTag,
        AType_,
        BType_,
        CType_,
        void,
        pto_ext::Gemm::Tile::ScaleGranularity::PER_CHANNEL>::CopyL1ToFP;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator = typename pto_ext::Gemm::PtoElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyL0CToGm = typename std::conditional<
        std::is_same_v<ElementA, int8_t>,
        pto_ext::Gemm::PtoCopyL0CToGm<ArchTag, ElementAccumulator, CType_, Gemm::Tile::ScaleGranularity::PER_CHANNEL>,
        typename TileCopy_::CopyL0CToGm>::type;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using L1AAlignHelper = pto_ext::Gemm::PtoL1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = pto_ext::Gemm::PtoL1AlignHelper<ElementB, LayoutB>;
};

}  // namespace detail


template<AscendC::HardEvent event>
__aicore__ inline void SyncFlagFunc(int32_t eventID)
{
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <
    uint32_t PRELOAD_STAGES_,
    uint32_t L1_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2PreloadAsyncFixpipe<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2PreloadAsyncFixpipe<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >;
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
    using CopyL1ToFP = typename MatmulShell::CopyL1ToFP;
    using CopyL1ToL0A = typename MatmulShell::CopyL1ToL0A;
    using CopyL1ToL0B = typename MatmulShell::CopyL1ToL0B;
    using ElementAccumulator = typename MatmulShell::ElementAccumulator;
    using CopyL0CToGm = typename MatmulShell::CopyL0CToGm;
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
    static_assert(
        (std::is_same_v<ElementA, int8_t> 
            ? (L1A_TILE_SIZE + L1B_TILE_SIZE + L1S_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE
            : (L1A_TILE_SIZE + L1B_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE),
        "L1TileShape exceeding the L1 space for the given data type"
    );

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    PTO_DEVICE static LayoutAInL1 MakeL1ALayout()
    {
        return LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
    }

    PTO_DEVICE static LayoutBInL1 MakeL1BLayout()
    {
        return LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
    }

    PTO_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, __gm__ int32_t* flagPtr = nullptr, int32_t expertPerRank = 0, 
                uint32_t l1BufAddrStart = 0, uint32_t FpAddrStart = 0)
    {
        syncGroupIdx = 0;
        ptrSoftFlagBase_ = flagPtr;
        expertPerRank_ = expertPerRank;
        InitL1(resource, l1BufAddrStart);
        InitFpBuf(resource, FpAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
    }

    PTO_DEVICE
    ~BlockMmad()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
        }
    }

    PTO_DEVICE
    void operator()(
        __gm__ ElementA *gmBlockAPtr, LayoutA const &layoutA,
        __gm__ ElementB *gmBlockBPtr, LayoutB const &layoutB,
        __gm__ ElementC *gmBlockCPtr, LayoutC const &layoutC,
        __gm__ uint64_t *gmBlockSPtr, layout::VectorLayout const &layoutScale,
        PtoShape3D const &actualShape, int32_t syncLoopIdx = -1, int32_t flag = 0
    )
    {
        uint32_t actualM = GetPtoShapeM(actualShape);
        uint32_t actualN = GetPtoShapeN(actualShape);
        uint32_t actualK = GetPtoShapeK(actualShape);
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualK);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualM);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualN);

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ?
                (startTileIdx + kLoopIdx) : (startTileIdx + kLoopIdx - kTileCount);

            uint32_t kActual = (kTileIdx < kTileCount - 1) ?
                L1TileShape::K : (actualK - kTileIdx * L1TileShape::K);

            // Emission load instruction from GM to L1
            auto gmTileAOffset = MakePtoCoord2D(0, kTileIdx * L1TileShape::K);
            auto gmTileBOffset = MakePtoCoord2D(kTileIdx * L1TileShape::K, 0);
            __gm__ ElementA *gmTileA = gmBlockAPtr + layoutA.GetOffset(gmTileAOffset);
            __gm__ ElementB *gmTileB = gmBlockBPtr + layoutB.GetOffset(gmTileBOffset);
            // Load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakePtoCoord2D(actualM, kActual));
            detail::PtoLoadNdGmToNzL1<ElementA, L1TileShape::M, L1TileShape::K>(
                l1AOffsetList[l1ListId], gmTileA, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            // Load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakePtoCoord2D(kActual, actualN));
            detail::PtoLoadNzGmToNzL1<ElementB, L1TileShape::K, L1TileShape::N>(
                l1BOffsetList[l1ListId], gmTileB, MakeL1BLayout(), layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            // If the number of preload instructions reaches the upper limit, perform an mmad calculation on L1 tile
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Store the current load status
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES) ?
                (l1TileMmadParamsId + preloadCount) : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1ListId = l1ListId;
            l1TileMmadParams.mRound = mRound;
            l1TileMmadParams.nRound = nRound;
            l1TileMmadParams.kActual = kActual;
            l1TileMmadParams.isKLoopFirst = (kLoopIdx == 0);
            l1TileMmadParams.isKLoopLast = (kLoopIdx == kTileCount - 1);
            l1TileMmadParams.flag = flag;
            if (kLoopIdx == kTileCount - 1) {
                l1TileMmadParams.gmBlockC = gmBlockCPtr;
                l1TileMmadParams.gmBlockS = gmBlockSPtr;
                l1TileMmadParams.layoutCInGm = layoutC.GetTileLayout(GetPtoShapeMN(actualShape));
                l1TileMmadParams.layoutScale = layoutScale;
                l1TileMmadParams.syncLoopIdx = syncLoopIdx;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    PTO_DEVICE
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

    PTO_DEVICE
    void Finalize(int32_t target, int32_t flag = 0)
    {
        if (ptrSoftFlagBase_ != nullptr) {
            if (target < 0) {
                return;
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(EVENT_ID0);
            __gm__ int32_t *flagPtr = reinterpret_cast<__gm__ int32_t*>(ptrSoftFlagBase_) +
                                      (expertPerRank_ + AscendC::GetBlockIdx()) * FLAGSTRIDE;
            detail::PtoStoreSoftFlagL1(flagPtr, l1FBaseOffset + static_cast<uint64_t>(target * 16) * sizeof(int32_t), FLAGSTRIDE);
        }
        else {
            for(;syncGroupIdx <= target; syncGroupIdx++) {
                int32_t flagId = syncGroupIdx / 15 + flag;
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
            }
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
        PTO_DEVICE
        L1TileMmadParams() = default;
    };

    PTO_DEVICE
    void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;

        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1AOffsetList[i] = resource.l1Buf.GetBufferAddrByByte(l1AOffset + L1A_TILE_SIZE * i);
            l1BOffsetList[i] = resource.l1Buf.GetBufferAddrByByte(l1BOffset + L1B_TILE_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        uint32_t l1SOffset = l1BOffset + L1B_TILE_SIZE * L1_STAGES;
        if constexpr (std::is_same_v<ElementA, int8_t>) {
            l1SBaseOffset = resource.l1Buf.GetBufferAddrByByte(l1SOffset);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
        }
        if (ptrSoftFlagBase_ != nullptr) {
            // Initialize the flag matrix (structure as below):
            // 1 0 0 0 0 0 0 0
            // 2 0 0 0 0 0 0 0
            // ...
            // 16 0 0 0 0 0 0 0
            // Then move it to L1
            uint32_t l1FOffset = l1SOffset + L1S_TILE_SIZE;
            l1FBaseOffset = resource.l1Buf.GetBufferAddrByByte(l1FOffset);
            __gm__ int32_t *flagBase = reinterpret_cast<__gm__ int32_t*>(ptrSoftFlagBase_);
            detail::PtoLoadSoftFlagL1(l1FBaseOffset, flagBase, expertPerRank_ * FLAGSTRIDE);
        }
    }

    PTO_DEVICE
    void InitFpBuf(Arch::Resource<ArchTag> &resource, uint32_t FpAddrStart)
    {
        uint32_t FpOffset = FpAddrStart;
        fixpipeBaseOffset = resource.fpBuf.GetBufferAddrByByte(FpOffset);
    }

    PTO_DEVICE
    void InitL0A(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0AOffsetList[i] = resource.l0ABuf.GetBufferAddrByByte(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    PTO_DEVICE
    void InitL0B(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BOffsetList[i] = resource.l0BBuf.GetBufferAddrByByte(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    PTO_DEVICE
    void InitL0C(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0COffsetList[i] = resource.l0CBuf.GetBufferAddrByByte(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    PTO_DEVICE
    void L1TileMmad(L1TileMmadParams const &params)
    {
        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(params.mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(params.nRound);
        uint32_t kPartLoop = CeilDiv<L0TileShape::K>(params.kActual);
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakePtoCoord2D(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                L0TileShape::M : (params.mRound - mPartIdx * L0TileShape::M);

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                    L0TileShape::K : (params.kActual - kPartIdx * L0TileShape::K);

                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                auto l1AOffset = MulPtoCoord2D(MakePtoCoord2D(mPartIdx, kPartIdx), L0TileShape::ToPtoShapeMK());
                const uint64_t l1AOffsetBytes = l1AOffsetList[params.l1ListId] +
                    static_cast<uint64_t>(MakeL1ALayout().GetOffset(l1AOffset)) * sizeof(ElementA);
                const uint64_t l0AStagingOffsetBytes = l0AOffsetList[l0AListId] +
                    static_cast<uint64_t>(layoutAInL0.GetOffset(MakePtoCoord2D(0, 0))) * sizeof(ElementA);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
                }
                detail::PtoMoveL1ToL0A<ElementA, L0TileShape>(l0AStagingOffsetBytes, l1AOffsetBytes, mPartActual, kPartActual);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                        L0TileShape::N : (params.nRound - nPartIdx * L0TileShape::N);

                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                    auto l1BOffset = MulPtoCoord2D(MakePtoCoord2D(kPartIdx, nPartIdx), L0TileShape::ToPtoShapeKN());
                    const uint64_t l1BOffsetBytes = l1BOffsetList[params.l1ListId] +
                        static_cast<uint64_t>(MakeL1BLayout().GetOffset(l1BOffset)) * sizeof(ElementB);
                    const uint64_t l0BStagingOffsetBytes = l0BOffsetList[l0BListId] +
                        static_cast<uint64_t>(layoutBInL0.GetOffset(MakePtoCoord2D(0, 0))) * sizeof(ElementB);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
                    }
                    detail::PtoMoveL1ToL0B<ElementB, L0TileShape>(l0BStagingOffsetBytes, l1BOffsetBytes, kPartActual, nPartActual);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    auto l0COffset = MulPtoCoord2D(MakePtoCoord2D(mPartIdx, nPartIdx), L0TileShape::ToPtoShapeMN());

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (params.isKLoopLast &&
                            (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    const uint64_t l0AOffsetBytes = l0AOffsetList[l0AListId] +
                        static_cast<uint64_t>(layoutAInL0.GetOffset(MakePtoCoord2D(0, 0))) * sizeof(ElementA);
                    const uint64_t l0BOffsetBytes = l0BOffsetList[l0BListId] +
                        static_cast<uint64_t>(layoutBInL0.GetOffset(MakePtoCoord2D(0, 0))) * sizeof(ElementB);
                    const uint64_t l0COffsetBytes = l0COffsetList[l0CListId] +
                        static_cast<uint64_t>(layoutCInL0.GetOffset(l0COffset)) * sizeof(ElementAccumulator);
                    detail::PtoTileMmad<ElementAccumulator, ElementA, ElementB, L0TileShape>(
                        l0COffsetBytes, l0AOffsetBytes, l0BOffsetBytes, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);
                detail::StagePerChannelScale<L1TileShape::N>(
                    l1SBaseOffset,
                    fixpipeBaseOffset,
                    params.gmBlockS,
                    params.layoutScale,
                    layoutCInGm.shape(1));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(0);
                AscendC::PipeBarrier<PIPE_FIX>();
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                    params.gmBlockC, l0COffsetList[l0CListId], fixpipeBaseOffset, layoutCInGm);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                detail::StoreAccumulator<ElementA, ElementC, ElementAccumulator, L1TileShape::M, L1TileShape::N>(
                    params.gmBlockC, l0COffsetList[l0CListId], fixpipeBaseOffset, layoutCInGm, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
            if constexpr (std::is_same_v<ElementA, int8_t>) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
            }
            #ifdef __TILE_SYNC__
            if (params.flag > 0) {
                int32_t flagId = params.flag + params.syncLoopIdx / 8;
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
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
    uint64_t l1FBaseOffset{0};
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

    __gm__ int32_t* ptrSoftFlagBase_ = nullptr;
    int32_t expertPerRank_;
};

}  // namespace pto_ext::Gemm::Block

#endif  // PTO_EXT_GEMM_BLOCK_MMAD_PRELOAD_FIXPIPE_QUANT_HPP