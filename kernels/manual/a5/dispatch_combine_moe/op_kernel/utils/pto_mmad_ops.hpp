/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_EXT_DISPATCH_COMBINE_MOE_PTO_MMAD_OPS_HPP
#define PTO_EXT_DISPATCH_COMBINE_MOE_PTO_MMAD_OPS_HPP

#include "moe_pto_utils.hpp"
#include "pto_vector_ops.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

namespace pto_ext::Gemm::Block::detail {

using pto_ext::dispatch_combine_moe::pto_bridge::PtoGlobalNd;

template <typename TileAcc, typename TileLeft, typename TileRight>
__forceinline__ __aicore__ void LaunchPtoMatmul(TileAcc &cTile, TileLeft &aTile, TileRight &bTile, bool initC,
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
__forceinline__ __aicore__ void PtoTileMmad(uint64_t l0COffset, uint64_t l0AOffset, uint64_t l0BOffset, uint32_t m,
                                            uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
{
    using LeftTile = pto::TileLeft<ElementA, L0TileShape::M, L0TileShape::K, pto::DYNAMIC, pto::DYNAMIC>;
    using RightTile = pto::TileRight<ElementB, L0TileShape::K, L0TileShape::N, pto::DYNAMIC, pto::DYNAMIC>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, L0TileShape::M, L0TileShape::N, pto::DYNAMIC, pto::DYNAMIC>;

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

template <typename Element, class L0TileShape>
__forceinline__ __aicore__ void PtoMoveL1ToL0A(uint64_t dstL0Offset, uint64_t srcL1Offset, uint32_t m, uint32_t k)
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
__forceinline__ __aicore__ void PtoMoveL1ToL0B(uint64_t dstL0Offset, uint64_t srcL1Offset, uint32_t k, uint32_t n)
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
__forceinline__ __aicore__ void PtoLoadNdGmToNzL1(uint64_t dstL1Offset, __gm__ Element *src,
                                                  layout::ND const &layoutSrc)
{
    using L1Tile = pto::Tile<pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::ND>;

    const uint32_t rows = static_cast<uint32_t>(layoutSrc.shape(0));
    const uint32_t cols = static_cast<uint32_t>(layoutSrc.shape(1));
    const uint32_t leadingDim = static_cast<uint32_t>(layoutSrc.stride(0));

    if (leadingDim < STRIDE_LIMIT) {
        SrcShape srcShape(rows, cols);
        SrcStride srcStride(static_cast<int64_t>(rows) * leadingDim, static_cast<int64_t>(rows) * leadingDim,
                            static_cast<int64_t>(rows) * leadingDim, leadingDim);
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
__forceinline__ __aicore__ void PtoLoadNzGmToNzL1(uint64_t dstL1Offset, __gm__ Element *src,
                                                  layout::Zn const &layoutDst, layout::Zn const &layoutSrc)
{
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using L1Tile = pto::Tile<pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::RowMajor>;
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
        SrcStride srcStride(static_cast<int64_t>(srcColBlockStride) * colBlocks, srcColBlockStride, rowBlockStride);
        SrcGlobal srcGlobal(src, srcShape, srcStride);
        L1Tile dstTile(validRows, validCols);
        pto::TASSIGN(dstTile, dstL1Offset);
        pto::TLOAD(dstTile, srcGlobal);
    } else {
        for (uint32_t colBlock = 0; colBlock < colBlocks; ++colBlock) {
            SrcShape srcShape(1, rowBlocks);
            SrcStride srcStride(static_cast<int64_t>(rowBlockStride) * rowBlocks, rowBlockStride * rowBlocks,
                                rowBlockStride);
            SrcGlobal srcGlobal(src + static_cast<uint64_t>(colBlock) * srcColBlockStride, srcShape, srcStride);
            L1Tile dstTile(validRows, ELE_NUM_PER_C0);
            pto::TASSIGN(dstTile, dstL1Offset + static_cast<uint64_t>(colBlock) * dstColBlockStride * sizeof(Element));
            pto::TLOAD(dstTile, srcGlobal);
        }
    }
}

template <bool ReluEnable, typename AccTile, typename GlobalDataOut>
__forceinline__ __aicore__ void PtoStoreAccTileToGm(GlobalDataOut &dstGlobal, AccTile &accTile, uint8_t unitFlag)
{
    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        if (unitFlag == 0b11) {
            pto::TSTORE<pto::STPhase::Final, AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(dstGlobal,
                                                                                                            accTile);
        } else if (unitFlag == 0b10) {
            pto::TSTORE<pto::STPhase::Partial, AccTile, GlobalDataOut, pto::AtomicType::AtomicNone, reluMode>(dstGlobal,
                                                                                                              accTile);
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
__forceinline__ __aicore__ void PtoStoreAccToGm(__gm__ ElementDst *dst, uint64_t accOffset, uint64_t scaleOffset,
                                                layout::ND const &dstLayout)
{
    using GlobalDataOut = PtoGlobalNd<ElementDst>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC,
                                  pto::SLayout::NoneBox>;

    const int validRow = static_cast<int>(dstLayout.shape(0));
    const int validCol = static_cast<int>(dstLayout.shape(1));
    const int64_t leadingDim = static_cast<int64_t>(dstLayout.stride(0));

    GlobalDataOut dstGlobal =
        pto_ext::dispatch_combine_moe::pto_bridge::MakeGlobalFromPtr(dst, validRow, validCol, leadingDim);
    AccTile accTile(validRow, validCol);
    ScalingTile scalingTile(validCol);

    pto::TASSIGN(accTile, accOffset);
    pto::TASSIGN(scalingTile, scaleOffset);

    if constexpr (ReluEnable) {
        constexpr auto reluMode = pto::ReluPreMode::NormalRelu;
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, pto::AtomicType::AtomicNone, reluMode>(dstGlobal, accTile,
                                                                                                   scalingTile);
    } else {
        pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, accTile, scalingTile);
    }
}

template <typename ElementDst, typename ElementAccumulator, int Rows, int Cols, bool ReluEnable = false>
__forceinline__ __aicore__ void PtoStoreAccToGm(__gm__ ElementDst *dst, uint64_t accOffset, layout::ND const &dstLayout,
                                                uint8_t unitFlag = 0)
{
    using GlobalDataOut = PtoGlobalNd<ElementDst>;
    using AccTile = pto::TileAccCompact<ElementAccumulator, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

    const int validRow = static_cast<int>(dstLayout.shape(0));
    const int validCol = static_cast<int>(dstLayout.shape(1));
    const int64_t leadingDim = static_cast<int64_t>(dstLayout.stride(0));

    GlobalDataOut dstGlobal =
        pto_ext::dispatch_combine_moe::pto_bridge::MakeGlobalFromPtr(dst, validRow, validCol, leadingDim);
    AccTile accTile(validRow, validCol);

    pto::TASSIGN(accTile, accOffset);
    PtoStoreAccTileToGm<ReluEnable>(dstGlobal, accTile, unitFlag);
}

template <int Cols>
__forceinline__ __aicore__ void StagePerChannelScale(uint64_t l1SOffset, uint64_t fixpipeOffset,
                                                     __gm__ uint64_t *gmBlockS, layout::VectorLayout const &layoutScale,
                                                     uint32_t cols)
{
    using ScaleMatTile = pto::Tile<pto::TileType::Mat, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC,
                                   pto::SLayout::NoneBox>;
    using ScalingTile = pto::Tile<pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC,
                                  pto::SLayout::NoneBox>;

    auto layoutTileS = layoutScale.GetTileLayout(PtoCoord1D(cols));
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
__forceinline__ __aicore__ void StoreAccumulator(__gm__ ElementC *dst, uint64_t accOffset, uint64_t scaleOffset,
                                                 layout::ND const &dstLayout, uint8_t unitFlag = 0)
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

    using CopyL1ToFPTraits = typename pto_ext::Gemm::Tile::QuantTileCopy<
        ArchTag, AType_, BType_, CType_, void, pto_ext::Gemm::Tile::ScaleGranularity::PER_CHANNEL>::CopyL1ToFPTraits;
    using CopyL1ToL0ATraits = typename TileCopy_::CopyL1ToL0ATraits;
    using CopyL1ToL0BTraits = typename TileCopy_::CopyL1ToL0BTraits;
    using ElementAccumulator =
        typename pto_ext::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyL0CToGmTraits =
        typename std::conditional<std::is_same_v<ElementA, int8_t>,
                                  pto_ext::Gemm::Tile::CopyL0CToGmTraits<ArchTag, ElementAccumulator, CType_,
                                                                         Gemm::Tile::ScaleGranularity::PER_CHANNEL>,
                                  typename TileCopy_::CopyL0CToGmTraits>::type;
    using LayoutAInL1 = typename CopyL1ToL0ATraits::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0BTraits::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0ATraits::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0BTraits::LayoutDst;
    using L1AAlignHelper = pto_ext::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = pto_ext::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;
};

} // namespace pto_ext::Gemm::Block::detail

#endif // PTO_EXT_DISPATCH_COMBINE_MOE_PTO_MMAD_OPS_HPP
