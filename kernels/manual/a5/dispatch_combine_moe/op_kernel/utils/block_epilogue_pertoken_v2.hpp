/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_V2_ONLY_HPP
#define PTO_EXT_EPILOGUE_BLOCK_PER_TOKEN_V2_ONLY_HPP

#include "moe_pto_utils.hpp"
#include "dispatch_policy_custom.hpp"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "hccl_window.hpp"
#include "layout3d.hpp"
#include "pto_vector_ops.hpp"

namespace pto_ext::Epilogue::Block {
namespace detail {

using pto_ext::PtoPipeBarrier;
using pto_ext::PtoSetFlag;
using pto_ext::PtoWaitFlag;

using pto_ext::dispatch_combine_moe::pto_bridge::PtoElemOffsetBytes;

template <typename Element>
__forceinline__ __aicore__ __ubuf__ Element *PtoUbPtr(uint64_t ubOffsetBytes)
{
    return reinterpret_cast<__ubuf__ Element *>(ubOffsetBytes);
}

template <typename DstElement, typename SrcElement, int TileElems = 128>
__forceinline__ __aicore__ void PtoCastVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum,
                                              pto::RoundMode mode)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoCastVector<DstElement, SrcElement, TileElems>(
        dstUbOffsetBytes, srcUbOffsetBytes, elemNum, mode);
}

template <typename Element, int TileElems = 128>
__forceinline__ __aicore__ void PtoMulVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum,
                                             Element scalar)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoMulVector<Element, TileElems>(dstUbOffsetBytes, srcUbOffsetBytes,
                                                                                elemNum, scalar);
}

template <typename Element, int TileElems = 128>
__forceinline__ __aicore__ void PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t elemNum)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoLoadVector<Element, TileElems>(dstUbOffsetBytes, src, elemNum);
}

template <typename Element, int TileElems = 128>
__forceinline__ __aicore__ void PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreVector<Element, TileElems>(dst, srcUbOffsetBytes, elemNum);
}

template <typename Element, int TileElems = 128>
__forceinline__ __aicore__ void PtoLoadMatrixRows(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t rowNum,
                                                  uint32_t colNum, uint32_t dstStride, uint32_t srcStride)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoLoadMatrixRows<Element, TileElems>(dstUbOffsetBytes, src, rowNum,
                                                                                     colNum, dstStride, srcStride);
}

template <typename Element, int TileElems = 128>
__forceinline__ __aicore__ void PtoStoreMatrixRows(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t rowNum,
                                                   uint32_t colNum, uint32_t dstStride, uint32_t srcStride)
{
    pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreMatrixRows<Element, TileElems>(dst, srcUbOffsetBytes, rowNum,
                                                                                      colNum, dstStride, srcStride);
}

} // namespace detail

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA5PerTokenDequantV2<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_> {
public:
    using DispatchPolicy = EpilogueAtlasA5PerTokenDequantV2<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        PtoRemoteWindow remoteWindow;
        int32_t offsetD;
        int32_t scratchOffset;
        Layout3D tokenPerExpertLayout;
        __forceinline__ __aicore__ Params(){};
        __forceinline__ __aicore__ Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_,
                                          __gm__ int32_t *ptrTokenPerExpert_, LayoutC layoutC_, int32_t n2_,
                                          int32_t n0_, PtoRemoteWindow &remoteWindow_, int32_t offsetD_,
                                          int32_t scratchOffset_, Layout3D tokenPerExpertLayout_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              remoteWindow(remoteWindow_),
              offsetD(offsetD_),
              scratchOffset(scratchOffset_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
    };

    __forceinline__ __aicore__ BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{})
        : params(params)
    {
        // ub:192KB
        n0 = params.n0;
        uint64_t ubOffset = 0;
        for (int32_t i = 0; i < 2; i++) {
            ubCOffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(ElementC);
            ubDOffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(ElementD);
            ubFp32OffsetList[i] = ubOffset;
            ubOffset += max_len * sizeof(float);
            scaleUbOffsetList[i] = ubOffset;
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpertPtr = reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert);
        tokenPerExpertLayout = params.tokenPerExpertLayout;
        is_ping = true;
    }
    __forceinline__ __aicore__ void SetFlag()
    {
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID0);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID1);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID3);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::S_MTE2>(EVENT_ID2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::S_MTE2>(EVENT_ID3);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE3_V>(EVENT_ID0);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE3_V>(EVENT_ID1);
    }

    __forceinline__ __aicore__ void Finalize()
    {
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID0);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID1);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID2);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(EVENT_ID3);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::S_MTE2>(EVENT_ID2);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::S_MTE2>(EVENT_ID3);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE3_V>(EVENT_ID0);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE3_V>(EVENT_ID1);
    }
    __forceinline__ __aicore__ ~BlockEpilogue()
    {}
    __forceinline__ __aicore__ void operator()(__gm__ ElementC *gmCPtr, __gm__ ElementPerTokenScale *gmPerTokenScalePtr,
                                               PtoCoord2D const &blockCoord, PtoShape2D const &actualBlockShape,
                                               int32_t groupIdx, int32_t preSrcExpertSum,
                                               __gm__ int32_t *preSumBeforeRank)
    {
        is_ping = !is_ping;
        int32_t eventId = is_ping ? EVENT_ID0 : EVENT_ID1;
        int32_t eventId2 = is_ping ? EVENT_ID2 : EVENT_ID3;
        int32_t blockRow = static_cast<int32_t>(blockCoord.shape[0]);
        int32_t blockCol = static_cast<int32_t>(blockCoord.shape[1]);
        uint32_t actualM = static_cast<uint32_t>(actualBlockShape.shape[0]);
        uint32_t actualN = static_cast<uint32_t>(actualBlockShape.shape[1]);

        uint64_t ubCOffset = ubCOffsetList[is_ping];
        uint64_t ubDOffset = ubDOffsetList[is_ping];
        uint64_t ubCFp32Offset = ubFp32OffsetList[is_ping];
        uint64_t scaleUbOffset = scaleUbOffsetList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockRow * params.n2 + blockCol;

        LoadTileC(gmCPtr + gmCOffset, actualM, actualN, eventId, ubCOffset, ubCFp32Offset);
        ScaleAndCastTile(gmPerTokenScalePtr, preSrcExpertSum + blockRow, actualM, actualN, eventId, eventId2,
                         scaleUbOffset, ubCFp32Offset, ubDOffset);
        StoreTileD(groupIdx, blockRow, blockCol, actualM, actualN, preSumBeforeRank, eventId, ubDOffset);
    }

private:
    __forceinline__ __aicore__ void LoadTileC(__gm__ ElementC *gmTileC, uint32_t actualM, uint32_t actualN,
                                              int32_t eventId, uint64_t ubCOffset, uint64_t ubCFp32Offset)
    {
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(eventId);
        detail::PtoLoadMatrixRows(ubCOffset, gmTileC, actualM, actualN, n0, params.n2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_V>(eventId);

        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_V>(eventId);
        for (uint32_t row = 0; row < actualM; ++row) {
            detail::PtoCastVector<float, ElementC>(ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row),
                                                   ubCOffset + detail::PtoElemOffsetBytes<ElementC>(n0 * row), actualN,
                                                   pto::RoundMode::CAST_NONE);
        }
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(eventId);
    }

    __forceinline__ __aicore__ void ScaleAndCastTile(__gm__ ElementPerTokenScale *gmPerTokenScalePtr,
                                                     int32_t gmScaleOffset, uint32_t actualM, uint32_t actualN,
                                                     int32_t eventId, int32_t eventId2, uint64_t scaleUbOffset,
                                                     uint64_t ubCFp32Offset, uint64_t ubDOffset)
    {
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE2>(eventId2);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::S_MTE2>(eventId2);
        if (source_scale_offset[eventId] != gmScaleOffset) {
            source_scale_offset[eventId] = gmScaleOffset;
            detail::PtoLoadVector(scaleUbOffset, gmPerTokenScalePtr + gmScaleOffset, actualM);
        }
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_S>(eventId2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE2_V>(eventId2);

        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_V>(eventId2);
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE2_S>(eventId2);
        detail::PtoPipeBarrier<PIPE_V>();
        __ubuf__ float *scaleUbPtr = detail::PtoUbPtr<float>(scaleUbOffset);
        for (uint32_t row = 0; row < actualM; ++row) {
            float scale = scaleUbPtr[row];
            uint64_t rowFp32Offset = ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row);
            detail::PtoMulVector<float>(rowFp32Offset, rowFp32Offset, actualN, scale);
        }
        detail::PtoPipeBarrier<PIPE_V>();
        detail::PtoWaitFlag<pto_ext::PtoHardEvent::MTE3_V>(eventId);
        for (uint32_t row = 0; row < actualM; ++row) {
            detail::PtoCastVector<ElementD, float>(ubDOffset + detail::PtoElemOffsetBytes<ElementD>(n0 * row),
                                                   ubCFp32Offset + detail::PtoElemOffsetBytes<float>(n0 * row), actualN,
                                                   pto::RoundMode::CAST_RINT);
        }
        detail::PtoSetFlag<pto_ext::PtoHardEvent::S_MTE2>(eventId2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE2>(eventId2);
        detail::PtoSetFlag<pto_ext::PtoHardEvent::V_MTE3>(eventId);
    }

    __forceinline__ __aicore__ void StoreRemoteRows(__gm__ void *dstPeermemPtr, int64_t gmDstOffset, uint32_t lenData,
                                                    uint32_t actualN, int32_t tileOffset, uint64_t ubDOffset)
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using TputGlobal = pto::GlobalTensor<ElementD, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using TputTile = pto::Tile<pto::TileType::Vec, ElementD, 1, 128, pto::BLayout::RowMajor, -1, -1>;

        int32_t logicalSubCoreIdx = pto_ext::PtoAivLogicalIdx();
        int64_t scratchOffsetBytes =
            params.scratchOffset + static_cast<int64_t>(logicalSubCoreIdx) * n0 * sizeof(ElementD);
        __gm__ ElementD *localScratch =
            reinterpret_cast<__gm__ ElementD *>(params.remoteWindow(scratchOffsetBytes, params.rank));
        ShapeDyn rowShape(1, 1, 1, 1, actualN);
        StrideDyn localStride(actualN, actualN, actualN, actualN, 1);
        StrideDyn remoteStride(params.n2, params.n2, params.n2, params.n2, 1);
        TputTile tputTile(1, actualN);
        __gm__ ElementD *remotePeerBase = reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr) + gmDstOffset;

        for (uint32_t rowIdx = 0; rowIdx < lenData; ++rowIdx) {
            detail::PtoStoreVector(
                localScratch, ubDOffset + detail::PtoElemOffsetBytes<ElementD>((tileOffset + rowIdx) * n0), actualN);
            TputGlobal localRowG(localScratch, rowShape, localStride);
            TputGlobal remoteRowG(remotePeerBase + rowIdx * params.n2, rowShape, remoteStride);
            pto::comm::TPUT(remoteRowG, localRowG, tputTile);
            detail::PtoPipeBarrier<PIPE_ALL>();
        }
    }

    __forceinline__ __aicore__ void StoreTileD(int32_t groupIdx, int32_t blockRow, int32_t blockCol, uint32_t actualM,
                                               uint32_t actualN, __gm__ int32_t *preSumBeforeRank, int32_t eventId,
                                               uint64_t ubDOffset)
    {
        int32_t stTile = blockRow;
        int32_t edTile = stTile + static_cast<int32_t>(actualM);
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        detail::PtoWaitFlag<pto_ext::PtoHardEvent::V_MTE3>(eventId);
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
            int32_t lenRankInExpert =
                gm_load(tokenPerExpertPtr + tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = gm_load(preSumBeforeRank + dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            } else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0) {
                continue;
            }

            uint32_t dstOffsetInExpert = (stTile > stRankInExpert) ? (stTile - stRankInExpert) : 0;
            __gm__ void *dstPeermemPtr = params.remoteWindow(params.offsetD, dstEpIdx);
            auto dstOffset = PtoCoord2D(dstOffsetInExpert + dstExpertOffset, blockCol);
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            __gm__ ElementD *gmTileD = reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr) + gmDstOffset;
            if (dstEpIdx == params.rank) {
                detail::PtoStoreMatrixRows(gmTileD, ubDOffset + detail::PtoElemOffsetBytes<ElementD>(tileOffset * n0),
                                           lenData, actualN, params.n2, n0);
            } else {
                StoreRemoteRows(dstPeermemPtr, gmDstOffset, lenData, actualN, tileOffset, ubDOffset);
            }
            tileOffset += lenData;
        }
        detail::PtoSetFlag<pto_ext::PtoHardEvent::MTE3_V>(eventId);
    }

    Params params;
    uint64_t ubCOffsetList[UB_STAGES];
    uint64_t ubDOffsetList[UB_STAGES];
    uint64_t ubFp32OffsetList[UB_STAGES];
    uint64_t scaleUbOffsetList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    __gm__ int32_t *tokenPerExpertPtr;
    Layout3D tokenPerExpertLayout;
};
} // namespace pto_ext::Epilogue::Block
#endif