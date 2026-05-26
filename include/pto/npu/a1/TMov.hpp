/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP
#include "common.hpp"
#include "TExtract.hpp"
#include "TCopy.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, QuantMode_t quantPre>
PTO_INTERNAL constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == AccToVecMode::DualModeSplitM || mode == AccToVecMode::DualModeSplitN) {
        static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
        static_assert((!(!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox)),
                      "Dual Dst Mode is not support in nz2dn.");
        return ((mode == AccToVecMode::DualModeSplitM) ? 1 : 2);
    }
    return 0;
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL constexpr uint32_t GetTmovAccDstStride()
{
    if constexpr (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        return DstTileData::Cols;
    } else if constexpr (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        return DstTileData::Rows;
    }
    constexpr bool channelSplitEnable =
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<typename DstTileData::DType, float>)&&(DstTileData::SFractalSize == 512);
    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (DstTileData::SFractalSize == 1024) ?
                                    2 * C0_SIZE_BYTE / sizeof(typename DstTileData::DType) :
                                    C0_SIZE_BYTE / sizeof(typename DstTileData::DType);
    return DstTileData::Rows * c0Size;
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, QuantMode_t quantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToUb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                              uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTileData, SrcTileData, mode, quantPre>();
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Nz = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable =
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<typename DstTileData::DType, float>)&&(DstTileData::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    if constexpr (enableNz2Nz) {
        validRow = SrcTileData::Rows;
        if constexpr ((mode == AccToVecMode::SingleModeVec0 || mode == AccToVecMode::SingleModeVec1)) {
            if constexpr (std::is_same_v<typename DstTileData::DType, float>) {
                constexpr int32_t align = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
                validCol = CeilDivision(validCol, align) * align;
            } else {
                validCol = CeilDivision(validCol, c0Size) * c0Size;
            }
        } else if constexpr (mode == AccToVecMode::DualModeSplitM) {
            validCol = CeilDivision(validCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        } else {
            validCol = CeilDivision(validCol, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
        }
    }
    auto srcStride = (validRow + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);

    if (!enableNz2Nd) {
        copy_matrix_cc_to_ubuf(dstAddr, srcData, 0, validCol, validRow, 0, 0, CRMODE_NONE);
        return;
    }
    // 使用循环来搬运, 实现NZ2ND
    for (size_t i = 0; i < validRow; i++) {
        auto srcData_ = srcData + i * 32;
        auto dstAddr_ = dstAddr + i * validCol;
        uint32_t nBlock = (validCol * 2 - 1) / 32 + 1;
        uint32_t stride = (validRow - 1);
        copy_matrix_cc_to_ubuf(dstAddr_, srcData_, 0, nBlock, 1, stride, 0, CRMODE_NONE);
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TMovToVec(DstTileData &dst, SrcTileData &src)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    constexpr unsigned srcStride = SrcTileData::RowStride;
    constexpr unsigned dstStride = DstTileData::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TCopy<DstTileData, SrcTileData, blockSizeElem, srcStride, dstStride>(dst.data(), src.data(), validRow, validCol);
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToCb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                              uint16_t validRow, uint16_t validCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
    __cc__ SrcType *srcAddr = (__cc__ SrcType *)__cce_get_tile_ptr(src);
    __cbuf__ DstType *dstAddr = (__cbuf__ DstType *)__cce_get_tile_ptr(dst);

    constexpr uint32_t dstStride_dst_D = DstTileData::Rows;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    validCol = CeilDivision(validCol, c0Size) * c0Size;
    copy_matrix_cc_to_cbuf(dstAddr, srcAddr, 0, validCol, SrcTileData::Rows, dstStride_dst_D, srcStride, 0, QuantPre,
                           reluMode, false, false);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToLeft(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                                dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToACompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                               dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToRight(DstTileData &dst, SrcTileData &src)
{
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                                dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBCompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                               dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_CONVTILE_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(),
                                                  dst.GetValidCol(), 0, 0);
    }
}
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_TILE_IMPL(DstTileData &dst, SrcTileData &src)
{
    static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
                  "TMov: The shape of src needs to be the same as that of dst.");
    static_assert((SrcTileData::Loc == TileType::Mat &&
                   (DstTileData::Loc == TileType::Left || DstTileData::Loc == TileType::Right ||
                    DstTileData::Loc == TileType::Bias || DstTileData::Loc == TileType::Scaling)) ||
                      (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) ||
                      (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Acc),
                  "TMov: Invalid TileType.");
    if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Left) {
        TMovToLeft<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Right) {
        TMovToRight<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec) {
        TMovToVec<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TMovCcToCb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
    }
}
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TMOV_CONVTILE_IMPL(dst, src);
    } else {
        TMOV_TILE_IMPL(dst, src);
    }
}
// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    // CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
                  "Quant is not support in dual Dst Mode.");
    static_assert((DstTileData::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TMovCcToUb<DstTileData, SrcTileData, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    (void)fp;
    // SetFPC<FpTileData>(fp.data());
    TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif // TMOV_HPP
