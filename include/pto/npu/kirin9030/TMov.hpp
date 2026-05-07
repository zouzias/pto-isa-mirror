/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "TPartAdd.hpp"

namespace pto {
template <typename DstTile, typename SrcTile>
__tf__ AICORE void TMovToBt(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using DstType = typename DstTile::DType;
    using SrcType = typename SrcTile::DType;
    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                      (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>),
                  "Fix: TMOV: Bias data type only supports int32_t or half.");
    constexpr const int BIAS_TABLE_UNIT_ELEM = 16;
    static_assert(SrcTile::Rows == 1, "Fix: TMov: When TileType is Bias, row must be 1.");
    static_assert(DstTile::Cols % BIAS_TABLE_UNIT_ELEM == 0,
                  "Fix: TMov: When TileType is Bias, col must be aligned to 16.");
    static_assert(DstTile::Cols * sizeof(DstType) <= PTO_BIAS_SIZE_BYTES,
                  "Fix: TMov: The memory occupation of BiasTile exceeds 1.0KB bias table size.");

    __cbuf__ SrcType *srcP = (__cbuf__ SrcType *)(src);
    uint64_t dstAddr = (uint64_t)dst;
    constexpr uint16_t burstLen = SrcTile::Numel * sizeof(SrcType) / BLOCK_BYTE_SIZE;

    copy_cbuf_to_bt(dstAddr, srcP, false /* convControl */, 1 /* nBurst */, burstLen, 0 /* srcGap */, 0 /* dstGap */);
}

template <typename DstTile, typename SrcTile>
__tf__ AICORE void TMovToFb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
{
    using SrcType = typename SrcTile::DType;
    using DstType = typename DstTile::DType;
    constexpr const int FIXPIPE_BUFFER_UNIT = 128;
    constexpr const int FIXPIPE_BUFFER_SIZE = 7 * 1024;
    static_assert(SrcTile::Rows == 1, "TMov: When TileType is Scaling, row must be 1.");
    static_assert(DstTile::Cols * sizeof(DstType) % FIXPIPE_BUFFER_UNIT == 0,
                  "TMov: When TileType is Scaling, col * sizeof(Dtype) must be aligned to 128.");
    static_assert(DstTile::Cols * sizeof(DstType) <= FIXPIPE_BUFFER_SIZE,
                  "TMov: The memory occupation of FbTile exceeds 7.0KB fixpipe buffer size.");

    __cbuf__ SrcType *srcP = (__cbuf__ SrcType *)(src);
    __fbuf__ DstType *dstP = (__fbuf__ DstType *)(dst);

    constexpr uint16_t burstLen = SrcTile::Numel * sizeof(SrcType) / FIXP_BURST_UNIT_LEN;

    copy_cbuf_to_fbuf(dstP, srcP, 1 /* nBurst */, burstLen, 0 /* srcGap */, 0 /* dstGap */);
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, QuantMode_t quantPre>
PTO_INTERNAL constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == AccToVecMode::DualModeSplitM || mode == AccToVecMode::DualModeSplitN) {
        static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
        static_assert((!(!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox)),
                      "Dual Dst Mode is not support in nz2dn.");
        return ((mode == AccToVecMode::DualModeSplitM) ? 1 : 2);
    }
    return 0;
}

PTO_INTERNAL void SetLoop3Para()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0x0;
    constexpr uint16_t srcNdStride = 0x0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL constexpr uint32_t GetTmovAccDstStride()
{
    if constexpr (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox) {
        return DstTile::Cols;
    } else if constexpr (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox) {
        return DstTile::Rows;
    }
    constexpr bool channelSplitEnable =
        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<typename DstTile::DType, float>)&&(DstTile::SFractalSize == 512);
    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (DstTile::SFractalSize == 1024) ?
                                    2 * C0_SIZE_BYTE / sizeof(typename DstTile::DType) :
                                    C0_SIZE_BYTE / sizeof(typename DstTile::DType);
    return DstTile::Rows * c0Size;
}

template <typename DstTile, typename SrcTile, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToCb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                              uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool enableNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable =
        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<typename DstTile::DType, float>)&&(DstTile::SFractalSize == 512);
    if constexpr (enableNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr (std::is_same_v<typename DstTile::DType, float>) {
            constexpr int32_t alignSize = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
            validCol = CeilDivision(validCol, alignSize) * alignSize;
        } else {
            validCol = CeilDivision(validCol, c0Size) * c0Size;
        }
    }

    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    if constexpr (enableNz2Nd || enableNz2Dn) {
        SetLoop3Para();
    }
    if constexpr (enableNz2Dn) {
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }

    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)(src);

    copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, SrcTile::Rows, 0, 0, QuantPre, reluMode,
                           channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false, false,
                           enableNz2Dn);
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, QuantMode_t quantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToUb(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                              uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTile, SrcTile, mode, quantPre>();
    constexpr bool enableNz2Nd = (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Nz = (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable =
        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<typename DstTile::DType, float>)&&(DstTile::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
    static_assert(((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
                  "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    if constexpr (enableNz2Nz) {
        validRow = SrcTile::Rows;
        if constexpr ((mode == AccToVecMode::SingleModeVec0 || mode == AccToVecMode::SingleModeVec1)) {
            if constexpr (std::is_same_v<typename DstTile::DType, float>) {
                constexpr int32_t alignSize = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
                validCol = CeilDivision(validCol, alignSize) * alignSize;
            } else {
                validCol = CeilDivision(validCol, c0Size) * c0Size;
            }
        } else if constexpr (mode == AccToVecMode::DualModeSplitM) {
            validCol = CeilDivision(validCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        } else {
            validCol = CeilDivision(validCol, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
        }
    }
    if constexpr (enableNz2Dn) {
        SetLoop3Para();
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    } else if constexpr (enableNz2Nd) {
        SetLoop3Para();
    }

    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src);
    copy_matrix_cc_to_ub(dstAddr, srcData, 0, validCol, validRow, dstStride, SrcTile::Rows, 0, 0, quantPre, reluMode,
                         channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false, false,
                         enableNz2Dn);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL constexpr void CommonCheck()
{
    using T = typename DstTile::DType;
    using U = typename SrcTile::DType;
    static_assert(std::is_same_v<T, U>, "Fix: TMov Destination and Source tile data types must be the same.");

    if constexpr (DstTile::Loc == TileType::Left) {
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
                      "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(DstTile::SFractal == SLayout::RowMajor && !DstTile::isRowMajor,
                      "Fix: TMov: Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    }
    if constexpr (DstTile::Loc == TileType::Right) {
        static_assert(std::is_same_v<T, half> || std::is_same_v<T, int8_t>,
                      "Fix: TMov: Unsupported data type! Supported types: int8_t, half");
        static_assert(DstTile::SFractal == SLayout::ColMajor && DstTile::isRowMajor,
                      "Fix: TMov: Dst fractal format should be (BFractal: RowMajor, SFractal: ColMajor).");
    }

    static_assert((SrcTile::SFractal == SLayout::ColMajor && SrcTile::isRowMajor) ||
                      (SrcTile::SFractal == SLayout::RowMajor && !SrcTile::isRowMajor) || (SrcTile::isRowMajor),
                  "TMov: SrcTile Invalid Fractal.");
}

template <typename T, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovToVecNd2Nz(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                        uint32_t validRow, uint32_t validCol,
                                        unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, float>::value) ||
                      (std::is_same<T, int32_t>::value) || (std::is_same<T, int8_t>::value),
                  "Dst and src must be float/int32_t/half/int8_t.");
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr int32_t srcRow = SrcTile::Rows;
    constexpr int32_t srcCol = SrcTile::Cols;
    constexpr int32_t srcByteSize = srcRow * srcCol * sizeof(T);
    constexpr int32_t dstByteSize = DstTile::Rows * DstTile::Cols * sizeof(T);

    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr bool isOptForConflict = (dstByteSize >= (srcRow + 1) * srcCol * sizeof(T)) ? true : false;
    uint32_t alignRow = (validRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    uint32_t blockStride = isOptForConflict ? ((alignRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
                                              (alignRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t virtualRow = isOptForConflict ? alignRow + 1 : alignRow;
    uint32_t repeatStride = 1;
    uint16_t innerLoopNum = validRow - 1;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        MaskReg pReg;
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
            uint32_t count =
                ((j + 1) * elementsPerRepeat >= validCol ? (validCol - j * elementsPerRepeat) : elementsPerRepeat);
            pReg = CreatePredicate<T>(count);
            for (uint16_t i = 0; i < (uint16_t)innerLoopNum; ++i) {
                vlds(vreg, srcPtr, (i * SrcTile::RowStride + j * elementsPerRepeat), NORM);
                vsstb(vreg, dstPtr, (blockStride << 16u) | (1 & 0xFFFFU), pReg, POST_UPDATE);
            }
            vlds(vreg, srcPtr, (innerLoopNum * SrcTile::RowStride + j * elementsPerRepeat), NORM);
            repeatStride = (CCE_VL * virtualRow - innerLoopNum * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE;
            vsstb(vreg, dstPtr, (blockStride << 16u) | (repeatStride & 0xFFFFU), pReg, POST_UPDATE);
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TMOV)
    OP_TYPE(element_wise) void TMovVecToVec(typename DstTile::TileDType __out__ dstData,
                                            typename SrcTile::TileDType __in__ srcData, unsigned validRow,
                                            unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    constexpr unsigned nRepeatElem = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg pReg;
        uint32_t sreg;
        uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sreg = (uint32_t)validCol;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                pReg = CreatePredicate<T>(sreg);
                vlds(vreg0, src, i * SrcTile::RowStride + j * nRepeatElem, NORM);
                vsts(vreg0, dst, i * DstTile::RowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMovToVec(DstTile &dst, SrcTile &src)
{
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TMovVecToVec<DstTile, SrcTile>(dst.data(), src.data(), validRow, validCol);
}

template <typename DstTile, typename SrcTile>
AICORE void TMovToLeft(DstTile &dst, SrcTile &src)
{
    CommonCheck<DstTile, SrcTile>();
    if constexpr (SrcTile::Rows == 1 && SrcTile::isRowMajor) {
        TExtractToAVector<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidCol());
    } else if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToATransCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                      dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTile, typename SrcTile>
AICORE void TMovToRight(DstTile &dst, SrcTile &src)
{
    CommonCheck<DstTile, SrcTile>();
    if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToBTransCompact<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, dst.GetValidRow(),
                                                      dst.GetValidCol());
        } else {
            TExtractToB<DstTile, SrcTile, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    if constexpr (SrcTile::Loc == TileType::Mat) {
        static_assert((SrcTile::Rows == DstTile::Rows) && ((SrcTile::Cols == DstTile::Cols)),
                      "TMov: The shape of destination and source tile must be the same.");
        if constexpr (DstTile::Loc == TileType::Bias) {
            TMovToBt<DstTile, SrcTile>(dst.data(), src.data());
        } else if constexpr (DstTile::Loc == TileType::Scaling) {
            TMovToFb<DstTile, SrcTile>(dst.data(), src.data());
        } else if constexpr (DstTile::Loc == TileType::Left) {
            TMovToLeft(dst, src);
        } else if constexpr (DstTile::Loc == TileType::Right) {
            TMovToRight(dst, src);
        } else if constexpr (DstTile::Loc == TileType::ScaleLeft) {
            static_assert(sizeof(DstTile::DType) == 0, "TMov: ScaleLeft tile type is not supported.");
        } else if constexpr (DstTile::Loc == TileType::ScaleRight) {
            static_assert(sizeof(DstTile::DType) == 0, "TMov: ScaleRight tile type is not supported.");
        }
    } else if constexpr (SrcTile::Loc == TileType::Acc) {
        CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
        if constexpr (DstTile::Loc == TileType::Vec) {
            TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(dst.data(),
                                                                                                      src.data(), m, n);
        } else if constexpr (DstTile::Loc == TileType::Mat) {
            TMovCcToCb<DstTile, SrcTile, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
        }
    } else if constexpr (SrcTile::Loc == TileType::Vec) {
        if constexpr (DstTile::Loc == TileType::Vec) {
            if constexpr ((SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) &&
                          (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
                TMovToVecNd2Nz<typename DstTile::DType, DstTile, SrcTile>(dst.data(), src.data(), src.GetValidRow(),
                                                                          src.GetValidCol());
            } else {
                TMovToVec<DstTile, SrcTile>(dst, src);
            }
        } else if constexpr (DstTile::Loc == TileType::Mat) {
            CommonCheck<DstTile, SrcTile>();
            TExtractVecToMat<DstTile, SrcTile>(dst.data(), src.data(), 0, 0, src.GetValidRow(), src.GetValidCol(),
                                               dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename DstTile, typename SrcTile, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename DstTile, typename SrcTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
                  "Quant is not support in dual Dst Mode.");
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename FpTile>
__tf__ PTO_INTERNAL void SetFPC(typename FpTile::TileDType __in__ fp)
{
    __fbuf__ typename FpTile::DType *dstAddrFp = (__fbuf__ typename FpTile::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

template <typename DstTile, typename SrcTile, typename FpTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, FpTile &fp)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    SetFPC<FpTile>(fp.data());
    if constexpr (DstTile::Loc == TileType::Vec) {
        TMovCcToUb<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(dst.data(), src.data(), m, n);
    } else if constexpr (DstTile::Loc == TileType::Mat) {
        TMovCcToCb<DstTile, SrcTile, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTile, typename SrcTile, typename FpTile, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTile &dst, SrcTile &src, FpTile &fp)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
                  "Quant is not support in dual Dst Mode.");
    static_assert((DstTile::Loc == TileType::Vec), "Destination location only support Vec.");
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTile>(fp.data());
    TMovCcToUb<DstTile, SrcTile, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif
