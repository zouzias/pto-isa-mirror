/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include <cassert>
#include "common.hpp"
#include <cmath>

namespace pto {

template <typename DstTileData, typename SrcTileData, QuantModeCPU_t quantMode, bool applyRelu>
PTO_INTERNAL void TExtract_Impl(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol,
                               uint64_t scalar = 0)
{
    assert(dst.GetValidRow() + idxRow <= src.GetValidRow() && dst.GetValidCol() + idxCol <= src.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < dst.GetValidCol(); c++) {
        const size_t subTileDstC = c / DstTileData::InnerCols;
        const size_t innerDstC = c % DstTileData::InnerCols;
        const size_t cSrc = c + idxCol;
        const size_t subTileSrcC = cSrc / SrcTileData::InnerCols;
        const size_t innerSrcC = cSrc % SrcTileData::InnerCols;

        for (size_t r = 0; r < src.GetValidRow(); r++) {
            size_t srcTileIdx;
            size_t dstTileIdx;
            if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(r, c);
            } else {
                const size_t subTileR = r / DstTileData::InnerRows;
                const size_t innerR = r % DstTileData::InnerRows;
                dstTileIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            const size_t rSrc = r + idxRow;

            if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(rDst, cSrc);
            } else {
                const size_t subTileR = rSrc / SrcTileData::InnerRows;
                const size_t innerR = rSrc % SrcTileData::InnerRows;
                srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR, innerR, subTileSrcC, innerSrcC);
            }
            if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                dst.data()[dstTileIdx] = quantize_element<D, S, quantMode, applyRelu>(src.data()[srcTileIdx], scalar);
            } else {
                S val = src.data()[srcTileIdx];
                if constexpr (applyRelu) {
                    val = ReLU(val);
                }
                dst.data()[dstTileIdx] = val;
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, QuantModeCPU_t quantMode, bool applyRelu>
PTO_INTERNAL void TExtract_Impl(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
        assert(dst.GetValidRow() + idxRow <= src.GetValidRow() && dst.GetValidCol() + idxCol <= src.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < dst.GetValidCol(); c++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, c);
        uint64_t quantScalar = static_cast<uint64_t>(fp.data()[quantTileIdx]);

        const size_t subTileDstC = c / DstTileData::InnerCols;
        const size_t innerDstC = c % DstTileData::InnerCols;
        const size_t cSrc = c + idxCol;
        const size_t subTileSrcC = cSrc / SrcTileData::InnerCols;
        const size_t innerSrcC = cSrc % SrcTileData::InnerCols;

        for (size_t r = 0; r < src.GetValidRow(); r++) {
            size_t srcTileIdx;
            size_t dstTileIdx;
            if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(r, c);
            } else {
                const size_t subTileR = r / DstTileData::InnerRows;
                const size_t innerR = r % DstTileData::InnerRows;
                dstTileIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            const size_t rSrc = r + idxRow;

            if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(rDst, cSrc);
            } else {
                const size_t subTileR = rSrc / SrcTileData::InnerRows;
                const size_t innerR = rSrc % SrcTileData::InnerRows;
                srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR, innerR, subTileSrcC, innerSrcC);
            }
            if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                dst.data()[dstTileIdx] = quantize_element<D, S, quantMode, applyRelu>(src.data()[srcTileIdx], scalar);
            } else {
                S val = src.data()[srcTileIdx];
                if constexpr (applyRelu) {
                    val = ReLU(val);
                }
                dst.data()[dstTileIdx] = val;
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cSrc = 0; cSrc < dst.GetValidCol(); ++cSrc) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cSrc + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cSrc);
            S data = src.data()[srcTileIdx];
            if constexpr (reluMode == ReluPreMode::NormalRelu) {
                data = ReLU(data);
            }
            dst.data()[dstTileIdx] = static_cast<D>(data);
        }
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<S, D>();
    constexpr bool apply_relu = reluMode == ReluPreMode::NormalRelu;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cSrc = 0; cSrc < dst.GetValidCol(); ++cSrc) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cSrc + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cSrc);
            dst.data()[dstTileIdx] =
                quantize_element<D, S, quantPre, apply_relu>(src.data()[srcTileIdx], preQuantScalar);
        }
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    constexpr QuantModeCPU_t quantPre = GetVectorPreQuantMode<S, D>();
    constexpr bool apply_relu = reluMode == ReluPreMode::NormalRelu;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cSrc = 0; cSrc < dst.GetValidCol(); ++cSrc) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cSrc + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cSrc);
            const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, cSrc);
            uint64_t quantScalar = static_cast<uint64_t>(fp.data()[quantTileIdx]);
            dst.data()[dstTileIdx] = quantize_element<D, S, quantPre, apply_relu>(src.data()[srcTileIdx], quantScalar);
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, idxRow, idxCol);
}

} // namespace pto
#endif // TEXTRACT_HPP
