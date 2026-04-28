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

        for (size_t r = 0; r < dst.GetValidRow(); r++) {
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
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(rSrc, cSrc);
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

inline int get_index(int validRow, int validCol, int idxR, int idxC, int dtypeSize) {
    int block_width = 32 / dtypeSize;
    int R = idxR / 16;
    int i = idxR % 16;
    int C = idxC / block_width;
    int j = idxC % block_width;
    int index = R * 16 * validCol + C * 16 * block_width + i * block_width + j;
    return index;
}



template <typename DstTileData, typename SrcTileData, typename FpTileData, QuantModeCPU_t quantMode, bool applyRelu>
PTO_INTERNAL void TExtract_Impl(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{   
    if constexpr (std::is_same_v<typename SrcTileData::DType, int32_t>)
        get(src.data());
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

        for (size_t r = 0; r < dst.GetValidRow(); r++) {
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
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(rSrc, cSrc);
            } else {
                const size_t subTileR = rSrc / SrcTileData::InnerRows;
                const size_t innerR = rSrc % SrcTileData::InnerRows;
                srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR, innerR, subTileSrcC, innerSrcC);
            }

            srcTileIdx = get_index(src.GetValidRow(), src.GetValidCol(), r + idxRow, c + idxCol, sizeof(S));
            dstTileIdx = get_index(dst.GetValidRow(), dst.GetValidCol(), r, c, sizeof(D));

            if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                dst.data()[dstTileIdx] = quantize_element<D, S, quantMode, applyRelu>(src.data()[srcTileIdx], quantScalar);
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
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, QuantModeCPU_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, QuantModeCPU_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, preQuantScalar);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, preQuantScalar);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, FpTileData, quantPre, useRelu>(dst, src, fp, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, FpTileData, quantPre, useRelu>(dst, src, fp, idxRow, idxCol);
}

} // namespace pto
#endif // TEXTRACT_HPP
