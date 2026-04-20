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

namespace pto {

template <typename T>
T ReLU(T val)
{
    if (val < 0)
        return 0;
    return val;
}

inline float apply_relu(float val)
{
    return val < 0.0f ? 0.0f : val;
}

template <typename DstType, typename SrcType, QuantMode_t mode, bool use_relu = false>
DstType quantize_element(SrcType src_val, uint64_t scalar)
{
    float f_scale;
    uint32_t low_bits = static_cast<uint32_t>(scalar & 0xFFFFFFFF);
    std::memcpy(&f_scale, &low_bits, sizeof(float));

    DstType result_f = static_cast<DstType>(src_val);

    if constexpr (mode == QuantMode_t::QF322B8_PRE || mode == QuantMode_t::VQF322B8_PRE || mode == QuantMode_t::REQ8 ||
                  mode == QuantMode_t::VREQ8) {
        float work = static_cast<float>(src_val) * f_scale;
        if constexpr (use_relu)
            work = ReLU(work);
        float rounded = std::round(work);
        float min = static_cast<float>(std::numeric_limits<DstType>::min());
        float max = static_cast<float>(std::numeric_limits<DstType>::max());
        result_f = std::clamp(rounded, min, max);
    } else if constexpr (mode == QuantMode_t::SHIFTS322S16 || mode == QuantMode_t::VSHIFTS322S16) {
        uint32_t shift_amt = low_bits & 0xFF;
        int32_t shifted = static_cast<int32_t>(src_val) >> shift_amt;
        if constexpr (use_relu)
            shifted = ReLU(shifted);

        int32_t min = -32768;
        int32_t max = 32767;
        result_f = std::clamp(shifted, min, max);
    } else if constexpr (mode == QuantMode_t::DEQF16 || mode == QuantMode_t::VDEQF16) {
        float work = static_cast<float>(src_val) * f_scale;
        if constexpr (use_relu)
            work = apply_relu(work);
        result_f = work;
    } else if constexpr (mode == QuantMode_t::F322F16 || mode == QuantMode_t::F322BF16 ||
                         mode == QuantMode_t::QF322BF16_PRE || mode == QuantMode_t::QF322BF16_PRE) {
        float work = static_cast<float>(src_val) * f_scale;
        if constexpr (use_relu)
            work = apply_relu(work);
        result_f = work;
    } else {
        if constexpr (use_relu)
            result_f = apply_relu(result_f);
    }

    return static_cast<DstType>(result_f);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());
    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
            dst.data()[dstTileIdx] = static_cast<D>(src.data()[srcTileIdx]);
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());

    if constexpr (reluMode == ReluPreMode::NoRelu) {
        TEXTRACT_IMPL<DstTileData, SrcTileData>(dst, src, idxRow, idxCol);
    } else {
        for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
            for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
                const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
                const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
                dst.data()[dstTileIdx] = static_cast<D>((src.data()[srcTileIdx]));
            }
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
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<S, D>();
    constexpr bool apply_relu = reluMode == ReluPreMode::NormalRelu;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
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
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<S, D>();
    constexpr bool apply_relu = reluMode == ReluPreMode::NormalRelu;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
            const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, cDst);
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
