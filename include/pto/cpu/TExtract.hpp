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
inline T ReLU(T val)
{
    if (val < 0)
        return 0;
    return val;
}

float extract_m1_from_quant(uint64_t quant)
{
    uint32_t m1_bits = static_cast<uint32_t>((quant >> 13) & 0x7FFFF);
    uint32_t sign_bit = (m1_bits >> 18) & 0x1;
    uint32_t exponent = (m1_bits >> 10) & 0xFF;
    uint32_t mantissa = m1_bits & 0x3FF;

    if (exponent == 0 && mantissa == 0)
        return 0.0f;

    float sign_val = (sign_bit == 1) ? -1.0f : 1.0f;
    float mantissa_val = 1.0f + (static_cast<float>(mantissa) / 1024.0f);
    float exponent_val = std::pow(2.0f, static_cast<float>(exponent) - 127.0f);

    return sign_val * mantissa_val * exponent_val;
}

template <typename DstType, typename SrcType, QuantMode_t mode, bool use_relu = false>
DstType quantize_element(SrcType src_val, uint64_t scalar)
{
    float f_scale = extract_m1_from_quant(scalar);
    uint32_t offset = static_cast<uint32_t>((scalar >> 37) & 0x1FF);
    uint32_t sign = static_cast<uint32_t>((scalar >> 46) & 0x1);

    float result_f = static_cast<DstType>(src_val);
    if constexpr (mode == QuantMode_t::QF322B8_PRE || mode == QuantMode_t::VQF322B8_PRE || mode == QuantMode_t::REQ8 ||
                  mode == QuantMode_t::VREQ8) {
        float work = static_cast<float>(src_val) * f_scale;
        float rounded = std::round(work) + offset;
        float min = sign == 1 ? -128.0f : 0.0f;
        float max = sign == 1 ? 127.0f : 255.0f;
        result_f = std::clamp(rounded, min, max);
    } else if constexpr (mode == QuantMode_t::DEQF16 || mode == QuantMode_t::VDEQF16 ||
                         mode == QuantMode_t::QF322F16_PRE) {
        float work = static_cast<float>(src_val) * f_scale;
        result_f = std::clamp(work, -F16_MAX, F16_MAX);
    } else if constexpr (mode == QuantMode_t::QF322BF16_PRE) {
        result_f = static_cast<float>(src_val) * f_scale;
    }
    if constexpr (use_relu)
        result_f = ReLU(result_f);
    return static_cast<DstType>(result_f);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
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
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<S, D>();
    constexpr bool apply_relu = reluMode == ReluPreMode::NormalRelu;

    for (size_t rDst = 0; rDst < dst.GetValidRow(); ++rDst) {
        for (size_t cDst = 0; cDst < dst.GetValidCol(); ++cDst) {
            const size_t srcTileIdx = GetTileElementOffset<SrcTileData>(rDst + idxRow, cDst + idxCol);
            const size_t dstTileIdx = GetTileElementOffset<DstTileData>(rDst, cDst);
            const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, cDst);
            uint64_t quantScalar = static_cast<uint64_t>(fp.data()[quantTileIdx]);
            dst.data()[dstTileIdx] =
                quantize_element<D, S, quantPre, apply_relu>(src.data()[srcTileIdx], quantScalar);
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
