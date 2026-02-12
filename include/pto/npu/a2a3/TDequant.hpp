/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDEQUANT_HPP
#define TDEQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/npu/a2a3/TBinSOp.hpp>

namespace pto {
/*
struct FmodOp {
    PTO_INTERNAL static void FmodF16Instr(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1)
    {
        vdiv(dst, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f162s16z((__ubuf__ int16_t *)dst, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_s162f16(dst, (__ubuf__ int16_t *)dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmul(dst, dst, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(dst, src0, dst, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
          unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TFmod(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
                               typename TileData::TileDType __in__ src1, unsigned validRows, unsigned validCols)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        unsigned colsFmodaining = validCols;
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T *s1Next = src1Ptr + i * src1RowStride;
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float16_t>) {
            FmodOp::FmodF16Instr(dstNext, s0Next, s1Next);
        } else {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2, "Fix: TFMOD has unsupported dtype size");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}
*/
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void ConvertToFp32(__ubuf__ typename TileDataDst::DType *dst,
                                          __ubuf__ typename TileDataSrc::DType *src, uint8_t repeatNum,
                                          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                          uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<typename TileDataDst::DType, float>::value &&
                         std::is_same<typename TileDataSrc::DType, int16_t>::value) { // int16 to float32
        vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataDst::DType, float>::value &&
                         std::is_same<typename TileDataSrc::DType, int8_t>::value) { // int8 to float32
        vconv_s82f16(reinterpret_cast<__ubuf__ half *>(dst), src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride); // to half
        pipe_barrier(PIPE_V);
        vconv_f162f32(dst, reinterpret_cast<__ubuf__ half *>(dst), repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride); // to float
        pipe_barrier(PIPE_V);
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned dstRowStride, unsigned srcRowStride>
PTO_INST void ConvertForDequant(__ubuf__ typename TileDataDst::DType * dstPtr,
                                __ubuf__ typename TileDataSrc::DType * srcPtr, unsigned dstValidRows)
{
    uint64_t repeatWidth =
        static_cast<uint64_t>(max(sizeof(typename TileDataDst::DType), sizeof(typename TileDataSrc::DType)));
    unsigned dstRepeatStride =
        repeatWidth == sizeof(typename TileDataDst::DType) ?
            BLOCK_MAX_PER_REPEAT :
            (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataSrc::DType) * sizeof(typename TileDataDst::DType));
    unsigned srcRepeatStride =
        repeatWidth == sizeof(typename TileDataSrc::DType) ?
            BLOCK_MAX_PER_REPEAT :
            (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataDst::DType) * sizeof(typename TileDataSrc::DType));

    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = dstValidCols / elementsPerRepeat; // Complete repeats
    unsigned numRemainPerLine = dstValidCols % elementsPerRepeat; // Remainder elements
    constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);

    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (uint32_t i = 0; i < dstValidRows; i++) {
            if (numLoop > 0) {
                for (uint32_t j = 0; j < numLoop; j++) {
                    ConvertToFp32<TileDataDst, TileDataSrc>(dstPtr + i * dstRowStride + j * elementsPerRepeat * REPEAT_MAX,
                                                    srcPtr + i * srcRowStride + j * elementsPerRepeat * REPEAT_MAX,
                                                    (uint8_t)REPEAT_MAX, 1, 1, (uint16_t)dstRepeatStride,
                                                    (uint16_t)srcRepeatStride);
                }
            }
            if (remainAfterLoop > 0) {
                ConvertToFp32<TileDataDst, TileDataSrc>(dstPtr + i * dstRowStride + numLoop * elementsPerRepeat * REPEAT_MAX,
                                                srcPtr + i * srcRowStride + numLoop * elementsPerRepeat * REPEAT_MAX,
                                                (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRepeatStride,
                                                (uint16_t)srcRepeatStride);
            }
        }
    }

    // Advance pointers to unaligned remainder region
    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    // Process remainder region with partial repeats (requires vector masking)
    if (numRemainPerLine > 0) {
        unsigned numLoop = dstValidRows / REPEAT_MAX;
        unsigned remainAfterLoop = dstValidRows % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                ConvertToFp32<TileDataDst, TileDataSrc>(dstPtr + j * dstRowStride * REPEAT_MAX, srcPtr + j * srcRowStride * REPEAT_MAX,
                                                    (uint8_t)REPEAT_MAX, 1, 1, (uint16_t)dstRowStride / dstNElemPerBlock,
                                                    (uint16_t)srcRowStride / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop > 0) {
            ConvertToFp32<TileDataDst, TileDataSrc>(dstPtr + numLoop * dstRowStride * REPEAT_MAX, srcPtr + numLoop * srcRowStride * REPEAT_MAX,
                                                (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRowStride / dstNElemPerBlock,
                                                (uint16_t)srcRowStride / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara,
          unsigned dstRowStride, unsigned srcRowStride, unsigned scaleRowStride>
__tf__ PTO_INTERNAL void TDequant(typename TileDataDst::TileDType __out__ dst/*fp32*/,
                                  typename TileDataSrc::TileDType __in__ src/*int8_16*/,
                                  typename TileDataPara::TileDType __in__ scale/*fp32*/,
                                  typename TileDataPara::TileDType __in__ offset,
                                  unsigned dstValidRows, unsigned dstValidCols)
{
    // src  int8 ---->  dst fp32
    __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src);
    ConvertForDequant<TileDataDst, TileDataSrc, dstRowStride, srcRowStride>(dstPtr, srcPtr, dstValidRows);
    /* dst_fp32(m,n)  src_int8/int16 (m,n)  scale/offset_fp32(m,1)*/
    /* dst_fp32 = TCVT(src_int8_int16) */
    /* dst = (dst_fp32 - offset) * scale  */
    /* (m,n) = {(m,n) - (m,1)} * (m,1) */
    /* TSUBS  TMULS */
    using T = typename TileDataPara::DType;
    __ubuf__ T *scalePtr = (__ubuf__ T *)__cce_get_tile_ptr(scale);
    __ubuf__ T *offsetPtr = (__ubuf__ T *)__cce_get_tile_ptr(offset);

    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);

    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataDst::RowStride;

    // dst = (dst - offset) * scales
    set_mask_count();
    set_vector_mask(0, dstValidCols);
    for (int i = 0; i < dstValidRows; i++) {
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
        T offsetValue = (T)(*(offsetPtr + i * scaleRowStride));
        T scaleValue = (T)(*(scalePtr + i * scaleRowStride));
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        TBinSInstr<SubSOp<T>, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
        dstPtr, srcPtr, offsetValue, 1, dstValidCols);
        TBinSInstr<MulSOp<T>, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
        dstPtr, srcPtr, scaleValue, 1, dstValidCols);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{ /*dst_fp32(m,n)  src_int8/int16 (m,n)  scale/offset_fp32(m,1)*/
    static_assert(
        std::is_same<TileDataDst::DType, float>::value || std::is_same<TileDataDst::DType, float32_t>::value ||
            std::is_same<TileDataPara::DType, float>::value || std::is_same<TileDataPara::DType, float32_t>::value,
        "Fix: TDEQUANT input tile src and dst currently supports float data types.");
    static_assert(std::is_same<TileDataSrc::DType, int8_t>::value || std::is_same<TileDataSrc, int16_t>::value,
                  "Fix: TDEQUANT input tile scale and offset currently supports int8_t and int16_t data types.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor && TileDataPara::isRowMajor,
                  "Fix: TDEQUANT only support row major layout.");
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    constexpr unsigned scaleRowStride = TileDataPara::RowStride;
    TDequant<TileDataDst, TileDataSrc, TileDataPara, dstRowStride, srcRowStride,
          scaleRowStride>(dst.data(), src.data(), scale.data(), offset.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif









