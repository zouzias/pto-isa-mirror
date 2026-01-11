/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_T_OP_TEMPLATES_HPP
#define PTO_NPU_A2A3_T_OP_TEMPLATES_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/constants.hpp>
#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>

namespace pto {

/**
 * A2/A3 vector op template helpers (Bin/BinS/Unary/...).
 *
 * Major execution modes used by these helpers:
 * - Count-mode: use `set_mask_count()` + `SetVectorCount(...)` for irregular tails or when repeat/stride limits make
 *   normal-mode inefficient/invalid.
 * - Normal-mode: use repeat-based vector intrinsics with (optional) repeat-strides; tail elements are handled by a
 *   temporary contiguous mask (`SetContMaskByDType`/`SetContinuousMask`).
 *
 * Layout assumptions:
 * - Most helpers target Vec tiles and operate on UB pointers. Higher-level ops should ensure tile layout checks.
 * - "Plus" variants handle different row strides for dst/src tiles (e.g., dst/src have different TileData).
 */

// Heuristic threshold used by some non-contiguous small-shape fallbacks.
constexpr unsigned SMALL_RPT_BINOP = 4;
constexpr unsigned SMALL_RPT_BINS = 4;
constexpr unsigned SMALL_RPT_UNARY = 4;

// ================================================================
// Binary ops (dst = op(src0, src1))
// ================================================================

// Note on naming:
// - "1L" helpers treat the tile as a single contiguous vector (best for continuous shapes).
// - "2L" helpers treat the tile as a 2D region (row-by-row) for non-contiguous shapes.
//
// Note on API:
// - `BinaryInstr<Op, TileData, ...>` is the canonical entry for same-stride tiles (dst/src0/src1 share `TileData`).
// - `BinaryPlusInstr<Op, T, TileDataDst, TileDataSrc0, TileDataSrc1>` handles different strides ("+ variants").

template <typename Op, typename T>
PTO_INTERNAL void Bin1LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL void Bin2LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned tileCols, uint8_t repeatStride>
PTO_INTERNAL void Bin1LNormModeSmall(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    (void)validCol;
    SetContMaskByDType<T>(tileCols);
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride, repeatStride);
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride,
    unsigned tileCols>
PTO_INTERNAL void Bin1LNormMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, headRepeats); // headRepeats can be zero
    if (tailElements) [[unlikely]] {
        unsigned offset = headRepeats * elementsPerRepeat;
        SetContMaskByDType<T>(tailElements);
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1);
        SetFullVecMaskByDType<T>();
    }
    (void)blockSizeElem;
    (void)rowStride;
    (void)tileCols;
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeColVLAlign(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeHead(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(validRow); i++) {
            if (numLoop) [[unlikely]] {
                for (int j = 0; j < static_cast<int>(numLoop); j++) {
                    unsigned offset = i * stride + j * elementsPerRepeat * REPEAT_MAX;
                    Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned offset = i * stride + numLoop * elementsPerRepeat * REPEAT_MAX;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, remainAfterLoop);
            }
        }
    }
    (void)Rows;
    (void)blockSizeElem;
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeTail(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    constexpr bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(numRemainPerLine);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(numLoop); i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    unsigned offset = i * REPEAT_MAX * stride + j * stride;
                    Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1, 1, 1, 1);
                }
            } else {
                unsigned offset = i * REPEAT_MAX * stride;
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, REPEAT_MAX, repeatStride,
                    repeatStride, repeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < remainAfterLoop; j++) {
                    unsigned offset = numLoop * REPEAT_MAX * stride + j * stride;
                    Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1, 1, 1, 1);
                }
            } else {
                unsigned offset = numLoop * REPEAT_MAX * stride;
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, remainAfterLoop, repeatStride,
                    repeatStride, repeatStride);
            }
        }
    } else {
        if constexpr (strideOverFlag) {
            for (uint64_t j = 0; j < remainAfterLoop; j++) {
                unsigned offset = j * stride;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1, 1, 1, 1);
            }
        } else {
            uint8_t repeatStride = stride / blockSizeElem;
            Op::BinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride, repeatStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeRowRpt(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && repeatStride <= (REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            unsigned offset = i * elementsPerRepeat;
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, validRow, repeatStride, repeatStride,
                repeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, validRow, repeatStride, repeatStride,
                repeatStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, numRepeatPerLine);
            unsigned offset = numRepeatPerLine * elementsPerRepeat;
            dstPtr += offset;
            src0Ptr += offset;
            src1Ptr += offset;
        }
        if (numRemainPerLine) {
            Bin2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrFastPath(__ubuf__ typename TileData::DType *dstPtr,
    __ubuf__ typename TileData::DType *src0Ptr, __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow,
    unsigned validCol)
{
    using T = typename TileData::DType;
    constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
    if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
        Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrGeneralPath(__ubuf__ typename TileData::DType *dstPtr,
    __ubuf__ typename TileData::DType *src0Ptr, __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow,
    unsigned validCol)
{
    using T = typename TileData::DType;
    // Continuous check in runtime (merge axis)
    if ((TileData::Cols == validCol) || (validRow == 1)) [[likely]] {
        unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
        bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
        if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) [[unlikely]] {
            Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else {
            Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        }
    } else { // Non-contiguous
        constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
        if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINOP)) {
            Bin2LCountMode<Op, T, rowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
            if ((validCol % elementsPerRepeat) > 0) {
                Bin2LCountMode<Op, T, rowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            } else {
                Bin2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(
                    dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            }
        } else {
            Bin2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstr(__ubuf__ typename TileData::DType *dstPtr, __ubuf__ typename TileData::DType *src0Ptr,
    __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow, unsigned validCol)
{
    // `TileData` encodes both the element type (`DType`) and the compile-time tile shape/strides.
    // Keep this API minimal: callers should not pass a separate `T` (it would be redundant and can desync from TileData).
    using T = typename TileData::DType;
    // Small shape optimization
    if constexpr ((TileData::Rows <= pto::REPEAT_MAX) && (TileData::Cols < elementsPerRepeat)) {
        constexpr uint8_t repeatStride = rowStride / blockSizeElem;
        Bin1LNormModeSmall<Op, T, elementsPerRepeat, TileData::Cols, repeatStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        return;
    }
    // Continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        BinaryInstrFastPath<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        BinaryInstrGeneralPath<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}

// ================================================================
// Binary plus ops (dst/src can have different row strides)
// ================================================================

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
    unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeHead(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned validRow,
    unsigned rptPerLine)
{
    if (rptPerLine > 0) {
        unsigned numLoop = rptPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = rptPerLine % REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(validRow); i++) {
            if (numLoop) [[unlikely]] {
                for (int j = 0; j < static_cast<int>(numLoop); j++) {
                    unsigned dstOffset = i * dstStride + j * elemPerRpt * REPEAT_MAX;
                    unsigned src0Offset = i * src0Stride + j * elemPerRpt * REPEAT_MAX;
                    unsigned src1Offset = i * src1Stride + j * elemPerRpt * REPEAT_MAX;
                    Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned offset = i * dstStride + numLoop * elemPerRpt * REPEAT_MAX;
                unsigned src0Offset = i * src0Stride + numLoop * elemPerRpt * REPEAT_MAX;
                unsigned src1Offset = i * src1Stride + numLoop * elemPerRpt * REPEAT_MAX;
                Op::BinInstr(dst + offset, src0 + src0Offset, src1 + src1Offset, remainAfterLoop);
            }
        }
    }
    (void)elemPerBlk;
}

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
    unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeTail(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned validRow,
    unsigned remain)
{
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr bool src0StrideOverFlag = (src0Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool src1StrideOverFlag = (src1Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool dstStrideOverFlag = (dstStride / elemPerBlk > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(remain);
    for (int i = 0; i < static_cast<int>(numLoop); i++) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                unsigned src0Offset = i * REPEAT_MAX * src0Stride + j * src0Stride;
                unsigned src1Offset = i * REPEAT_MAX * src1Stride + j * src1Stride;
                unsigned dstOffset = i * REPEAT_MAX * dstStride + j * dstStride;
                Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, 1, 1, 1, 1);
            }
        } else {
            unsigned src0Offset = i * REPEAT_MAX * src0Stride;
            unsigned src1Offset = i * REPEAT_MAX * src1Stride;
            unsigned dstOffset = i * REPEAT_MAX * dstStride;
            uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
            uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
            uint8_t dstBlkPerLine = dstStride / elemPerBlk;
            Op::BinInstr(
                dst + dstOffset, src0 + src0Offset, src1 + src1Offset, REPEAT_MAX, dstBlkPerLine, src0BlkPerLine,
                src1BlkPerLine);
        }
    }
    remainAfterLoop = validRow % REPEAT_MAX;
    if (remainAfterLoop) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                unsigned src0Offset = numLoop * REPEAT_MAX * src0Stride + j * src0Stride;
                unsigned src1Offset = numLoop * REPEAT_MAX * src1Stride + j * src1Stride;
                unsigned dstOffset = numLoop * REPEAT_MAX * dstStride + j * dstStride;
                Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, 1, 1, 1, 1);
            }
        } else {
            unsigned dstOffset = numLoop * REPEAT_MAX * dstStride;
            unsigned src0Offset = numLoop * REPEAT_MAX * src0Stride;
            unsigned src1Offset = numLoop * REPEAT_MAX * src1Stride;
            uint8_t dstBlkPerLine = dstStride / elemPerBlk;
            uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
            uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
            Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, remainAfterLoop, dstBlkPerLine,
                src0BlkPerLine, src1BlkPerLine);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
    unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeRowRpt(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned validRow,
    unsigned validCol)
{
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    Bin2LNormModeHead<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>(dst, src0, src1, validRow,
        rptPerLine);
    if (remain) {
        unsigned offset = rptPerLine * elemPerRpt;
        dst += offset;
        src0 += offset;
        src1 += offset;
        Bin2LNormModeTail<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>(dst, src0, src1, validRow,
            remain);
    }
}

template <typename Op, typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void BinaryPlusInstr(__ubuf__ typename TileDataDst::DType *dst, __ubuf__ typename TileDataSrc0::DType *src0,
    __ubuf__ typename TileDataSrc1::DType *src1, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elemPerRpt = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned elemPerBlk = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned src0Stride = TileDataSrc0::RowStride;
    constexpr unsigned src1Stride = TileDataSrc1::RowStride;

    Bin2LNormModeRowRpt<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>(dst, src0, src1, validRow,
        validCol);
}

// ================================================================
// Binary scalar ops (dst = op(src0, scalar))
// ================================================================

template <typename Op, typename T>
PTO_INTERNAL void BinS1LCountMode(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::BinSInstr(dst, src0, src1, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL void BinS2LCountMode(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinSInstr(dst + offset, src0 + offset, src1, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride, unsigned Cols>
PTO_INTERNAL void BinS1LNormMode(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinSInstr(dst, src0, src1, headRepeats);
    if (tailElements) [[unlikely]] {
        unsigned offset = headRepeats * elementsPerRepeat;
        SetContMaskByDType<T>(tailElements);
        Op::BinSInstr(dst + offset, src0 + offset, src1, 1);
        SetFullVecMaskByDType<T>();
    }
    (void)blockSizeElem;
    (void)rowStride;
    (void)Cols;
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
PTO_INTERNAL void BinS2LNormModeColVLAlign(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow,
    unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (uint32_t i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinSInstr(dst + offset, src0 + offset, src1, headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void BinS2LNormModeHead(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow,
    unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(validRow); i++) {
            if (numLoop) [[unlikely]] {
                for (int j = 0; j < static_cast<int>(numLoop); j++) {
                    unsigned offset = i * stride + j * elementsPerRepeat * REPEAT_MAX;
                    Op::BinSInstr(dst + offset, src0 + offset, src1, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned offset = i * stride + numLoop * elementsPerRepeat * REPEAT_MAX;
                Op::BinSInstr(dst + offset, src0 + offset, src1, remainAfterLoop);
            }
        }
    }
    (void)Rows;
    (void)blockSizeElem;
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void BinS2LNormModeTail(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow,
    unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    const bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(numRemainPerLine);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(numLoop); i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    unsigned offset = i * REPEAT_MAX * stride + j * stride;
                    Op::BinSInstr(dst + offset, src0 + offset, src1, 1, 1, 1);
                }
            } else {
                unsigned offset = i * REPEAT_MAX * stride;
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinSInstr(dst + offset, src0 + offset, src1, REPEAT_MAX, repeatStride, repeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < remainAfterLoop; j++) {
                    unsigned offset = numLoop * REPEAT_MAX * stride + j * stride;
                    Op::BinSInstr(dst + offset, src0 + offset, src1, 1, 1, 1);
                }
            } else {
                unsigned offset = numLoop * REPEAT_MAX * stride;
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinSInstr(dst + offset, src0 + offset, src1, remainAfterLoop, repeatStride, repeatStride);
            }
        }
    } else {
        if constexpr (strideOverFlag) {
            for (uint64_t j = 0; j < remainAfterLoop; j++) {
                unsigned offset = j * stride;
                Op::BinSInstr(dst + offset, src0 + offset, src1, 1, 1, 1);
            }
        } else {
            uint8_t repeatStride = stride / blockSizeElem;
            Op::BinSInstr(dst, src0, src1, validRow, repeatStride, repeatStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void BinS2LNormModeRowRpt(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = stride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && repeatStride <= (REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            unsigned offset = i * elementsPerRepeat;
            Op::BinSInstr(dst + offset, src0 + offset, src1, validRow, repeatStride, repeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::BinSInstr(dst + offset, src0 + offset, src1, validRow, repeatStride, repeatStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            BinS2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, stride>(dst, src0, src1, validRow,
                numRepeatPerLine);
            unsigned offset = numRepeatPerLine * elementsPerRepeat;
            dst += offset;
            src0 += offset;
        }
        if (numRemainPerLine) {
            BinS2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, stride>(dst, src0, src1, validRow,
                numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void TBinSInstr(__ubuf__ typename TileData::DType __out__ *dst, __ubuf__ typename TileData::DType __in__ *src0,
    typename TileData::DType __in__ src1, unsigned validRow, unsigned validCol)
{
    // Scalar binary ops follow the same shape/stride rules as normal binary ops, but with `src1` as an immediate scalar.
    // For different dst/src strides, use `TBinSPlusInstr<...>` (the "+ variant").
    using T = typename TileData::DType;
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
        if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
            BinS1LCountMode<Op, T>(dst, src0, src1, validRow, validCol);
        } else {
            BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(dst, src0, src1, validRow,
                validCol);
        }
    } else {
        if ((TileData::Cols == validCol) || (validRow == 1)) [[likely]] {
            unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
            bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
            if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) [[unlikely]] {
                BinS1LCountMode<Op, T>(dst, src0, src1, validRow, validCol);
            } else {
                BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(dst, src0, src1, validRow,
                    validCol);
            }
        } else {
            constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
            if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINS)) {
                BinS2LCountMode<Op, T, rowStride>(dst, src0, src1, validRow, validCol);
            } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
                unsigned tailElements = validCol % elementsPerRepeat;
                if (tailElements) {
                    BinS2LCountMode<Op, T, rowStride>(dst, src0, src1, validRow, validCol);
                } else {
                    BinS2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(dst, src0, src1, validRow, validCol);
                }
            } else {
                BinS2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1,
                    validRow, validCol);
            }
        }
    }
}

// ================================================================
// Binary scalar plus ops (dst/src can have different row strides)
// ================================================================

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinSPlusHead(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned rptPerLine)
{
    if (rptPerLine > 0) {
        unsigned numLoop = rptPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = rptPerLine % REPEAT_MAX;
        for (int i = 0; i < static_cast<int>(validRow); i++) {
            if (numLoop) [[unlikely]] {
                for (int j = 0; j < static_cast<int>(numLoop); j++) {
                    unsigned dstOffset = i * dstStride + j * elemPerRpt * REPEAT_MAX;
                    unsigned srcOffset = i * srcStride + j * elemPerRpt * REPEAT_MAX;
                    Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned dstOffset = i * dstStride + numLoop * elemPerRpt * REPEAT_MAX;
                unsigned srcOffset = i * srcStride + numLoop * elemPerRpt * REPEAT_MAX;
                Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, remainAfterLoop);
            }
        }
    }
    (void)elemPerBlk;
}

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinSPlusTail(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned remain)
{
    const bool strideOverFlag =
        ((dstStride / elemPerBlk > REPEAT_STRIDE_MAX) || (srcStride / elemPerBlk > REPEAT_STRIDE_MAX));
    unsigned dstOffset = 0;
    unsigned srcOffset = 0;
    SetContMaskByDType<T>(remain);
    if constexpr (strideOverFlag) {
        for (int i = 0; i < static_cast<int>(validRow); i++) {
            Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, 1, 1, 1);
            dstOffset += dstStride;
            srcOffset += srcStride;
        }
    } else {
        unsigned numLoop = validRow / REPEAT_MAX;
        unsigned remainAfterLoop = validRow % REPEAT_MAX;
        constexpr uint8_t dstRptStride = dstStride / elemPerBlk;
        constexpr uint8_t srcRptStride = srcStride / elemPerBlk;
        for (int i = 0; i < static_cast<int>(numLoop); i++) {
            Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, REPEAT_MAX, dstRptStride, srcRptStride);
            dstOffset += REPEAT_MAX * dstStride;
            srcOffset += REPEAT_MAX * srcStride;
        }

        if (remainAfterLoop) {
            Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, remainAfterLoop, dstRptStride, srcRptStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void BinSPlusRowRpt(
    __ubuf__ T *dst, __ubuf__ T *src, T scalar, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr unsigned elemPerBlk = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcRptStride = srcStride / elemPerBlk;
    constexpr unsigned dstRptStride = dstStride / elemPerBlk;
    constexpr bool condRowRpt = ((TileDataDst::Rows <= pto::REPEAT_MAX) && (dstRptStride <= REPEAT_STRIDE_MAX) &&
        (TileDataSrc::Rows <= pto::REPEAT_MAX) && (srcRptStride <= REPEAT_STRIDE_MAX));
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    unsigned offset = 0;
    if constexpr (condRowRpt) {
        for (unsigned i = 0; i < rptPerLine; i++) {
            Op::BinSInstr(dst + offset, src + offset, scalar, validRow, dstRptStride, srcRptStride);
            offset += elemPerRpt;
        }

        if (remain) {
            SetContMaskByDType<T>(remain);
            Op::BinSInstr(dst + offset, src + offset, scalar, validRow, dstRptStride, srcRptStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        BinSPlusHead<Op, T, elemPerRpt, elemPerBlk, dstStride, srcStride>(dst, src, scalar, validRow, rptPerLine);
        offset = rptPerLine * elemPerRpt;
        dst += offset;
        src += offset;
        if (remain) {
            BinSPlusTail<Op, T, elemPerRpt, elemPerBlk, dstStride, srcStride>(dst, src, scalar, validRow, remain);
        }
    }
}

template <typename Op, typename T, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TBinSPlusInstr(__ubuf__ T *dst, __ubuf__ T *src, T scalar, unsigned validRow, unsigned validCol)
{
    BinSPlusRowRpt<Op, T, TileDataDst, TileDataSrc>(dst, src, scalar, validRow, validCol);
}

// ================================================================
// Unary ops (dst = op(src))
// ================================================================

template <typename Op, typename T>
PTO_INTERNAL void Unary1LCountMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::UnaryInstr(dstPtr, srcPtr, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL void Unary2LCountMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat>
PTO_INTERNAL void Unary1LNormMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;

    Op::UnaryInstr(dstPtr, srcPtr, headRepeats); // headRepeats can be zero
    if (tailElements) [[unlikely]] {
        unsigned offset = headRepeats * elementsPerRepeat;
        SetContMaskByDType<T>(tailElements);
        Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1);
        SetFullVecMaskByDType<T>();
    }
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
PTO_INTERNAL void Unary2LNormModeColVLAlign(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::UnaryInstr(dstPtr + offset, srcPtr + offset, headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Unary2LNormModeHead(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            if (numLoop) [[unlikely]] {
                for (unsigned j = 0; j < numLoop; j++) {
                    unsigned offset = i * rowStride + j * elementsPerRepeat * REPEAT_MAX;
                    Op::UnaryInstr(dstPtr + offset, srcPtr + offset, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned offset = i * rowStride + numLoop * elementsPerRepeat * REPEAT_MAX;
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, remainAfterLoop);
            }
        }
    }
    (void)Rows;
    (void)blockSizeElem;
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Unary2LNormModeTail(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    constexpr bool strideOverFlag = (rowStride / blockSizeElem > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(numRemainPerLine);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (unsigned i = 0; i < numLoop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    unsigned offset = i * REPEAT_MAX * rowStride + j * rowStride;
                    Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1, 1, 1);
                }
            } else {
                unsigned offset = i * REPEAT_MAX * rowStride;
                uint8_t repeatStride = rowStride / blockSizeElem;
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, REPEAT_MAX, repeatStride, repeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                unsigned offset = numLoop * REPEAT_MAX * rowStride + j * rowStride;
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1, 1, 1);
            }
        } else {
            unsigned offset = numLoop * REPEAT_MAX * rowStride;
            uint8_t repeatStride = rowStride / blockSizeElem;
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, remainAfterLoop, repeatStride, repeatStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Unary2LNormModeRowRpt(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            unsigned offset = i * elementsPerRepeat;
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, validRow, repeatStride, repeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, validRow, repeatStride, repeatStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Unary2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, srcPtr, validRow, numRepeatPerLine);
            unsigned offset = numRepeatPerLine * elementsPerRepeat;
            dstPtr += offset;
            srcPtr += offset;
        }
        if (numRemainPerLine) {
            Unary2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, srcPtr, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void UnaryInstr(
    __ubuf__ typename TileData::DType *dstPtr, __ubuf__ typename TileData::DType *srcPtr, unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    // Fast path: compile-time contiguous layout.
    // Prefer repeat-based mode (better throughput), fall back to count-mode when `repeats` overflows.
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        if constexpr (totalRepeats > pto::REPEAT_MAX) {
            Unary1LCountMode<Op, T>(dstPtr, srcPtr, validRow, validCol);
        } else {
            Unary1LNormMode<Op, T, elementsPerRepeat>(dstPtr, srcPtr, validRow, TileData::Cols);
        }
    } else {
        // General path:
        // - Runtime contiguous check (merge-axis): handle dynamic tails by mask.
        // - Otherwise treat as 2D non-contiguous with row-stride handling.
        if ((TileData::Cols == validCol) || (validRow == 1)) [[likely]] {
            unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
            if (totalRepeats > pto::REPEAT_MAX) [[unlikely]] {
                Unary1LCountMode<Op, T>(dstPtr, srcPtr, validRow, validCol);
            } else {
                Unary1LNormMode<Op, T, elementsPerRepeat>(dstPtr, srcPtr, validRow, validCol);
            }
        } else { // Non-contiguous
            constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
            if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_UNARY)) {
                Unary2LCountMode<Op, T, rowStride>(dstPtr, srcPtr, validRow, validCol);
            } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
                unsigned tailElements = validCol % elementsPerRepeat;
                if (tailElements) {
                    Unary2LCountMode<Op, T, rowStride>(dstPtr, srcPtr, validRow, validCol);
                } else {
                    Unary2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(dstPtr, srcPtr, validRow, validCol);
                }
            } else {
                Unary2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                    dstPtr, srcPtr, validRow, validCol);
            }
        }
    }
}

template <typename T>
using unaryFuncPtr = void (*)(__ubuf__ T *, __ubuf__ T *, uint8_t, uint16_t, uint16_t, uint8_t, uint8_t);

// Wrap a raw vector unary intrinsic into the `Op::UnaryInstr(...)` interface expected by helpers above.
template <typename T, unaryFuncPtr<T> funcPtr>
struct UnaryOperation {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeats)
    {
        funcPtr(dst, src, repeats, 1, 1, 8, 8);
    }
    PTO_INTERNAL static void UnaryInstr(
        __ubuf__ T *dst, __ubuf__ T *src, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        funcPtr(dst, src, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename TileData, unaryFuncPtr<typename TileData::DType> funcPtr, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned rowStride>
__tf__ AICORE void TUnaryOp(
    typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src, unsigned validRow, unsigned validCol)
{
    __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    UnaryInstr<UnaryOperation<typename TileData::DType, funcPtr>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
        dstPtr, srcPtr, validRow, validCol);
}

// ================================================================
// Unary plus ops (dst/src can have different row strides)
// ================================================================

template <typename Op, typename T, unsigned elemPerRpt, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void UnaryPlusHead(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned rptPerLine)
{
    if (rptPerLine > 0) {
        unsigned numLoop = rptPerLine / REPEAT_MAX;
        unsigned remain = rptPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            if (numLoop) [[unlikely]] {
                for (unsigned j = 0; j < numLoop; j++) {
                    unsigned dstOffset = i * dstRowStride + j * elemPerRpt * REPEAT_MAX;
                    unsigned srcOffset = i * srcRowStride + j * elemPerRpt * REPEAT_MAX;
                    Op::UnaryInstr(dst + dstOffset, src + srcOffset, REPEAT_MAX);
                }
            }
            if (remain) {
                unsigned dstOffset = i * dstRowStride + numLoop * elemPerRpt * REPEAT_MAX;
                unsigned srcOffset = i * srcRowStride + numLoop * elemPerRpt * REPEAT_MAX;
                Op::UnaryInstr(dst + dstOffset, src + srcOffset, remain);
            }
        }
    }
}

template <typename Op, typename T, unsigned elemPerBlk, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void UnaryPlusTail(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned remainElem)
{
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr uint8_t dstRptStride = dstRowStride / elemPerBlk;
    constexpr uint8_t srcRptStride = srcRowStride / elemPerBlk;
    // If either repeat-stride exceeds the ISA limit, fall back to 1-repeat per-row execution.
    constexpr bool strideOverFlag =
        ((dstRowStride / elemPerBlk > REPEAT_STRIDE_MAX) || (srcRowStride / elemPerBlk > REPEAT_STRIDE_MAX));

    SetContMaskByDType<T>(remainElem);
    for (unsigned i = 0; i < numLoop; i++) {
        if constexpr (strideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                unsigned dstOffset = i * REPEAT_MAX * dstRowStride + j * dstRowStride;
                unsigned srcOffset = i * REPEAT_MAX * srcRowStride + j * srcRowStride;
                Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, 1, 1, 1);
            }
        } else {
            unsigned dstOffset = i * REPEAT_MAX * dstRowStride;
            unsigned srcOffset = i * REPEAT_MAX * srcRowStride;
            Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, REPEAT_MAX, dstRptStride, srcRptStride);
        }
    }

    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                unsigned dstOffset = numLoop * REPEAT_MAX * dstRowStride + j * dstRowStride;
                unsigned srcOffset = numLoop * REPEAT_MAX * srcRowStride + j * srcRowStride;
                Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, 1, 1, 1);
            }
        } else {
            unsigned dstOffset = numLoop * REPEAT_MAX * dstRowStride;
            unsigned srcOffset = numLoop * REPEAT_MAX * srcRowStride;
            Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, remainAfterLoop, dstRptStride, srcRptStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename T, typename Op, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void UnaryPlusInstr(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elemPerBlk = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = dstRowStride / elemPerBlk;
    constexpr unsigned srcStride = srcRowStride / elemPerBlk;

    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    unsigned offset = 0;
    constexpr bool condRowRpt =
        ((TileDataDst::Rows <= pto::REPEAT_MAX) && (dstStride <= REPEAT_STRIDE_MAX) && (TileDataSrc::Rows <= pto::REPEAT_MAX) &&
            (srcStride <= REPEAT_STRIDE_MAX));

    // If both repeat-strides are valid, we can use a single vector instruction with repeat-strides to
    // walk rows (dst/src can have different RowStride).
    if constexpr (condRowRpt) {
        for (unsigned i = 0; i < rptPerLine; i++) {
            Op::UnaryInstr(dst + offset, src + offset, validRow, dstStride, srcStride);
            offset += elemPerRpt;
        }

        if (remain) {
            SetContMaskByDType<T>(remain);
            Op::UnaryInstr(dst + offset, src + offset, validRow, dstStride, srcStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        // Otherwise, run a "head/tail" 2D loop:
        // - head: full repeats per row (no mask needed)
        // - tail: masked last repeat (handles the per-row remainder)
        UnaryPlusHead<Op, T, elemPerRpt, dstRowStride, srcRowStride>(dst, src, validRow, rptPerLine);
        offset = rptPerLine * elemPerRpt;
        dst += offset;
        src += offset;
        if (remain) {
            UnaryPlusTail<Op, T, elemPerBlk, dstRowStride, srcRowStride>(dst, src, validRow, remain);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, unaryFuncPtr<typename TileDataDst::DType> func,
    typename T = typename TileDataDst::DType>
__tf__ PTO_INTERNAL void TUnaryPlusOp(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ srcData, unsigned validRow, unsigned validCol)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    UnaryPlusInstr<T, UnaryOperation<T, func>, TileDataDst, TileDataSrc>(dst, src, validRow, validCol);
}

} // namespace pto

#endif // PTO_NPU_A2A3_T_OP_TEMPLATES_HPP
