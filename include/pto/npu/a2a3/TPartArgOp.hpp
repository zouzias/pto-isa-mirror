/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGOP_HPP
#define TPARTARGOP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename T, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TPartCopyInstr(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, uint64_t validRow, uint64_t validCol,
                                 uint64_t startRow)
{
    constexpr uint64_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    validRow -= startRow;
    srcPtr += startRow * srcStride;
    dstPtr += startRow * dstStride;
    if constexpr (dstStride == srcStride) {
        set_mask_count();
        SetVectorCount(dstStride * validRow);
        uint64_t blockCount = CeilDivision(dstStride * validRow, elemPerBlock);
        pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, blockCount, 1, 1);
    } else {
        set_mask_count();
        SetVectorCount(validCol);
        uint64_t blockCount = CeilDivision(validCol, elemPerBlock);
        for (uint64_t i = 0; i < validRow; i++) {
            pto_copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr + i * srcStride, 1, blockCount, 1, 1);
        }
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx,
    typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void PartByRow(__ubuf__ T *dstValPtr, __ubuf__ T *dstIdxPtr, __ubuf__ T *srcVal0Ptr, __ubuf__ T *srcIdx0Ptr,
                           __ubuf__ T *srcVal1Ptr, __ubuf__ T *srcIdx1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        Op::CmpInstr(srcVal0Ptr + i * TileSrcVal0::RowStride, srcVal1Ptr + i * TileSrcVal1::RowStride, 0);
        pipe_barrier(PIPE_V);
        vsel(dstValPtr + i * TileDstVal::RowStride, srcVal0Ptr + i * TileSrcVal0::RowStride, srcVal1Ptr + i * TileSrcVal1::RowStride, 0, 1, 1, 1, 8, 8, 8);
        vsel(dstIdxPtr + i * TileDstIdx::RowStride, srcVal0Ptr + i * TileSrcIdx0::RowStride, srcVal1Ptr + i * TileSrcIdx1::RowStride, 0, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    SetFullVecMaskByDType<TVal>();
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx,
    typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void PartByGroupTail(__ubuf__ TVal *dstValPtr, __ubuf__ TIdx *dstIdxPtr, __ubuf__ TVal *srcVal0Ptr, __ubuf__ TIdx *srcIdx0Ptr,
                           __ubuf__ TVal *srcVal1Ptr, __ubuf__ TIdx *srcIdx1Ptr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(TVal);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TVal);
    constexpr unsigned dstValRepeatStride = TileDstVal::RowStride / blockSizeElem;
    constexpr unsigned dstIdxRepeatStride = TileDstIdx::RowStride / blockSizeElem;
    constexpr unsigned srcVal0RepeatStride = TileSrcVal0::RowStride / blockSizeElem;
    constexpr unsigned srcIdx0RepeatStride = TileSrcIdx0::RowStride / blockSizeElem;
    constexpr unsigned srcVal1RepeatStride = TileSrcVal1::RowStride / blockSizeElem;
    constexpr unsigned srcIdx1RepeatStride = TileSrcIdx1::RowStride / blockSizeElem;
    constexpr unsigned dstValRepeatMaxStride = REPEAT_MAX * TileDstVal::RowStride;
    constexpr unsigned dstIdxRepeatMaxStride = REPEAT_MAX * TileDstIdx::RowStride;
    constexpr unsigned srcVal0RepeatMaxStride = REPEAT_MAX * TileSrcVal0::RowStride;
    constexpr unsigned srcIdx0RepeatMaxStride = REPEAT_MAX * TileSrcIdx0::RowStride;
    constexpr unsigned srcVal1RepeatMaxStride = REPEAT_MAX * TileSrcVal1::RowStride;
    constexpr unsigned srcIdx1RepeatMaxStride = REPEAT_MAX * TileSrcIdx1::RowStride;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;
    SetContMaskByDType<T>(numRemainPerLine);
    if constexpr (dstRow >= REPEAT_MAX) {
        for (unsigned j = 0; j < numRepeatPerCol; j++) {
            Op::CmpInstr(srcVal0Ptr + j * srcVal0RepeatMaxStride,
                            srcVal1Ptr + j * srcVal1RepeatMaxStride, REPEAT_MAX,
                            dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
            pipe_barrier(PIPE_V);
            vsel(dstValPtr + j * dstValRepeatMaxStride,
                srcVal0Ptr + j * srcVal0RepeatMaxStride,
                srcVal1Ptr + j * srcVal1RepeatMaxStride,
                REPEAT_MAX, 1, 1, 1, dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
            vsel(dstIdxPtr + j * dstIdxRepeatMaxStride,
                srcIdx0Ptr + j * srcIdx0RepeatMaxStride,
                srcVal1Ptr + j * srcIdx1RepeatMaxStride,
                REPEAT_MAX, 1, 1, 1, dstIdxRepeatStride, srcIdx0RepeatStride, srcIdx1RepeatStride);
            pipe_barrier(PIPE_V);
        }
    }
    if (numRemainPerCol) {
        Op::CmpInstr(srcVal0Ptr + numRemainPerCol * srcVal0RepeatMaxStride,
                        srcVal1Ptr + numRemainPerCol * srcVal1RepeatMaxStride, REPEAT_MAX,
                        dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
        pipe_barrier(PIPE_V);
        vsel(dstValPtr + numRemainPerCol * dstValRepeatMaxStride,
            srcVal0Ptr + numRemainPerCol * srcVal0RepeatMaxStride,
            srcVal1Ptr + numRemainPerCol * srcVal1RepeatMaxStride,
            REPEAT_MAX, 1, 1, 1, dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
        vsel(dstIdxPtr + numRemainPerCol * dstIdxRepeatMaxStride,
            srcIdx0Ptr + numRemainPerCol * srcIdx0RepeatMaxStride,
            srcVal1Ptr + numRemainPerCol * srcIdx1RepeatMaxStride,
            REPEAT_MAX, 1, 1, 1, dstIdxRepeatStride, srcIdx0RepeatStride, srcIdx1RepeatStride);
        pipe_barrier(PIPE_V);
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx,
    typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void PartByGroup(__ubuf__ TVal *dstValPtr, __ubuf__ TIdx *dstIdxPtr, __ubuf__ TVal *srcVal0Ptr, __ubuf__ TIdx *srcIdx0Ptr,
                           __ubuf__ TVal *srcVal1Ptr, __ubuf__ TIdx *srcIdx1Ptr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(TVal);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TVal);
    constexpr unsigned dstValRepeatStride = TileDstVal::RowStride / blockSizeElem;
    constexpr unsigned dstIdxRepeatStride = TileDstIdx::RowStride / blockSizeElem;
    constexpr unsigned srcVal0RepeatStride = TileSrcVal0::RowStride / blockSizeElem;
    constexpr unsigned srcIdx0RepeatStride = TileSrcIdx0::RowStride / blockSizeElem;
    constexpr unsigned srcVal1RepeatStride = TileSrcVal1::RowStride / blockSizeElem;
    constexpr unsigned srcIdx1RepeatStride = TileSrcIdx1::RowStride / blockSizeElem;
    constexpr unsigned dstValRepeatMaxStride = REPEAT_MAX * TileDstVal::RowStride;
    constexpr unsigned dstIdxRepeatMaxStride = REPEAT_MAX * TileDstIdx::RowStride;
    constexpr unsigned srcVal0RepeatMaxStride = REPEAT_MAX * TileSrcVal0::RowStride;
    constexpr unsigned srcIdx0RepeatMaxStride = REPEAT_MAX * TileSrcIdx0::RowStride;
    constexpr unsigned srcVal1RepeatMaxStride = REPEAT_MAX * TileSrcVal1::RowStride;
    constexpr unsigned srcIdx1RepeatMaxStride = REPEAT_MAX * TileSrcIdx1::RowStride;
    if constexpr (TileDstVal::Cols >= static_cast<int>(elementsPerRepeat)) {
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        unsigned numRepeatPerCol = validRow / REPEAT_MAX;
        unsigned numRemainPerCol = validRow % REPEAT_MAX;
        unsigned dstRepeatStride = dstStride / blockSizeElem;
        unsigned dstRepeatMaxStride = REPEAT_MAX * dstStride;
        SetFullVecMaskByDType<TVal>();
        for (unsigned i = 0; i < numRepeatPerLine; i++) {
            if constexpr (dstRow >= REPEAT_MAX) {
                for (unsigned j = 0; j < numRepeatPerCol; j++) {
                    Op::CmpInstr(srcVal0Ptr + j * srcVal0RepeatMaxStride + i * elementsPerRepeat,
                                  srcVal1Ptr + j * srcVal1RepeatMaxStride + i * elementsPerRepeat, REPEAT_MAX,
                                  dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
                    pipe_barrier(PIPE_V);
                    vsel(dstValPtr + j * dstValRepeatMaxStride + i * elementsPerRepeat,
                        srcVal0Ptr + j * srcVal0RepeatMaxStride + i * elementsPerRepeat,
                        srcVal1Ptr + j * srcVal1RepeatMaxStride + i * elementsPerRepeat,
                        REPEAT_MAX, 1, 1, 1, dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
                    vsel(dstIdxPtr + j * dstIdxRepeatMaxStride + i * elementsPerRepeat,
                        srcIdx0Ptr + j * srcIdx0RepeatMaxStride + i * elementsPerRepeat,
                        srcVal1Ptr + j * srcIdx1RepeatMaxStride + i * elementsPerRepeat,
                        REPEAT_MAX, 1, 1, 1, dstIdxRepeatStride, srcIdx0RepeatStride, srcIdx1RepeatStride);
                    pipe_barrier(PIPE_V);
                }
            }
            if (numRemainPerCol) {
                Op::CmpInstr(src0Ptr + numRepeatPerCol * srcVal0RepeatMaxStride + i * elementsPerRepeat,
                              src1Ptr + numRepeatPerCol * srcVal1RepeatMaxStride + i * elementsPerRepeat,
                              numRemainPerCol, dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
                pipe_barrier(PIPE_V);
                vsel(dstValPtr + numRepeatPerCol * dstValRepeatMaxStride + i * elementsPerRepeat,
                    srcVal0Ptr + numRepeatPerCol * srcVal0RepeatMaxStride + i * elementsPerRepeat,
                    srcVal1Ptr + numRepeatPerCol * srcVal1RepeatMaxStride + i * elementsPerRepeat,
                    numRemainPerCol, 1, 1, 1, dstValRepeatStride, srcVal0RepeatStride, srcVal1RepeatStride);
                vsel(dstIdxPtr + numRepeatPerCol * dstIdxRepeatMaxStride + i * elementsPerRepeat,
                    srcIdx0Ptr + numRepeatPerCol * srcIdx0RepeatMaxStride + i * elementsPerRepeat,
                    srcVal1Ptr + numRepeatPerCol * srcIdx1RepeatMaxStride + i * elementsPerRepeat,
                    numRemainPerCol, 1, 1, 1, dstIdxRepeatStride, srcIdx0RepeatStride, srcIdx1RepeatStride);
                pipe_barrier(PIPE_V);
            }
        }
        unsigned offset = numRepeatPerLine * elementsPerRepeat;
        dstValPtr += offset;
        dstIdxPtr += offset;
        srcVal0Ptr += offset;
        srcIdx0Ptr += offset;
        srcVal1Ptr += offset;
        srcIdx1Ptr += offset;
    }
    if (numRemainPerLine) {
        PartByGroupTail<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
            (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, validRow, validCol);
    }
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx,
    typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void TPartOps(__ubuf__ T *dstValPtr, __ubuf__ T *dstIdxPtr, __ubuf__ T *srcVal0Ptr, __ubuf__ T *srcIdx0Ptr,
                           __ubuf__ T *srcVal1Ptr, __ubuf__ T *srcIdx1Ptr, unsigned validRow,
                           unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(TVal);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TVal);
    bool constexpr strideOverFlag =
        ((TileSrcVal0::RowStride / blockSizeElem > REPEAT_STRIDE_MAX) ||
         (TileSrcVal1::RowStride / blockSizeElem > REPEAT_STRIDE_MAX) ||
         (TileDstVal::RowStride / blockSizeElem > REPEAT_STRIDE_MAX) ||
         (TileSrcIdx0::RowStride / blockSizeElem > REPEAT_STRIDE_MAX) ||
         (TileSrcIdx1::RowStride / blockSizeElem > REPEAT_STRIDE_MAX) ||
         (TileDstIdx::RowStride / blockSizeElem > REPEAT_STRIDE_MAX));
    if (validRow == 0 || validCol == 0) {
        return;
    }
    if constexpr (strideOverFlag) {
        PartByRow<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
            (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, validRow, validCol);
    } else {
        if (validRow < CeilDivision(validCol, elementsPerRepeat) * CeilDivision(validRow, REPEAT_MAX)) {
            PartByRow<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
                (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, validRow, validCol);
        } else {
            PartByGroup<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
                (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, validRow, validCol);
        }
    }
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx,
    typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void TPartArgInstr(__ubuf__ TVal *dstValPtr, __ubuf__ TIdx *dstIdxPtr, __ubuf__ TVal *srcVal0Ptr, __ubuf__ TIdx *srcIdx0Ptr,
                             __ubuf__ TVal *srcVal1Ptr, __ubuf__ TTIdx *srcIdx1Ptr,
                             unsigned dstValidRow, unsigned dstValidCol,
                             unsigned src0ValidRow, unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol)
{
    bool rowDiffer = (src0ValidRow != src1ValidRow && src0ValidCol == src1ValidCol);
    bool colDiffer = (src0ValidRow == src1ValidRow && src0ValidCol != src1ValidCol);
    bool sameSize = (src0ValidRow == src1ValidRow && src0ValidCol == src1ValidCol);
    if (rowDiffer) {
        TPartOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
            (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr,
                dstValidRow, dstValidCol, min(src0ValidRow, src1ValidRow), src0ValidCol);
        if (src0ValidRow < src1ValidRow) {
            TPartCopyInstr<TVal, TileDstVal::Cols, TileSrcVal1::Cols>(dstValPtr, srcVal1Ptr, src1ValidRow, src0ValidCol, src0ValidRow);
            TPartCopyInstr<TIdx, TileDstIdx::Cols, TileSrcIdx1::Cols>(dstIdxPtr, srcIdx1Ptr, src1ValidRow, src0ValidCol, src0ValidRow);
        } else {
            TPartCopyInstr<TVal, TileDstVal::Cols, TileSrcVal1::Cols>(dstValPtr, srcVal0Ptr, src0ValidRow, src1ValidCol, src1ValidRow);
            TPartCopyInstr<TIdx, TileDstIdx::Cols, TileSrcIdx1::Cols>(dstIdxPtr, srcIdx0Ptr, src0ValidRow, src1ValidCol, src1ValidRow);
        }
    } else if (colDiffer) {
        if (src0ValidCol < src1ValidCol) {
            TPartCopyInstr<TVal, TileDstVal::Cols, TileSrcVal1::Cols>(dstValPtr, srcVal1Ptr, src0ValidRow, src1ValidCol, 0);
            TPartCopyInstr<TIdx, TileDstIdx::Cols, TileSrcIdx1::Cols>(dstIdxPtr, srcIdx1Ptr, src0ValidRow, src1ValidCol, 0);
        } else {
            TPartCopyInstr<TVal, TileDstVal::Cols, TileSrcVal1::Cols>(dstValPtr, srcVal0Ptr, src0ValidRow, src0ValidCol, 0);
            TPartCopyInstr<TIdx, TileDstIdx::Cols, TileSrcIdx1::Cols>(dstIdxPtr, srcIdx0Ptr, src0ValidRow, src0ValidCol, 0);
        }
        if (src0ValidCol != 0 && src1ValidCol != 0) {
            TPartOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
                (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr,
                    dstValidRow, dstValidCol, src0ValidRow, min(src0ValidCol, src1ValidCol));
        }
    } else if (sameSize) {
        TPartOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
            (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr,
                dstValidRow, dstValidCol, src0ValidRow, src0ValidCol);
    } else {
        PTO_ASSERT(rowDiffer || colDiffer || sameSize,
            "TPARTARGOPS: Only one entry in the valid-rows and valid-cols of src0 and src1 is smaller than dst.");
    }
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL bool checkTiles(unsigned dstValidRow, unsigned dstValidCol, unsigned dstIdxValidRow, unsigned dstIdxValidCol,
    unsigned src0ValidRow, unsigned src0ValidCol, unsigned srcIdx0ValidRow, unsigned srcIdx0ValidCol,
    unsigned src1ValidRow, unsigned src1ValidCol, unsigned srcIdx1ValidRow, unsigned srcIdx1ValidCol)
{
    using TVal = typename TileDstVal::DType;
    using TIdx = typename TileDstIdx::DType;
    static_assert(std::is_same_v<TVal, typename TileSrcVal0::DType> && std::is_same_v<TVal, typename TileSrcVal1::DType>,
        "TPARTARGOPS: dstVal srcVal0 srcVal1 type must be consistent");
    static_assert(std::is_same_v<TIdx, typename TileSrcIdx0::DType> && std::is_same_v<TIdx typename TileSrcIdx1::DType>,
        "TPARTARGOPS: dstIdx srcIdx0 srcIdx1 type must be consistent");
    static_assert(std::is_same_v<TVal, float> || std::is_same_v<TVal, half>,
        "TPARTARGOPS: val type only support float, half.");
    static_assert(std::is_integral_v<TIdx> && sizeof(TIdx) == sizeof(TVal)),
        "TPARTARGOPS: idx must be integral and the same size as val");
    if (dstValidRow != dstIdxValidRow || dstValidCol != dstIdxValidCol || src0ValidRow != srcIdx0ValidRow || src0ValidCol != srcIdx0ValidCol ||
        src1ValidRow != srcIdx1ValidRow || src1ValidCol != srcIdx1ValidCol) {
        PTO_ASSERT(false, "TPARTARGOPS: idxTile validRow/validCol must be consistent with of valTile.");
        return false;
    }
    if (dstValidRow == 0 || dstValidCol == 0) {
        PTO_ASSERT(false, "TPARTARGOPS: dst valid size must be non zero.");
        return false;
    }
    if (dstValidRow != max(src0ValidRow, src1ValidRow) || dstValidCol != max(src0ValidCol, src1ValidCol)) {
        PTO_ASSERT(false, "TPARTARGOPS: dst valid size must be consistent with src of bigger size.");
        return false;
    }
    return true;
}

template <typename T>
struct PartMaxArgCmpOp {
    PTO_INTERNAL static void CmpInstr(__ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vcmp_ge(src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void CmpInstr(__ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                       uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vcmp_ge(src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename TileDstVal, typename TileDstIdx, typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
__tf__ PTO_INTERNAL void TPartArgMax(typename TileDstVal::TileDType __out__ dstVal, typename TileDstIdx::TileDType __out__ dstIdx,
                                  typename TileSrcVal0::TileDType __in__ srcVal0, typename TileSrcIdx0::TileDType __in__ srcIdx0,
                                  typename TileSrcVal1::TileDType __in__ srcVal1, typename TileSrcIdx1::TileDType __in__ srcIdx1,
                                  unsigned dstValidRow, unsigned dstValidCol,
                                  unsigned src0ValidRow, unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol)
{
    using TVal = typename TileDstVal::DType;
    using TIdx = typename TileDstIdx::DType;
    TPartInstr<PartMaxArgCmpOp<TVal>, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
        (dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr,
        dstValidRow, dstValidCol, src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol);
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void TPARTARGMAX_IMPL(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrcVal0 &srcVal0, TileSrcIdx0 &srcIdx0, TileSrcVal1 &srcVal1, TileSrcIdx1 &srcIdx1)
{
    unsigned src0ValidRow = srcVal0.GetValidRow();
    unsigned src0ValidCol = srcVal0.GetValidCol();
    unsigned src1ValidRow = srcVal1.GetValidRow();
    unsigned src1ValidCol = srcVal1.GetValidCol();
    unsigned dstValidRow = dstVal.GetValidRow();
    unsigned dstValidCol = dstVal.GetValidCol();
    unsigned srcIdx0ValidRow = srcIdx0.GetValidRow();
    unsigned srcIdx0ValidCol = srcIdx0.GetValidCol();
    unsigned srcIdx1ValidRow = srcIdx1.GetValidRow();
    unsigned srcIdx1ValidCol = srcIdx1.GetValidCol();
    unsigned dstIdxValidRow = dstIdx.GetValidRow();
    unsigned dstIdxValidCol = dstIdx.GetValidCol();
    if (!checkTiles<TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
        (dstValidRow, dstValidCol, dstIdxValidRow, dstIdxValidCol, src0ValidRow, src0ValidCol,
        srcIdx0ValidRow, srcIdx0ValidCol, src1ValidRow, src1ValidCol, srcIdx1ValidRow, srcIdx1ValidCol)) {
        return;
    }
    TPartArgMax<TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>
        (dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol, dstValidRow, dstValidCol);
}

} // namespace pto
#endif