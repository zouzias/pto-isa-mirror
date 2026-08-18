/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_REDUCE_HPP
#define INT64_REDUCE_HPP

#include <pto/npu/a5/Int64Binary.hpp>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <Int64Op Op, typename T>
PTO_INTERNAL void Int64PartCalcRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    if constexpr (Op == Int64Op::Add) {
        Int64AddRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    } else {
        Int64MinMax<Op, T>(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSameStride(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dstCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = src0Rows < src1Rows ? src0Rows : src1Rows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vlds(al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, 0, DINTLV_B32);
                vlds(bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, 0, DINTLV_B32);
                Int64PartCalcRegs<Op, T>(dl, dh, al, ah, bl, bh, mask);
                vsts(dl, dh, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, INTLV_B32, mask);
                remainingCols -= cols;
            }
        }
    }
    __VEC_SCOPE__
    {
        vector_s32 low, high;
        uint16_t firstRow = src0Rows < src1Rows ? src0Rows : src1Rows;
        uint16_t rows = dstRows;
        for (uint16_t row = firstRow; row < rows; ++row) {
            uint32_t remainingCols = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                if (src0Rows > src1Rows)
                    vlds(low, high, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, 0, DINTLV_B32);
                else
                    vlds(low, high, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, 0, DINTLV_B32);
                vsts(low, high, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, INTLV_B32, mask);
                remainingCols -= cols;
            }
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartMergeOverlap(
    vector_s32& dl, vector_s32& dh, vector_s32& al, vector_s32& ah, vector_s32& bl, vector_s32& bh, unsigned src0Cols,
    unsigned src1Cols, unsigned colOffset, unsigned cols)
{
    bool useSrc0Base = src0Cols >= src1Cols;
    if (useSrc0Base) {
        dl = al;
        dh = ah;
    } else {
        dl = bl;
        dh = bh;
    }
    unsigned overlapCols = src0Cols < src1Cols ? src0Cols : src1Cols;
    if (colOffset >= overlapCols) {
        return;
    }
    uint32_t overlap = overlapCols - colOffset;
    overlap = overlap > cols ? cols : overlap;
    uint32_t maskCols = overlap;
    MaskReg opMask = plt_b32(maskCols, POST_UPDATE);
    vector_s32 ol, oh;
    Int64PartCalcRegs<Op, T>(ol, oh, al, ah, bl, bh, opMask);
    if (useSrc0Base) {
        vsel(dl, ol, al, opMask);
        vsel(dh, oh, ah, opMask);
    } else {
        vsel(dl, ol, bl, opMask);
        vsel(dh, oh, bh, opMask);
    }
}

template <typename T, unsigned SrcCols>
PTO_INTERNAL void Int64PartLoadRegs(
    vector_s32& low, vector_s32& high, __ubuf__ T* src, unsigned row, unsigned colOffset)
{
    vlds(low, high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralSingleRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = dstRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t storeCols = dstCols;
            MaskReg storeMask = plt_b32(storeCols, POST_UPDATE);
            bool hasSrc0 = row < src0Rows;
            bool hasSrc1 = row < src1Rows;
            if (hasSrc0)
                Int64PartLoadRegs<T, Src0Cols>(al, ah, src0, row, 0);
            if (hasSrc1)
                Int64PartLoadRegs<T, Src1Cols>(bl, bh, src1, row, 0);
            if (hasSrc0 && hasSrc1) {
                Int64PartMergeOverlap<Op, T, DstCols, Src0Cols, Src1Cols>(
                    dl, dh, al, ah, bl, bh, src0Cols, src1Cols, 0, dstCols);
            } else if (hasSrc0) {
                dl = al;
                dh = ah;
            } else {
                dl = bl;
                dh = bh;
            }
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, storeMask);
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralMultiRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dstCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = dstRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t storeCols = cols;
                MaskReg storeMask = plt_b32(storeCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                bool hasSrc0 = row < src0Rows && colOffset < src0Cols;
                bool hasSrc1 = row < src1Rows && colOffset < src1Cols;
                if (hasSrc0)
                    Int64PartLoadRegs<T, Src0Cols>(al, ah, src0, row, colOffset);
                if (hasSrc1)
                    Int64PartLoadRegs<T, Src1Cols>(bl, bh, src1, row, colOffset);
                if (row < src0Rows && row < src1Rows) {
                    Int64PartMergeOverlap<Op, T, DstCols, Src0Cols, Src1Cols>(
                        dl, dh, al, ah, bl, bh, src0Cols, src1Cols, colOffset, cols);
                } else if (hasSrc0) {
                    dl = al;
                    dh = ah;
                } else {
                    dl = bl;
                    dh = bh;
                }
                vsts(dl, dh, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, INTLV_B32, storeMask);
                remainingCols -= cols;
            }
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneral(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    if (dstCols <= elementsPerRepeat) {
        Int64PartGeneralSingleRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
            dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
        return;
    }
    Int64PartGeneralMultiRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
        dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
}

template <Int64Op Op, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL bool Int64PartUseSameStride(unsigned src0Cols, unsigned src1Cols, unsigned dstCols)
{
    constexpr bool supportsSameStride = Op == Int64Op::Add || Op == Int64Op::Max || Op == Int64Op::Min;
    constexpr bool hasSameStride = DstCols == Src0Cols && DstCols == Src1Cols;
    return supportsSameStride && hasSameStride && src0Cols == dstCols && src1Cols == dstCols;
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Part(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    if (Int64PartUseSameStride<Op, DstCols, Src0Cols, Src1Cols>(src0Cols, src1Cols, dstCols)) {
        Int64PartSameStride<Op, T, DstCols, Src0Cols, Src1Cols>(
            dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
        return;
    }
    Int64PartGeneral<Op, T, DstCols, Src0Cols, Src1Cols>(
        dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ColReduce(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, sl, sh, nl, nh;
        uint32_t remainingCols = validCols;
        for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
            uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
            uint32_t maskCols = cols;
            MaskReg mask = plt_b32(maskCols, POST_UPDATE);
            uint32_t colOffset = colRepeat * elementsPerRepeat;
            vlds(dl, dh, (__ubuf__ int32_t*)src + colOffset * 2, 0, DINTLV_B32);
            uint16_t rows = validRows;
            for (uint16_t row = 1; row < rows; ++row) {
                vlds(sl, sh, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
                if constexpr (Op == Int64Op::Add) {
                    MaskReg carry, carryOut;
                    vaddc(carry, nl, dl, sl, mask);
                    vaddcs(carryOut, nh, dh, sh, carry, mask);
                } else {
                    Int64MinMax<Op, T>(nl, nh, dl, dh, sl, sh, mask);
                }
                dl = nl;
                dh = nh;
            }
            vsts(dl, dh, (__ubuf__ int32_t*)dst + colOffset * 2, 0, INTLV_B32, mask);
            remainingCols -= cols;
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSum(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_u32 low, high, low16, mid16, tmp, mask16, outLow, outHigh, accLow, accHigh;
        vbr(mask16, 0xffffu);
        MaskReg oneMask = pset_b32(PAT_VL1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                vlds(
                    (vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0,
                    DINTLV_B32);
                vand(low16, low, mask16, mask, MODE_ZEROING);
                vcadd(low16, low16, mask, MODE_ZEROING);
                vshrs(mid16, low, 16, mask, MODE_ZEROING);
                vcadd(mid16, mid16, mask, MODE_ZEROING);
                vcadd(outHigh, high, mask, MODE_ZEROING);
                vshrs(tmp, low16, 16, mask, MODE_ZEROING);
                vadd(mid16, mid16, tmp, mask, MODE_ZEROING);
                vshrs(tmp, mid16, 16, mask, MODE_ZEROING);
                vadd(outHigh, outHigh, tmp, mask, MODE_ZEROING);
                vand(low16, low16, mask16, mask, MODE_ZEROING);
                vand(mid16, mid16, mask16, mask, MODE_ZEROING);
                vshls(mid16, mid16, 16, mask, MODE_ZEROING);
                vor(outLow, low16, mid16, mask);
                if (colRepeat == 0) {
                    accLow = outLow;
                    accHigh = outHigh;
                } else {
                    MaskReg carry, carryOut;
                    vaddc(carry, (vector_s32&)accLow, (vector_s32&)accLow, (vector_s32&)outLow, oneMask);
                    vaddcs(carryOut, (vector_s32&)accHigh, (vector_s32&)accHigh, (vector_s32&)outHigh, carry, oneMask);
                }
                remainingCols -= cols;
            }
            vsts(
                (vector_s32&)accLow, (vector_s32&)accHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32,
                oneMask);
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowReduceHigh(vector_s32& reducedHigh, vector_s32& high, MaskReg& mask)
{
    if constexpr (Op == Int64Op::Max) {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmax(reducedHigh, high, mask, MODE_ZEROING);
        else
            vcmax((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmin(reducedHigh, high, mask, MODE_ZEROING);
        else
            vcmin((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
    }
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowSelectHigh(vector_s32& selectedHigh, vector_s32& high, MaskReg& equalLow)
{
    if constexpr (Op == Int64Op::Max) {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmax(selectedHigh, high, equalLow, MODE_ZEROING);
        else
            vcmax((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmin(selectedHigh, high, equalLow, MODE_ZEROING);
        else
            vcmin((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
    }
}

template <Int64Op Op, typename T, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMaxRepeat(
    vector_s32& outLow, vector_s32& outHigh, __ubuf__ T* src, unsigned row, unsigned colOffset, MaskReg& mask,
    MaskReg& allMask)
{
    vector_s32 low, high, reducedHigh, highDup, selectedHigh, lowDup;
    vector_u32 reducedLow;
    vlds(low, high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
    Int64RowReduceHigh<Op, T, 1, SrcCols>(reducedHigh, high, mask);
    vdup(highDup, reducedHigh, allMask, POS_LOWEST, MODE_ZEROING);
    MaskReg equalHigh;
    vcmp_eq(equalHigh, highDup, high, mask);
    if constexpr (Op == Int64Op::Max)
        vcmax(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);
    else
        vcmin(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);

    vdup((vector_s32&)lowDup, (vector_s32&)reducedLow, allMask, POS_LOWEST, MODE_ZEROING);
    MaskReg equalLow;
    vcmp_eq(equalLow, (vector_u32&)lowDup, (vector_u32&)low, mask);
    Int64RowSelectHigh<Op, T>(selectedHigh, high, equalLow);
    outLow = (vector_s32&)reducedLow;
    outHigh = selectedHigh;
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowMinMaxInitAcc(
    vector_s32& accLow, vector_s32& accHigh, vector_s32& repeatLow, vector_s32& repeatHigh)
{
    accLow = repeatLow;
    accHigh = repeatHigh;
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowMinMaxAccumulate(
    vector_s32& accLow, vector_s32& accHigh, vector_s32& repeatLow, vector_s32& repeatHigh, MaskReg& mask)
{
    Int64MinMax<Op, T>(accLow, accHigh, accLow, accHigh, repeatLow, repeatHigh, mask);
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64RowMinMaxStore(vector_s32& accLow, vector_s32& accHigh, __ubuf__ T* dst, unsigned row)
{
    MaskReg oneMask = pset_b32(PAT_VL1);
    vsts(accLow, accHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, oneMask);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMax(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 accLow, accHigh, repeatLow, repeatHigh;
        MaskReg allMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                uint32_t maskCols = cols;
                MaskReg mask = plt_b32(maskCols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64RowMinMaxRepeat<Op, T, SrcCols>(repeatLow, repeatHigh, src, row, colOffset, mask, allMask);
                if (colRepeat == 0) {
                    Int64RowMinMaxInitAcc<Op, T>(accLow, accHigh, repeatLow, repeatHigh);
                } else {
                    MaskReg oneMask = pset_b32(PAT_VL1);
                    Int64RowMinMaxAccumulate<Op, T>(accLow, accHigh, repeatLow, repeatHigh, oneMask);
                }
                remainingCols -= cols;
            }
            Int64RowMinMaxStore<T, DstCols>(accLow, accHigh, dst, row);
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See Int64Binary.hpp for details.
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Part(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols);

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ColReduce(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSum(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMax(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);
#endif

} // namespace pto

#endif
