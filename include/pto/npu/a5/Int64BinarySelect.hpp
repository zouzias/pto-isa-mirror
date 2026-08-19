#ifndef INT64_BINARY_SELECT_HPP
#define INT64_BINARY_SELECT_HPP

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectStore(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& selectMask,
    MaskReg& validMask, vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High,
    vector_s32& src1Low, vector_s32& src1High)
{
    uint32_t src0Offset = (row * Src0Cols + colOffset) * 2;
    uint32_t src1Offset = (row * Src1Cols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, src0Offset, DINTLV_B32);
    vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, src1Offset, DINTLV_B32);
    vsel(dstLow, src0Low, src1Low, selectMask);
    vsel(dstHigh, src0High, src1High, selectMask);
    MaskReg lowMask, highMask;
    vector_s32 half0, half1;
    pintlv_b32(lowMask, highMask, validMask, validMask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalarStore(
    __ubuf__ T* dst, __ubuf__ T* src, uint16_t row, uint32_t colOffset, MaskReg& selectMask, MaskReg& validMask,
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, vector_s32& scalarLow,
    vector_s32& scalarHigh)
{
    uint32_t srcOffset = (row * SrcCols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, srcOffset, DINTLV_B32);
    vsel(dstLow, srcLow, scalarLow, selectMask);
    vsel(dstHigh, srcHigh, scalarHigh, selectMask);
    MaskReg lowMask, highMask;
    vector_s32 half0, half1;
    pintlv_b32(lowMask, highMask, validMask, validMask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

template <unsigned ElementsPerRepeat, unsigned MaskRowBytes>
PTO_INTERNAL void Int64SelectPairMasks(
    __ubuf__ uint8_t* packedMask, uint16_t row, uint16_t pairRepeat, uint32_t& colOffset, MaskReg& selectMask0,
    MaskReg& selectMask1)
{
    colOffset = pairRepeat * ElementsPerRepeat * 2;
    MaskReg packed;
    plds(packed, (__ubuf__ uint32_t*)packedMask, row * MaskRowBytes + colOffset / 8, US);
    MaskReg allMask = pset_b16(PAT_ALL);
    pintlv_b16(selectMask0, selectMask1, packed, allMask);
}

template <unsigned ElementsPerRepeat>
PTO_INTERNAL void Int64SelectValidMask(uint32_t remainingCols, uint32_t& cols, MaskReg& validMask)
{
    cols = remainingCols > ElementsPerRepeat ? ElementsPerRepeat : remainingCols;
    validMask = plt_b32(cols, POST_UPDATE);
}

template <unsigned ElementsPerRepeat, unsigned MaskRowBytes>
PTO_INTERNAL void Int64SelectTailMasks(
    __ubuf__ uint8_t* packedMask, uint16_t row, uint16_t pairRepeatTimes, uint32_t remainingCols, uint32_t& colOffset,
    MaskReg& selectMask, MaskReg& validMask)
{
    colOffset = pairRepeatTimes * ElementsPerRepeat * 2;
    uint32_t cols;
    Int64SelectValidMask<ElementsPerRepeat>(remainingCols, cols, validMask);
    MaskReg packed;
    plds(packed, (__ubuf__ uint32_t*)packedMask, row * MaskRowBytes + colOffset / 8, US);
    punpack(selectMask, packed, LOWER);
}

template <bool Scalar, typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectImpl(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, T scalar, unsigned validRows,
    unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    uint16_t pairRepeatTimes = repeatTimes / 2;
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        if constexpr (Scalar) {
            uint64_t scalarBits = static_cast<uint64_t>(scalar);
            vbr(src1Low, static_cast<int32_t>(scalarBits));
            vbr(src1High, static_cast<int32_t>(scalarBits >> 32));
        }
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t pairRepeat = 0; pairRepeat < pairRepeatTimes; ++pairRepeat) {
                uint32_t colOffset;
                MaskReg selectMask0, selectMask1;
                Int64SelectPairMasks<elementsPerRepeat, MaskRowBytes>(
                    packedMask, row, pairRepeat, colOffset, selectMask0, selectMask1);
                uint32_t cols0;
                MaskReg validMask0;
                Int64SelectValidMask<elementsPerRepeat>(remainingCols, cols0, validMask0);
                if constexpr (Scalar)
                    Int64SelectScalarStore<T, DstCols, Src0Cols>(
                        dst, src0, row, colOffset, selectMask0, validMask0, dstLow, dstHigh, src0Low, src0High, src1Low,
                        src1High);
                else
                    Int64SelectStore<T, DstCols, Src0Cols, Src1Cols>(
                        dst, src0, src1, row, colOffset, selectMask0, validMask0, dstLow, dstHigh, src0Low, src0High,
                        src1Low, src1High);
                remainingCols -= cols0;
                if (remainingCols > 0) {
                    uint32_t cols1 = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                    MaskReg validMask1 = plt_b32(cols1, POST_UPDATE);
                    colOffset += elementsPerRepeat;
                    if constexpr (Scalar)
                        Int64SelectScalarStore<T, DstCols, Src0Cols>(
                            dst, src0, row, colOffset, selectMask1, validMask1, dstLow, dstHigh, src0Low, src0High,
                            src1Low, src1High);
                    else
                        Int64SelectStore<T, DstCols, Src0Cols, Src1Cols>(
                            dst, src0, src1, row, colOffset, selectMask1, validMask1, dstLow, dstHigh, src0Low,
                            src0High, src1Low, src1High);
                    remainingCols -= cols1;
                }
            }
            if ((repeatTimes & 1) != 0) {
                uint32_t colOffset;
                MaskReg selectMask, validMask;
                Int64SelectTailMasks<elementsPerRepeat, MaskRowBytes>(
                    packedMask, row, pairRepeatTimes, remainingCols, colOffset, selectMask, validMask);
                if constexpr (Scalar)
                    Int64SelectScalarStore<T, DstCols, Src0Cols>(
                        dst, src0, row, colOffset, selectMask, validMask, dstLow, dstHigh, src0Low, src0High, src1Low,
                        src1High);
                else
                    Int64SelectStore<T, DstCols, Src0Cols, Src1Cols>(
                        dst, src0, src1, row, colOffset, selectMask, validMask, dstLow, dstHigh, src0Low, src0High,
                        src1Low, src1High);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Select(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows,
    unsigned validCols)
{
    Int64SelectImpl<false, T, DstCols, MaskRowBytes, Src0Cols, Src1Cols>(
        dst, packedMask, src0, src1, T(), validRows, validCols);
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalar(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    Int64SelectImpl<true, T, DstCols, MaskRowBytes, SrcCols, SrcCols>(
        dst, packedMask, src, src, scalar, validRows, validCols);
}

#endif
