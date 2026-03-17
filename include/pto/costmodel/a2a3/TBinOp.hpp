/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBIN_HPP
#define TBIN_HPP

#include <pto/common/constants.hpp>

namespace pto {
constexpr unsigned SMALL_RPT_BINOP = 4;

template <typename Op, typename T>
PTO_INTERNAL void Bin1LCountMode(unsigned validRow, unsigned validCol)
{
    Op::BinInstr(0);
}

template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL void Bin2LCountMode(unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(0);
    }
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned tileCols, uint8_t repeatStride>
PTO_INTERNAL void Bin1LNormModeSmall(unsigned validRow, unsigned validCol)
{
    Op::BinInstr(validRow, repeatStride, repeatStride, repeatStride);
    return;
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride,
          unsigned tileCols>
PTO_INTERNAL void Bin1LNormMode(unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinInstr(headRepeats); // headRepeats can be zero
    if (tailElements)
        [[unlikely]]
        {
            Op::BinInstr(1);
        }
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeColVLAlign(unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeHead(unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop)
                [[unlikely]]
                {
                    for (int j = 0; j < numLoop; j++) {
                        Op::BinInstr(REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                Op::BinInstr(remainAfterLoop);
            }
        }
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeTail(unsigned validRow, unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    constexpr bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (int i = 0; i < numLoop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::BinInstr(1, 1, 1, 1);
                }
            } else {
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinInstr(REPEAT_MAX, repeatStride, repeatStride, repeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                Op::BinInstr(1, 1, 1, 1);
            }
        } else {
            uint8_t repeatStride = stride / blockSizeElem;
            Op::BinInstr(remainAfterLoop, repeatStride, repeatStride, repeatStride);
        }
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeRowRpt(unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            Op::BinInstr(validRow, repeatStride, repeatStride, repeatStride);
        }

        if (tailElements) {
            Op::BinInstr(validRow, repeatStride, repeatStride, repeatStride);
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(validRow, numRepeatPerLine);
        }
        if (numRemainPerLine) {
            Bin2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename T, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void BinaryInstrFastPath(unsigned validRow, unsigned validCol)
{
    constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
    if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
        Bin1LCountMode<Op, T>(validRow, validCol);
    } else {
        Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(validRow, validCol);
    }
}

template <typename Op, typename T, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void BinaryInstrGeneralPath(unsigned validRow, unsigned validCol)
{
    // Continuous check in runtime(merge axis)
    if ((TileData::Cols == validCol) || (validRow == 1))
        [[likely]]
        {
            unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
            bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
            if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX))
                [[unlikely]]
                {
                    Bin1LCountMode<Op, T>(validRow, validCol);
                }
            else {
                Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(validRow, validCol);
            }
        }
    else { // Non continuous
        constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
        if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINOP)) {
            Bin2LCountMode<Op, T, rowStride>(validRow, validCol);
        } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
            if ((validCol % elementsPerRepeat) > 0) {
                Bin2LCountMode<Op, T, rowStride>(validRow, validCol);
            } else {
                Bin2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(validRow, validCol);
            }
        } else {
            Bin2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(validRow, validCol);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void BinaryInstr(unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    // Small shape optimization
    if constexpr ((TileData::Rows <= pto::REPEAT_MAX) && (TileData::Cols < elementsPerRepeat)) {
        constexpr uint8_t repeatStride = rowStride / blockSizeElem;
        Bin1LNormModeSmall<Op, T, elementsPerRepeat, TileData::Cols, repeatStride>(validRow, validCol);
        return;
    }
    // Continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        BinaryInstrFastPath<Op, T, TileData, elementsPerRepeat, blockSizeElem, rowStride>(validRow, validCol);
    } else {
        BinaryInstrGeneralPath<Op, T, TileData, elementsPerRepeat, blockSizeElem, rowStride>(validRow, validCol);
    }
}

// support differnent tile shape of dst, src0, src1
template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride,
          unsigned src0Stride = dstStride, unsigned src1Stride = dstStride>
PTO_INTERNAL void Bin2LNormModeHead(unsigned validRow, unsigned rptPerLine)
{
    if (rptPerLine > 0) {
        unsigned numLoop = rptPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = rptPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop)
                [[unlikely]]
                {
                    for (int j = 0; j < numLoop; j++) {
                        Op::BinInstr(REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                Op::BinInstr(remainAfterLoop);
            }
        }
    }
}

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
          unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeTail(unsigned validRow, unsigned remain)
{
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr bool src0StrideOverFlag = (src0Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool src1StrideOverFlag = (src1Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool dstStrideOverFlag = (dstStride / elemPerBlk > REPEAT_STRIDE_MAX);
    for (int i = 0; i < numLoop; i++) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                Op::BinInstr(1, 1, 1, 1);
            }
        } else {
            uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
            uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
            uint8_t dstBlkPerLine = dstStride / elemPerBlk;
            Op::BinInstr(REPEAT_MAX, dstBlkPerLine, src0BlkPerLine, src1BlkPerLine);
        }
    }
    remainAfterLoop = validRow % REPEAT_MAX;
    if (remainAfterLoop) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                Op::BinInstr(1, 1, 1, 1);
            }
        } else {
            uint8_t dstBlkPerLine = dstStride / elemPerBlk;
            uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
            uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
            Op::BinInstr(remainAfterLoop, dstBlkPerLine, src0BlkPerLine, src1BlkPerLine);
        }
    }
}

template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
          unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeRowRpt(unsigned validRow, unsigned validCol)
{
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    Bin2LNormModeHead<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>(validRow, rptPerLine);
    if (remain) {
        Bin2LNormModeTail<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>(validRow, remain);
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void BinaryInstr( unsigned validRows,
                              unsigned validCols)
{
	using T = typename TileData::DType;
    Bin2LNormModeRowRpt<Op, T, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(validRows, validCols);
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
    unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
PTO_INTERNAL void TBinaryOp(unsigned validRows, unsigned validCols)
{
    if constexpr (dstRowStride == src0RowStride && dstRowStride == src1RowStride) {
        BinaryInstr<Op, TileData, elementsPerRepeat, blockSizeElem, dstRowStride>(validRows, validCols);
    }else{
        BinaryInstr<Op, TileData, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            validRows, validCols);
    }
}

template <typename T, typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void runBinaryOp(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1){
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    if constexpr (std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>){
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        TBinaryOp<Op, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride>(dst.GetValidRow(), dst.GetValidCol());
    } else{
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;

        TBinaryOp<Op, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst.GetValidRow(), dst.GetValidCol());
    }
}

} // namespace pto
#endif