/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <pto/common/constants.hpp>
namespace pto {
constexpr unsigned PTO_SMALL_RPT = 4;

template <typename Op, typename T>
PTO_INTERNAL void BinS1LCountMode(unsigned validRow, unsigned validCol)
{
    Op::BinSInstr(0);
}
template <typename Op, typename T, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LCountMode(unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        Op::BinSInstr(0);
    }
}
template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned Cols>
PTO_INTERNAL void BinS1LNormMode(unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinSInstr(headRepeats);
    if (tailElements)
        [[unlikely]]
        {
            Op::BinSInstr(1);
        }
}
template <typename Op, typename T, unsigned elementsPerRepeat, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeColVLAlign(unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (uint32_t i = 0; i < validRow; i++) {
        Op::BinSInstr(headRepeats);
    }
}
template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeHead(unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop)
                [[unlikely]]
                {
                    for (int j = 0; j < numLoop; j++) {
                        Op::BinSInstr(REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                Op::BinSInstr(remainAfterLoop);
            }
        }
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeTail(unsigned validRow, unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    const bool strideOverFlag =
        (dstStride / blockSizeElem > REPEAT_STRIDE_MAX) || (srcStride / blockSizeElem > REPEAT_STRIDE_MAX);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (int i = 0; i < numLoop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::BinSInstr(1, 1, 1);
                }
            } else {
                uint8_t dstRepeatStride = dstStride / blockSizeElem;
                uint8_t srcRepeatStride = srcStride / blockSizeElem;
                Op::BinSInstr(REPEAT_MAX, dstRepeatStride, srcRepeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }

    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                Op::BinSInstr(1, 1, 1);
            }
        } else {
            uint8_t dstRepeatStride = dstStride / blockSizeElem;
            uint8_t srcRepeatStride = srcStride / blockSizeElem;
            Op::BinSInstr(remainAfterLoop, dstRepeatStride, srcRepeatStride);
        }
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeRowRpt(unsigned validRow, unsigned validCol)
{
    constexpr unsigned dstRepeatStride = dstStride / blockSizeElem;
    constexpr unsigned srcRepeatStride = srcStride / blockSizeElem;
    constexpr bool condRowRpt =
        ((Rows <= pto::REPEAT_MAX) && dstRepeatStride <= (REPEAT_STRIDE_MAX) && srcRepeatStride <= (REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            Op::BinSInstr(validRow, dstRepeatStride, srcRepeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            Op::BinSInstr(validRow, dstRepeatStride, srcRepeatStride);
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            BinS2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(validRow,
                                                                                                    numRepeatPerLine);
        }
        if (numRemainPerLine) {
            BinS2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(validRow,
                                                                                                    numRemainPerLine);
        }
    }
}
template <typename Op, typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TBinSInstr(unsigned validRow, unsigned validCol)
{
    using T = typename TileDataDst::DType;
    constexpr bool tileDataContinue =
        ((TileDataDst::Cols == TileDataDst::ValidCol) && (TileDataSrc::Cols == TileDataSrc::ValidCol)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc::Rows == 1));
    if constexpr (tileDataContinue) {
        constexpr unsigned totalRepeats =
            (TileDataDst::Rows * TileDataDst::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        constexpr bool nonVLAligned =
            (((TileDataDst::Cols % elementsPerRepeat) != 0) && (TileDataDst::Cols > elementsPerRepeat));
        constexpr bool enbleCountMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
        if constexpr (enbleCountMode) {
            BinS1LCountMode<Op, T>(validRow, validCol);
        } else {
            BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, TileDataDst::Cols>(validRow, validCol);
        }
    } else {
        if (tileDataContinue)
            [[likely]]
            {
                unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
                bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
                bool enbleCountMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
                if (enbleCountMode)
                    [[unlikely]]
                    {
                        BinS1LCountMode<Op, T>(validRow, validCol);
                    }
                else {
                    BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, TileDataDst::Cols>(validRow, validCol);
                }
            }
        else {
            constexpr unsigned normColRepeat = TileDataDst::Cols / elementsPerRepeat;
            constexpr bool countMode = (normColRepeat > 1) && ((TileDataDst::Rows * normColRepeat) < PTO_SMALL_RPT) &&
                                       ((TileDataSrc::Rows * normColRepeat) < PTO_SMALL_RPT);
            constexpr bool isColRpt =
                (TileDataDst::Rows < (normColRepeat + 1)) && (TileDataSrc::Rows < (normColRepeat + 1));
            if constexpr (countMode) {
                BinS2LCountMode<Op, T, dstStride, srcStride>(validRow, validCol);
            } else if constexpr (isColRpt) {
                unsigned tailElements = validCol % elementsPerRepeat;
                if (tailElements) {
                    BinS2LCountMode<Op, T, dstStride, srcStride>(validRow, validCol);
                } else {
                    BinS2LNormModeColVLAlign<Op, T, elementsPerRepeat, dstStride, srcStride>(validRow, validCol);
                }
            } else {
                BinS2LNormModeRowRpt<Op, T, TileDataDst::Rows, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
                    validRow, validCol);
            }
        }
    }
}

template <typename T, typename Op, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TBinaryScalarOp(unsigned validRow, unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    TBinSInstr<Op, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(validRow,
                                                                                                     validCol);
}

template <typename T, typename Op, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void runBinaryScalarOp(TileDataDst &dst, TileDataSrc &src){
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    if ((dstValidRow != 0 && dstValidCol != 0) &&
        (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
        TBinaryScalarOp<T, Op, TileDataDst, TileDataSrc>(dstValidRow, dstValidCol);
    } else {
        PTO_ASSERT(false, "TADDS: dstTile validRow/validCol must be consistent with of src");
    }
}
} // namespace pto
#endif
