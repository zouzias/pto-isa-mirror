/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include <pto/common/constants.hpp>

namespace pto {
#define SMALL_RPT (4)
template <typename Op, typename T>

PTO_INTERNAL void Unary1LCountMode(unsigned validRow, unsigned validCol)
{
    Op::UnaryInstr(0);
}

template <typename Op, typename T>
PTO_INTERNAL void Unary1LNormMode(unsigned validRow, unsigned validCol)
{
    constexpr unsigned nRepeatElem = REPEAT_BYTE / sizeof(T);
    unsigned nElem = validRow * validCol;
    unsigned headRepeats = nElem / nRepeatElem;
    unsigned tailElements = nElem % nRepeatElem;
    // TODO 由于这里仅仅需要计数逻辑，是不是这里直接 Op::UnaryInst(headRepeats +
    // tailElements)即可，其他地方同理，且如果是0的地方，是不是整个文件什么操作都不需要

    Op::UnaryInstr(headRepeats);
    if (tailElements) {
        unsigned offset = headRepeats * nRepeatElem;
        Op::UnaryInstr(1);
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile>
PTO_INTERNAL void Unary2LCountMode(unsigned validRow, unsigned validCol)
{
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(0);
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeColVLAlign(unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / nRepeatElem;
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(headRepeats);
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeHead(unsigned validRow, unsigned nRepeatPerLine)
{
    if (nRepeatPerLine) {
        unsigned loop = nRepeatPerLine / REPEAT_MAX;
        unsigned remain = nRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            if (loop) {
                for (unsigned j = 0; j < loop; j++) {
                    Op::UnaryInstr(REPEAT_MAX);
                }
            }
            if (remain) {
                Op::UnaryInstr(remain);
            }
        }
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem, unsigned blockSizeElem>
PTO_INTERNAL void Unary2LNormModeTail(unsigned validRow, unsigned nRemainPerLine)
{
    constexpr unsigned dstStride = DstTile::RowStride / blockSizeElem;
    constexpr unsigned srcStride = SrcTile::RowStride / blockSizeElem;
    unsigned loop = 0;
    unsigned remain = validRow;
    constexpr bool strideOverFlag = (dstStride > REPEAT_STRIDE_MAX || srcStride > REPEAT_STRIDE_MAX);
    if constexpr (DstTile::Rows > pto::REPEAT_MAX || SrcTile::Rows > pto::REPEAT_MAX) {
        loop = validRow / REPEAT_MAX;
        for (uint32_t i = 0; i < loop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    // TODO 为什么这里的用法和其他地方不一样，有三个入参
                    Op::UnaryInstr(1, 1, 1);
                }
            } else {
                Op::UnaryInstr(REPEAT_MAX, dstStride, srcStride);
            }
        }
        remain = validRow % REPEAT_MAX;
    }
    if (remain) {
        if constexpr (strideOverFlag) {
            for (uint32_t j = 0; j < remain; j++) {
                Op::UnaryInstr(1, 1, 1);
            }
        } else {
            Op::UnaryInstr(remain, dstStride, srcStride);
        }
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeRowRpt(unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = DstTile::RowStride / blockSizeElem;
    constexpr unsigned srcStride = SrcTile::RowStride / blockSizeElem;
    constexpr bool condRowRpt = ((DstTile::Rows <= pto::REPEAT_MAX) && (dstStride <= REPEAT_STRIDE_MAX) &&
                                 (SrcTile::Rows <= pto::REPEAT_MAX) && (srcStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned loop = validCol / nRepeatElem;
        unsigned tailElements = validCol % nRepeatElem;
        for (uint32_t i = 0; i < loop; i++) {
            Op::UnaryInstr(validRow, dstStride, srcStride);
        }

        if (tailElements) {
            Op::UnaryInstr(validRow, dstStride, srcStride);
        }
    } else {
        unsigned nRepeatPerLine = validCol / nRepeatElem;
        unsigned remain = validCol % nRepeatElem;
        if constexpr (DstTile::Rows > nRepeatElem) {
            Unary2LNormModeHead<Op, T, DstTile, SrcTile, nRepeatElem>(validRow, nRepeatPerLine);
        }
        if (remain) {
            Unary2LNormModeTail<Op, T, DstTile, SrcTile, nRepeatElem, blockSizeElem>(validRow, remain);
        }
    }
}

template <typename Op, typename T, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LProcess(unsigned validRow, unsigned validCol)
{
    constexpr unsigned normColRepeat = DstTile::Cols / nRepeatElem;
    if constexpr ((normColRepeat > 1) && ((DstTile::Rows * normColRepeat) < SMALL_RPT)) {
        Unary2LCountMode<Op, T, DstTile, SrcTile>(validRow, validCol);
    } else if constexpr (DstTile::Rows < (normColRepeat + 1)) {
        unsigned tailElements = validCol % nRepeatElem;
        if (tailElements) {
            Unary2LCountMode<Op, T, DstTile, SrcTile>(validRow, validCol);
        } else {
            Unary2LNormModeColVLAlign<Op, T, DstTile, SrcTile, nRepeatElem>(validRow, validCol);
        }
    } else {
        Unary2LNormModeRowRpt<Op, T, DstTile, SrcTile, nRepeatElem>(validRow, validCol);
    }
}

template <typename T, typename Op, typename DstTile, typename SrcTile>
PTO_INTERNAL void TUnaryOp(unsigned validRow, unsigned validCol)
{   
    constexpr int nRepeatElem = REPEAT_BYTE / sizeof(T);
    constexpr bool isCombined = ((DstTile::ValidCol == DstTile::Cols) && (SrcTile::ValidCol == SrcTile::Cols)) ||
                                ((DstTile::Rows == 1) && (SrcTile::Rows == 1));

    if constexpr (isCombined) {
        constexpr unsigned totalRepeats = (DstTile::Rows * DstTile::Cols + nRepeatElem - 1) / nRepeatElem;
        if constexpr (totalRepeats > pto::REPEAT_MAX) {
            Unary1LCountMode<Op, T>(validRow, validCol);
        } else {
            Unary1LNormMode<Op, T>(validRow, DstTile::Cols);
        }
    } else {
        constexpr bool isSameShape = (DstTile::Cols == SrcTile::Cols) && (DstTile::Rows == SrcTile::Rows);
        if constexpr (isSameShape) {
            if ((validCol == DstTile::Cols) || (validRow == 1)) {
                unsigned totalRepeats = (validRow * validCol + nRepeatElem - 1) / nRepeatElem;
                if (totalRepeats > pto::REPEAT_MAX) {
                    Unary1LCountMode<Op, T>(validRow, validCol);
                } else {
                    Unary1LNormMode<Op, T>(validRow, validCol);
                }
            } else {
                Unary2LProcess<Op, T, DstTile, SrcTile, nRepeatElem>(validRow, validCol);
            }
        } else {
            Unary2LProcess<Op, T, DstTile, SrcTile, nRepeatElem>(validRow, validCol);
        }
    }
}

template <typename T, typename Op, typename DstTile, typename SrcTile, bool floatOnly = true>
PTO_INTERNAL void runUnaryOp(DstTile &dst, SrcTile &src)
{
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    TUnaryOp<T, Op, DstTile, SrcTile>(dstValidRow, dstValidCol);
}
} // namespace pto

#endif