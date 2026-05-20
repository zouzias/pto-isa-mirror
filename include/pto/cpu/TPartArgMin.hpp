/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGMIN_HPP
#define TPARTARGMIN_HPP

#include "TPartOp.hpp"

namespace pto {
template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
struct PartArgMinOp {
    PTO_INTERNAL static void PartInstr(typename TileDataDstVal::TileDType dstVal, typename TileDataDstIdx::TileDType dstIdx,
                                       typename TileDataSrc0Val::TileDType src0Val, typename TileDataSrc0Idx::TileDType src0Idx,
                                       typename TileDataSrc1Val::TileDType src1Val, typename TileDataSrc1Idx::TileDType src1Idx,
                                       int DstOffset, int Src0Offset, int Src1Offset)
    {
        if (src0Val[Src0Offset] <= src1Val[Src1Offset]) {
            dstVal[DstOffset] = src0Val[Src0Offset];
            dstIdx[DstOffset] = src0Idx[Src0Offset];
        } else {
            dstVal[DstOffset] = src1Val[Src1Offset];
            dstIdx[DstOffset] = src1Idx[Src1Offset];
        }
    }
};

template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
PTO_INTERNAL void TPARTARGMIN_IMPL(TileDataDstVal &dstVal, TileDataDstIdx &dstIdx, TileDataSrc0Val &src0Val, TileDataSrc0Idx &src0Idx,
                                   TileDataSrc1Val &src1Val, TileDataSrc1Idx &src1Idx)
{
    using T = typename TileDataSrc0Val::DType;
    TPartCheck<T, TileDataDstVal, TileDataSrc0Val, TileDataSrc1Val>(dstVal.GetValidRow(), dstVal.GetValidCol());
    int DstValidRow = dstVal.GetValidRow();
    int DstValidCol = dstVal.GetValidCol();
    int Src0ValidRow = src0Val.GetValidRow();
    int Src0ValidCol = src0Val.GetValidCol();
    int Src1ValidRow = src1Val.GetValidRow();
    int Src1ValidCol = src1Val.GetValidCol();
    TPartInstr2<PartArgMinOp<TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>,
                TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>(
        dstVal.data(), dstIdx.data(), src0Val.data(), src0Idx.data(), src1Val.data(), src1Idx.data(),
        DstValidRow, DstValidCol, Src0ValidRow, Src0ValidCol, Src1ValidRow, Src1ValidCol);
}
} // namespace pto
#endif