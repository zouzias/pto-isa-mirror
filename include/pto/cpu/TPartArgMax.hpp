/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGMAX_HPP
#define TPARTARGMAX_HPP

#include "TPartOp.hpp"

namespace pto {
<<<<<<< HEAD
template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
struct PartArgMaxOp {
    PTO_INTERNAL static void PartInstr(typename TileDataDstVal::TileDType dstVal, typename TileDataDstIdx::TileDType dstIdx,
                                       typename TileDataSrc0Val::TileDType src0Val, typename TileDataSrc0Idx::TileDType src0Idx,
                                       typename TileDataSrc1Val::TileDType src1Val, typename TileDataSrc1Idx::TileDType src1Idx,
                                       int DstOffset, int Src0Offset, int Src1Offset)
    {
        if (src0Val[Src0Offset] >= src1Val[Src1Offset]) {
            dstVal[DstOffset] = src0Val[Src0Offset];
            dstIdx[DstOffset] = src0Idx[Src0Offset];
        } else {
            dstVal[DstOffset] = src1Val[Src1Offset];
            dstIdx[DstOffset] = src1Idx[Src1Offset];
=======
template <typename TileData>
struct ValidRegion {
    int row;
    int col;
    
    ValidRegion(const TileData& tile) 
        : row(tile.GetValidRow()), col(tile.GetValidCol()) {}
};

template <typename DstVal, typename Src0Val, typename Src1Val>
static inline void CheckAndGetValidRegions(
    const DstVal& dstVal, const Src0Val& src0Val, const Src1Val& src1Val,
    int& dstRow, int& dstCol, int& src0Row, int& src0Col, int& src1Row, int& src1Col)
{
    using T = typename Src0Val::DType;
    TPartCheck<T, DstVal, Src0Val, Src1Val>(dstVal.GetValidRow(), dstVal.GetValidCol());
    dstRow = dstVal.GetValidRow();
    dstCol = dstVal.GetValidCol();
    src0Row = src0Val.GetValidRow();
    src0Col = src0Val.GetValidCol();
    src1Row = src1Val.GetValidRow();
    src1Col = src1Val.GetValidCol();
}

template <typename ValType, typename IdxType>
struct MaxWithIndexOp {
    static inline void apply(
        ValType& dstVal, IdxType& dstIdx,
        ValType src0Val, IdxType src0Idx,
        ValType src1Val, IdxType src1Idx)
    {
        if (src0Val >= src1Val) {
            dstVal = src0Val;
            dstIdx = src0Idx;
        } else {
            dstVal = src1Val;
            dstIdx = src1Idx;
>>>>>>> 7f052fe0 (Add TPartArgMax, TPartArgMin, TPow, TPows for CPU SIM)
        }
    }
};

template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
<<<<<<< HEAD
PTO_INTERNAL void TPARTARGMAX_IMPL(TileDataDstVal &dstVal, TileDataDstIdx &dstIdx, TileDataSrc0Val &src0Val, TileDataSrc0Idx &src0Idx, 
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
    TPartInstr2<PartArgMaxOp<TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>,
                TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>(
        dstVal.data(), dstIdx.data(), src0Val.data(), src0Idx.data(), src1Val.data(), src1Idx.data(),
        DstValidRow, DstValidCol, Src0ValidRow, Src0ValidCol, Src1ValidRow, Src1ValidCol);
}
} // namespace pto
#endif
=======
struct PartArgMaxOp {
    PTO_INTERNAL static void PartInstr(typename TileDataDstVal::TileDType dstVal,
                                       typename TileDataDstIdx::TileDType dstIdx,
                                       typename TileDataSrc0Val::TileDType src0Val,
                                       typename TileDataSrc0Idx::TileDType src0Idx,
                                       typename TileDataSrc1Val::TileDType src1Val,
                                       typename TileDataSrc1Idx::TileDType src1Idx,
                                       int DstOffset, int Src0Offset, int Src1Offset)
    {
        MaxWithIndexOp<
            typename TileDataSrc0Val::DType,
            typename TileDataSrc0Idx::DType
        >::apply(
            dstVal[DstOffset], dstIdx[DstOffset],
            src0Val[Src0Offset], src0Idx[Src0Offset],
            src1Val[Src1Offset], src1Idx[Src1Offset]
        );
    }
};

template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
          typename TileDataSrc1Val, typename TileDataSrc1Idx>
PTO_INTERNAL void TPARTARGMAX_IMPL(TileDataDstVal &dstVal, TileDataDstIdx &dstIdx, TileDataSrc0Val &src0Val,
                                   TileDataSrc0Idx &src0Idx, TileDataSrc1Val &src1Val, TileDataSrc1Idx &src1Idx)
{
    int dstRow, dstCol, src0Row, src0Col, src1Row, src1Col;
    CheckAndGetValidRegions(dstVal, src0Val, src1Val, dstRow, dstCol, src0Row, src0Col, src1Row, src1Col);
    
    TPartInstr2<PartArgMaxOp<TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx,
                             TileDataSrc1Val, TileDataSrc1Idx>,
                TileDataDstVal, TileDataDstIdx, TileDataSrc0Val, TileDataSrc0Idx, TileDataSrc1Val, TileDataSrc1Idx>(
        dstVal.data(), dstIdx.data(), src0Val.data(), src0Idx.data(), src1Val.data(), src1Idx.data(),
        dstRow, dstCol, src0Row, src0Col, src1Row, src1Col);
}
} // namespace pto

#endif // TPARTARGMAX_HPP
>>>>>>> 7f052fe0 (Add TPartArgMax, TPartArgMin, TPow, TPows for CPU SIM)
