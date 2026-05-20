/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kValidRows1 = 32;
constexpr int kValidCols1 = 32;

} // namespace

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
AICORE void runTPARTARGMIN(__gm__ float __out__ *outVal, __gm__ float __in__ *src0Val, __gm__ float __in__ *src1Val,
                           __gm__ uint32_t __out__ *outIdx, __gm__ uint32_t __in__ *src0Idx, __gm__ uint32_t __in__ *src1Idx)
{
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;
    using GlobalDataVal = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using GlobalDataIdx = GlobalTensor<uint32_t, DynShapeDim5, DynStridDim5>;
    using GlobalData1Val = GlobalTensor<float, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;
    using GlobalData1Idx = GlobalTensor<uint32_t, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;

    using TileT = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileIdx = Tile<TileType::Vec, uint32_t, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileT src0ValTile(kRows, kCols);
    TileT src1ValTile(kValidRows1, kValidCols1);
    TileT dstValTile(kRows, kCols);
    TileIdx src0IdxTile(kRows, kCols);
    TileIdx src1IdxTile(kValidRows1, kValidCols1);
    TileIdx dstIdxTile(kRows, kCols);

    GlobalDataVal src0ValGlobal(src0Val);
    GlobalData1Val src1ValGlobal(src1Val);
    GlobalDataVal dstValGlobal(outVal);
    GlobalDataIdx src0IdxGlobal(src0Idx);
    GlobalData1Idx src1IdxGlobal(src1Idx);
    GlobalDataIdx dstIdxGlobal(outIdx);

    TASSIGN(src0ValTile, 0);
    TASSIGN(src1ValTile, kRows * kCols * sizeof(float));
    TASSIGN(dstValTile, 2 * kRows * kCols * sizeof(float));
    TASSIGN(src0IdxTile, 3 * kRows * kCols * sizeof(float));
    TASSIGN(src1IdxTile, 3 * kRows * kCols * sizeof(float) + kRows * kCols * sizeof(uint32_t));
    TASSIGN(dstIdxTile, 3 * kRows * kCols * sizeof(float) + 2 * kRows * kCols * sizeof(uint32_t));

    TLOAD(src0ValTile, src0ValGlobal);
    TLOAD(src1ValTile, src1ValGlobal);
    TLOAD(src0IdxTile, src0IdxGlobal);
    TLOAD(src1IdxTile, src1IdxGlobal);
    TPARTARGMIN(dstValTile, dstIdxTile, src0ValTile, src0IdxTile, src1ValTile, src1IdxTile);
    TSTORE(dstValGlobal, dstValTile);
    TSTORE(dstIdxGlobal, dstIdxTile);
    outVal = dstValGlobal.data();
    outIdx = dstIdxGlobal.data();
}

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTARGMIN(float *outVal, float *src0Val, float *src1Val, uint32_t *outIdx, uint32_t *src0Idx, uint32_t *src1Idx, void *stream)
{
    (void)stream;
    runTPARTARGMIN<kRows, kCols, kValidRows1, kValidCols1>(outVal, src0Val, src1Val, outIdx, src0Idx, src1Idx);
}

template void LaunchTPARTARGMIN<kRows, kCols, kValidRows1, kValidCols1>(float *outVal, float *src0Val, float *src1Val, uint32_t *outIdx,
                                                                        uint32_t *src0Idx, uint32_t *src1Idx, void *stream);