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
#include "../../../../../include/pto/cpu/TPartArgKernelCommon.hpp"
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
    using Common = TPartArgKernelCommon<kRows, kCols, kValidRows1, kValidCols1>;

    typename Common::TileVal0 src0ValTile(kRows, kCols);
    typename Common::TileVal1 src1ValTile(kValidRows1, kValidCols1);
    typename Common::TileVal0 dstValTile(kRows, kCols);

    typename Common::TileIdx0 src0IdxTile(kRows, kCols);
    typename Common::TileIdx1 src1IdxTile(kValidRows1, kValidCols1);
    typename Common::TileIdx0 dstIdxTile(kRows, kCols);

    typename Common::GlobalDataVal src0ValGlobal(src0Val);
    typename Common::GlobalData1Val src1ValGlobal(src1Val);
    typename Common::GlobalDataVal dstValGlobal(outVal);

    typename Common::GlobalDataIdx src0IdxGlobal(src0Idx);
    typename Common::GlobalData1Idx src1IdxGlobal(src1Idx);
    typename Common::GlobalDataIdx dstIdxGlobal(outIdx);

    Common::AssignTiles(src0ValTile, src1ValTile, dstValTile, src0IdxTile, src1IdxTile, dstIdxTile);
    Common::LoadTiles(src0ValTile, src1ValTile, src0IdxTile, src1IdxTile, src0ValGlobal, src1ValGlobal, src0IdxGlobal, src1IdxGlobal);
    TPARTARGMIN(dstValTile, dstIdxTile, src0ValTile, src0IdxTile, src1ValTile, src1IdxTile);
    Common::StoreTiles(dstValTile, dstIdxTile, dstValGlobal, dstIdxGlobal);

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
