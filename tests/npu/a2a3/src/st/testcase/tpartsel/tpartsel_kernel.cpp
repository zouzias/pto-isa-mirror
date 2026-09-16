/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
__global__ AICORE void runTPartSel(
    __gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    constexpr unsigned maskRow = Rows;
    constexpr unsigned maskCol = ((((Cols + 7) / 8) + 31) / 32) * 32;
    constexpr unsigned maskVRow = ValidRows;
    constexpr unsigned maskVCol = (ValidCols + 7) / 8;

    using DynShapeDim5 = pto::Shape<1, 1, 1, ValidRows, ValidCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, ValidCols, 1>;

    using DynShapeDim5m = pto::Shape<1, 1, 1, maskVRow, maskVCol>;
    using DynStridDim5m = pto::Stride<1, 1, 1, maskVCol, 1>;

    using GlobalData = pto::GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    constexpr unsigned src0Cols = Cols + 32 / sizeof(T);
    constexpr unsigned src1Cols = Cols + 64 / sizeof(T);
    using DstTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using Src0Tile = Tile<TileType::Vec, T, Rows + 1, src0Cols, BLayout::RowMajor, -1, -1>;
    using Src1Tile = Tile<TileType::Vec, T, Rows + 2, src1Cols, BLayout::RowMajor, -1, -1>;

    using MaskGlobal = pto::GlobalTensor<uint8_t, DynShapeDim5m, DynStridDim5m>;
    using MaskTile = Tile<TileType::Vec, uint8_t, maskRow, maskCol, BLayout::RowMajor, -1, -1>;

    using TmpTile = Tile<TileType::Vec, uint32_t, 1, 16>;

    Src0Tile src0Tile(ValidRows, ValidCols);
    Src1Tile src1Tile(ValidRows, ValidCols);
    DstTile dstTile(ValidRows, ValidCols);
    MaskTile maskTile(maskVRow, maskVCol);
    TmpTile tmpTile;

    constexpr uint64_t src0Size = (Rows + 1) * src0Cols * sizeof(T);
    constexpr uint64_t src1Size = (Rows + 2) * src1Cols * sizeof(T);
    constexpr uint64_t dstSize = Rows * Cols * sizeof(T);
    constexpr uint64_t maskOffset = src0Size + src1Size + dstSize;
    constexpr uint64_t tmpOffset = ((maskOffset + maskRow * maskCol + 511) / 512) * 512;
    static_assert(tmpOffset + 64 <= 184 * 1024, "UB size overflow.");

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, src0Size);
    TASSIGN(dstTile, src0Size + src1Size);
    TASSIGN(maskTile, maskOffset);
    TASSIGN(tmpTile, tmpOffset);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);
    MaskGlobal maskGlobal(mask);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(maskTile, maskGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TPARTSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTPartSel(T* out, uint8_t* mask, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPartSel<half, Rows, Cols, ValidRows, ValidCols>
            <<<1, nullptr, stream>>>((half*)(out), mask, (half*)(src0), (half*)(src1));
    } else {
        runTPartSel<T, Rows, Cols, ValidRows, ValidCols><<<1, nullptr, stream>>>(out, mask, src0, src1);
    }
}

template void LaunchTPartSel<float, 4, 128, 4, 128>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTPartSel<float, 4, 192, 3, 131>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTPartSel<aclFloat16, 4, 128, 4, 128>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPartSel<aclFloat16, 4, 192, 3, 131>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPartSel<int32_t, 4, 64, 3, 37>(
    int32_t* out, uint8_t* mask, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTPartSel<int16_t, 4, 64, 3, 37>(
    int16_t* out, uint8_t* mask, int16_t* src0, int16_t* src1, void* stream);
