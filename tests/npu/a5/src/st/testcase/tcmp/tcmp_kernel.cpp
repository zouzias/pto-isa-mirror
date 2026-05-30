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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int DstRows, int DstCols, int Src0Rows, int Src0Cols, int Src1Rows, int Src1Cols, int ValidRows, int ValidCols, CmpMode cmpMode>
__global__ AICORE void runTCmp(__gm__ uint8_t *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, ValidRows, ValidCols),
        pto::Stride(Src0Rows * Src0Cols, Src0Rows * Src0Cols, Src0Rows * Src0Cols, Src0Cols, 1));
    GlobalData src1Global(src1, pto::Shape(1, 1, 1, ValidRows, ValidCols),
        pto::Stride(Src1Rows * Src1Cols, Src1Rows * Src1Cols, Src1Rows * Src1Cols, Src1Cols, 1));
    using GlobalDataDst = GlobalTensor<uint8_t, DynShape, DynStride>;
    int ValidColsDst = (ValidCols + 7) / 8;
    GlobalDataDst dstGlobal(out, pto::Shape(1, 1, 1, ValidRows, ValidColsDst),
        pto::Stride(DstRows * DstCols, DstRows * DstCols, DstRows * DstCols, DstCols, 1));
    using dstTileData = Tile<TileType::Vec, uint8_t, DstRows, DstCols, BLayout::RowMajor, -1, -1>;
    using src0TileData = Tile<TileType::Vec, T, Src0Rows, Src0Cols, BLayout::RowMajor, -1, -1>;
    using src1TileData = Tile<TileType::Vec, T, Src1Rows, Src1Cols, BLayout::RowMajor, -1, -1>;
    dstTileData dstTile(ValidRows, ValidColsDst);
    src0TileData src0Tile(ValidRows, ValidCols);
    src1TileData src1Tile(ValidRows, ValidCols);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<src0TileData::Numel * sizeof(T)>(src1Tile);
    TASSIGN<(src0TileData::Numel + src1TileData::Numel) * sizeof(T)>(dstTile);
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCMP(dstTile, src0Tile, src1Tile, cmpMode);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int DstRows, int DstCols, int Src0Rows, int Src0Cols, int Src1Rows, int Src1Cols, int ValidRows, int ValidCols, CmpMode cmpMode, bool isHalf = false>
void LaunchTCmp(uint8_t *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && isHalf) {
        runTCmp<half, DstRows, DstCols, Src0Rows, Src0Cols, Src1Rows, Src1Cols, ValidRows, ValidCols, cmpMode>
            <<<1, nullptr, stream>>>((out), (half *)(src0), (half *)(src1));
    } else {
        runTCmp<T, DstRows, DstCols, Src0Rows, Src0Cols, Src1Rows, Src1Cols, ValidRows, ValidCols, cmpMode>
            <<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTCmp<aclFloat16, 32, 32, 32, 32, 32, 32, 32, 32, CmpMode::EQ, true>(uint8_t *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                  void *stream);
template void LaunchTCmp<float, 8, 64, 8, 64, 8, 64, 8, 64, CmpMode::GT>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 4, 64, 4, 64, 4, 64, 4, 64, CmpMode::NE>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTCmp<int32_t, 128, 128, 128, 128, 128, 128, 64, 64, CmpMode::LT>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                 void *stream);
template void LaunchTCmp<int32_t, 64, 64, 64, 64, 64, 64, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                               void *stream);
template void LaunchTCmp<int32_t, 16, 32, 16, 32, 16, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<float, 128, 128, 128, 128, 128, 128, 64, 64, CmpMode::LE>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 77, 32, 77, 80, 77, 80, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                               void *stream);
template void LaunchTCmp<int32_t, 32, 32, 32, 32, 32, 32, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                               void *stream);
template void LaunchTCmp<int16_t, 32, 32, 32, 32, 32, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                                               void *stream);
template void LaunchTCmp<int16_t, 77, 32, 77, 80, 77, 80, 32, 32, CmpMode::LE>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                               void *stream);
template void LaunchTCmp<int32_t, 66, 32, 66, 88, 66, 80, 66, 79, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                               void *stream);
