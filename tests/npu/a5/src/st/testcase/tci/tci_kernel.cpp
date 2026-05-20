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

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse, int mode>
__global__ AICORE void runTci(__gm__ T __out__ *out, T S)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData dstTile(kTRows_, kTCols_);

    TASSIGN(dstTile, 0x0);

    GlobalData dstGlobal(out);

    if (mode == 0) {
        TCI<TileData, T, reverse>(dstTile, S);
    } else {
        using TileData_tmp = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512>;
        TileData_tmp tmpTile;
        TASSIGN(tmpTile, 0x20000);
        TCI<TileData, TileData_tmp, T, reverse>(dstTile, S, tmpTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse, int mode>
void LaunchTci(T *out, T S, void *stream)
{
    runTci<T, kGRows_, kGCols_, kTRows_, kTCols_, reverse, mode><<<1, nullptr, stream>>>((T *)(out), S);
}

template void LaunchTci<float, 32, 512, 32, 512, 0, 0>(float *out, float S = 1.0f, void *stream);
template void LaunchTci<float, 512, 32, 512, 32, 1, 0>(float *out, float S = 1.0f, void *stream);
template void LaunchTci<float, 128, 128, 128, 128, 0, 0>(float *out, float S = 1.0f, void *stream);