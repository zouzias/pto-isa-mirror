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
#include <gtest/gtest.h>
#include <cmath>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
AICORE void runTColMin(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLMIN(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();

    // accuracy compare
    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTColMin(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTColMin<half, kGRows_, kGCols_, kTRows_, kTCols_, profiling, accuracy>((half *)(out), (half *)(src));
    else
        runTColMin<T, kGRows_, kGCols_, kTRows_, kTCols_, profiling, accuracy>(out, src);
}

// float 64x64: numRepeatPerLine=1, (64-1)*1=63 repeats → effective=62 → 13+19+62*2=156
template void LaunchTColMin<float, 64, 64, 64, 64, 156.0f, 1.0f>(float *out, float *src, void *stream);
// half 64x64: numRemain=64 → 63 masked → effective=62 → 13+21+62*2=158
template void LaunchTColMin<aclFloat16, 64, 64, 64, 64, 158.0f, 1.0f>(aclFloat16 *out, aclFloat16 *src, void *stream);
// int16 64x64: 63 masked → effective=62 → 13+17+62*1=92
template void LaunchTColMin<int16_t, 64, 64, 64, 64, 92.0f, 1.0f>(int16_t *out, int16_t *src, void *stream);
// half 16x256: numRepeatPerLine=2, 15*2=30 → effective=29 → 13+21+29*2=92
template void LaunchTColMin<aclFloat16, 16, 256, 16, 256, 92.0f, 1.0f>(aclFloat16 *out, aclFloat16 *src, void *stream);
