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
#include <pto/common/pto_pipe.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTROWMIN(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using SrcTile = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kTRows_, 16, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(kTRows_, kTCols_);
    DstTile dstTile(kTRows_, 1);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x11000);

    GlobalData srcGlobal(src, DynShapeDim5(kTRows_, kTCols_), DynStridDim5(kTRows_, kTCols_));
    GlobalData dstGlobal(out, DynShapeDim5(kTRows_, 1), DynStridDim5(kTRows_, 1));

    PipeEvent<PIPE_MTE2, PIPE_V> evLoadToVec;
    PipeEvent<PIPE_V, PIPE_MTE3> evComputeToStore;
    evLoadToVec = TLOAD(srcTile, srcGlobal);
    evComputeToStore = TROWMIN(dstTile, srcTile, srcTile, evLoadToVec);
    TSTORE(dstGlobal, dstTile, evComputeToStore);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWMIN(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWMIN<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src));
    } else {
        runTROWMIN<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src);
    }
}

template void LaunchTROWMIN<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWMIN<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWMIN<aclFloat16, 161, 161, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWMIN<float, 77, 81, 32, 16>(float *out, float *src, void *stream);
template void LaunchTROWMIN<float, 32, 32, 32, 16>(float *out, float *src, void *stream);
