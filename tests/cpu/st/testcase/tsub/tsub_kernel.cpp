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
AICORE void runTSub( __gm__ T __out__ *out, __gm__ T __in__ *src0,  __gm__ T __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData src1Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    PipeEvent<PIPE_MTE2, PIPE_V> evLoad0ToVec;
    PipeEvent<PIPE_MTE2, PIPE_V> evLoad1ToVec;
    PipeEvent<PIPE_V, PIPE_MTE3> evComputeToStore;
    evLoad0ToVec = TLOAD(src0Tile, src0Global);
    evLoad1ToVec = TLOAD(src1Tile, src1Global);
    evComputeToStore = TSUB(dstTile, src0Tile, src1Tile, evLoad0ToVec, evLoad1ToVec);
    TSTORE(dstGlobal, dstTile, evComputeToStore);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTSub(T *out, T *src0, T *src1, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTSub<half, kGRows_, kGCols_, kTRows_, kTCols_>((half*)(out),
                                                                                  (half*)(src0),
                                                                                  (half*)(src1));
    else 
        runTSub<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src0, src1);
}

template void LaunchTSub<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTSub<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTSub<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void LaunchTSub<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
