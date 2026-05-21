/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include <type_traits>

#include "acl/acl.h"

using namespace pto;

namespace {

template <typename TileData>
AICORE inline void initTNegTiles(TileData &srcTile, TileData &dstTile)
{
    constexpr uint32_t kSrcTileAddr = 0x0;
    constexpr uint32_t kDstTileAddr = 0x20000;
    TASSIGN(srcTile, kSrcTileAddr);
    TASSIGN(dstTile, kDstTileAddr);
}

template <typename TileData>
AICORE inline void runTNegCore(TileData &dstTile, TileData &srcTile)
{
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TNEG(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTNegKernel(T *out, T *src, void *stream)
{
    runTNeg<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src);
}

} // namespace

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTNeg(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using TensorShape = Shape<1, 1, 1, kGRows_, kGCols_>;
    using TensorStride = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, TensorShape, TensorStride>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData dstTile(kTRows_, kTCols_);
    TileData srcTile(kTRows_, kTCols_);
    initTNegTiles(srcTile, dstTile);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    runTNegCore(dstTile, srcTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTNeg(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        launchTNegKernel<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src), stream);
    } else {
        launchTNegKernel<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src, stream);
    }
}

template void LaunchTNeg<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTNeg<int32_t, 32, 32, 32, 32>(int32_t *out, int32_t *src, void *stream);
template void LaunchTNeg<aclFloat16, 32, 64, 32, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTNeg<int16_t, 64, 16, 64, 16>(int16_t *out, int16_t *src, void *stream);
