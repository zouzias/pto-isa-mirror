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

template <bool isInPlace, typename TileData>
AICORE inline void initTLogTiles(TileData &srcTile, TileData &dstTile)
{
    constexpr uint32_t kSrcTileAddr = 0x0;
    constexpr uint32_t kDstTileAddr = isInPlace ? 0x0 : 0x20000;
    TASSIGN(srcTile, kSrcTileAddr);
    TASSIGN(dstTile, kDstTileAddr);
}

} // namespace

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
__global__ AICORE void runTLog(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using TensorShape = Shape<1, 1, 1, kGRows_, kGCols_>;
    using TensorStride = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, TensorShape, TensorStride>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData dstTile(kTRows_, kTCols_);
    TileData srcTile(kTRows_, kTCols_);
    initTLogTiles<isInPlace>(srcTile, dstTile);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TLOG(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void LaunchTLog(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTLog<half, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src));
    else
        runTLog<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTLog<float, 64, 64, 64, 64, true>(float *out, float *src, void *stream);
template void LaunchTLog<float, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTLog<aclFloat16, 64, 64, 64, 64, true>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTLog<aclFloat16, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
