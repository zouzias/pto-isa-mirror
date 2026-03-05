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

template <typename T, typename U, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol>
__global__ AICORE void runtexpm1(__gm__ T __out__ *out, __gm__ U __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using SrcGlobalData = GlobalTensor<U, DynShapeDim5, pto::Stride<1, 1, srcRow, srcCol, 1>>;
    using DstGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, dstRow, dstCol, 1>>;
    SrcGlobalData srcGlobal(src, DynShapeDim5(validRow, validCol));
    DstGlobalData dstGlobal(out, DynShapeDim5(validRow, validCol));

    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    using SrcTileData = Tile<TileType::Vec, U, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcRow * srcCol * sizeof(U));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TEXPM1(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, typename U, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol>
void LaunchTExpM1(T *out, U *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && std::is_same_v<U, aclFloat16>) {
        runtexpm1<half, half, dstRow, dstCol, srcRow, srcCol, validRow, validCol>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src));
    } else if constexpr (std::is_same_v<T, aclFloat16>){
        runtexpm1<half, U, dstRow, dstCol, srcRow, srcCol, validRow, validCol>
            <<<1, nullptr, stream>>>((half *)(out), src);
    } else if constexpr (std::is_same_v<U, aclFloat16>) {
        runtexpm1<T, half, dstRow, dstCol, srcRow, srcCol, validRow, validCol>
            <<<1, nullptr, stream>>>(out, (half *)(src));
    }else {
        runtexpm1<T, U, dstRow, dstCol, srcRow, srcCol, validRow, validCol><<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTExpM1<float, float, 16, 64, 16, 64, 16, 64>(float *out, float *src, void *stream);
template void LaunchTExpM1<float, float, 1, 1024, 1, 1024, 1, 1024>(float *out, float *src, void *stream);
template void LaunchTExpM1<aclFloat16, aclFloat16, 32, 32, 32, 32, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExpM1<float, aclFloat16, 32, 32, 32, 32, 32, 32>(float *out, aclFloat16 *src, void *stream);
template void LaunchTExpM1<float, int32_t, 32, 32, 32, 32, 32, 32>(float *out, int32_t *src, void *stream);
template void LaunchTExpM1<float, int16_t, 32, 32, 32, 32, 32, 32>(float *out, int16_t *src, void *stream);
template void LaunchTExpM1<aclFloat16, int16_t, 32, 32, 32, 32, 32, 32>(aclFloat16 *out, int16_t *src, void *stream);
