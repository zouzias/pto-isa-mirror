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

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW,
    int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTSELS(__gm__ T __out__ *out, __gm__ TMask *mask, __gm__ T __in__ *src, T __in__ scalar) {
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using GlobalDataMask = GlobalTensor<TMask, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalDataMask maskGlobal(mask, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(maskTileH * maskTileW, maskTileH * maskTileW, maskTileH * maskTileW, maskTileW, 1));
    GlobalData srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));
    
    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, TMask, maskTileH, maskTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile;
    TileDataMask maskTile;
    TileDataSrc srcTile;
    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t srcSize = sizeof(T) * srcTileH * srcTileW;
    size_t maskSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t totalSize = dstSize + srcSize + maskSize;
    size_t dstOffset = totalSize * block_idx;
    size_t srcOffset = totalSize * block_idx + dstSize;
    size_t maskOffset = totalSize * block_idx + dstSize + srcSize;
    TASSIGN(dstTile, dstOffset);
    TASSIGN(maskTile, maskOffset);
    TASSIGN(srcTile, srcOffset);

    TLOAD(maskTile, maskGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSELS(dstTile, maskTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW,
    int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream)
{
    runTSELS<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, mask, src, scalar);
}

template void LaunchTSels<float, uint8_t, 8, 8, 8, 32, 8, 8, 8, 8>
    (float *out, uint8_t *mask, float *src, float scalar, void *stream);
