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
namespace TRowExpandTest {

template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
__global__ AICORE void runROWEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, rows, srcCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, srcCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, rows, srcCols, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, rows, dstCols>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCols, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, rows, dstCols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(rows, 1);
    DstTileData dstTile(rows, dstValidCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0xF000); // UB最大到0x40000

    int offset = 0;
    GlobalData srcGlobal(src + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPAND(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    out = dstGlobal.data();
}

// [32,1] fp32 ColMajor (contiguous), TROWEXPAND to UB tile [32,8] RowMajor (valid 32x1), TSTORE to GM shape
// (32,1) with stride (256,1) in elements.
template <typename T, uint32_t kRows, uint32_t kExpandCols, uint32_t kGmRowStrideElems>
__global__ AICORE void runROWEXPAND_ColMajor_StridedGm(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using SrcDynShapeDim5 = Shape<1, 1, 1, kRows, 1>;
    using SrcDynStridDim5 = pto::Stride<1, 1, kRows, 1, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcDynShapeDim5, SrcDynStridDim5, Layout::DN>;
    using SrcTileData = Tile<TileType::Vec, T, kRows, 1, BLayout::ColMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, kRows, 1>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGmRowStrideElems, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, kRows, kExpandCols, BLayout::RowMajor, -1, -1>;

    SrcTileData srcTile(kRows, 1);
    DstTileData dstTile(kRows, 1);

    constexpr uint32_t kSrcBytes = sizeof(T) * kRows * 1;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, kSrcBytes);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPAND(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    out = dstGlobal.data();
}

template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
void launchTROWEXPAND(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPAND<half, rows, srcCols, dstValidCols, dstCols><<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runROWEXPAND<T, rows, srcCols, dstValidCols, dstCols><<<1, nullptr, stream>>>(out, src);
    }
}

template <typename T, uint32_t kRows, uint32_t kExpandCols, uint32_t kGmRowStrideElems>
void launchTROWEXPANDColMajorStridedGm(T *out, T *src, void *stream)
{
    runROWEXPAND_ColMajor_StridedGm<T, kRows, kExpandCols, kGmRowStrideElems><<<1, nullptr, stream>>>(out, src);
}

template void launchTROWEXPAND<aclFloat16, 16, 16, 512, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 256, 256>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<float, 16, 8, 128, 128>(float *out, float *src, void *stream);
template void launchTROWEXPAND<aclFloat16, 16, 16, 511, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 255, 256>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<float, 16, 8, 127, 128>(float *out, float *src, void *stream);
template void launchTROWEXPANDColMajorStridedGm<float, 32, 8, 256>(float *out, float *src, void *stream);
} // namespace TRowExpandTest