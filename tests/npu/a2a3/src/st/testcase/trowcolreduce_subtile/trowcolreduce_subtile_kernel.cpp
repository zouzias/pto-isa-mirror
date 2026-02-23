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
#include <pto/npu/a2a3/subtile/subtile_tile.hpp>
#include <pto/npu/a2a3/TColSum.hpp>
#include "acl/acl.h"


using namespace pto;


// Row-reduce using vcgadd/vcgmax for packed block (cols = 32B/sizeof(T))

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTRowReduceSumBlock(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, vRows, vCols>, Stride<1, 1, 1, vCols, 1>>;
    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, vCols, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vRows, vCols);

    __ubuf__ T *srcPtr = (__ubuf__ T *)srcTile.data();
    __ubuf__ T *dstPtr = (__ubuf__ T *)dstTile.data();

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

    constexpr uint16_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    uint16_t srcRptStride = TileDataSrc::RowStride / blockSizeElem;
    uint16_t dstRptStride = TileDataDst::RowStride / blockSizeElem;
    uint8_t vRowsBlock = (uint8_t)(vRows / 8);

    pipe_barrier(PIPE_ALL);
    // vcgadd: group reduce per row (packed 32B block)
    vcgadd(dstPtr, srcPtr, vRowsBlock, dstRptStride, (uint16_t)(srcRptStride * 8), (uint16_t)(srcRptStride * 8));
    pipe_barrier(PIPE_ALL);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTRowReduceMaxBlock(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, vRows, vCols>, Stride<1, 1, 1, vCols, 1>>;
    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, vCols, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vRows, vCols);

    __ubuf__ T *srcPtr = (__ubuf__ T *)srcTile.data();
    __ubuf__ T *dstPtr = (__ubuf__ T *)dstTile.data();

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

    constexpr uint16_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    uint16_t srcRptStride = TileDataSrc::RowStride / blockSizeElem;
    uint16_t dstRptStride = TileDataDst::RowStride / blockSizeElem;
    uint8_t vRowsBlock = (uint8_t)(vRows / 8);

    pipe_barrier(PIPE_ALL);
    // vcgmax: group reduce per row (packed 32B block)
    vcgmax(dstPtr, srcPtr, vRowsBlock, dstRptStride, (uint16_t)(srcRptStride * 8), (uint16_t)(srcRptStride * 8));
    pipe_barrier(PIPE_ALL);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// Col-reduce (sum) by forwarding: dst=src0, repeat stride = 0

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTColReduceSumForward(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, 1, vCols>, Stride<1, 1, 1, vCols, 1>>;
    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, 1, vCols, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(1, vCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

    __ubuf__ T *dstPtr = (__ubuf__ T *)dstTile.data();
    __ubuf__ T *srcPtr = (__ubuf__ T *)srcTile.data();
    constexpr int srcStride = TileDataSrc::RowStride;
    constexpr int DTypeSize = sizeof(T);
    int lenBurst = (vCols * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

    // init dst with row0
    copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, lenBurst, 0, 0);
    pipe_barrier(PIPE_V);

    set_mask_norm();
    set_vector_mask(0, vCols);
    constexpr uint16_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    uint16_t srcRptStride = TileDataSrc::RowStride / blockSizeElem; // rowStride/8 for fp32
    // single vadd, repeat = vRows-1, repeat stride derived from rowStride
    vadd(dstPtr, dstPtr, srcPtr, vRows - 1, 1, 1, 1, 0, 0, srcRptStride);
    pipe_barrier(PIPE_V);

    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_ALL);

TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTRowReduceSumBlock(T *out, T *src, void *stream)
{
    runTRowReduceSumBlock<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTRowReduceMaxBlock(T *out, T *src, void *stream)
{
    runTRowReduceMaxBlock<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTColReduceSumForward(T *out, T *src, void *stream)
{
    runTColReduceSumForward<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src);
}

// rowStride=16, cols=8 (32B block for fp32), rows<=255
template void LaunchTRowReduceSumBlock<float, 128, 16, 128, 8>(float *out, float *src, void *stream);
template void LaunchTRowReduceMaxBlock<float, 128, 16, 128, 8>(float *out, float *src, void *stream);
// col-reduce sum -> output 1x8
template void LaunchTColReduceSumForward<float, 128, 16, 128, 8>(float *out, float *src, void *stream);

