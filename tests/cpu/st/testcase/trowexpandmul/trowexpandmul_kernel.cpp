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
#include <algorithm>

using namespace pto;

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
AICORE void runTRowExpandMul(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRow, validCol>, Stride<1, 1, 1, validCol, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

    TileDataDst src0Tile(validRow, validCol);
    TileDataDst dstTile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, 1);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    std::fill_n(out, static_cast<size_t>(Row) * static_cast<size_t>(Col), static_cast<T>(0));
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    for (size_t i = 0; i < static_cast<size_t>(TileDataDst::Numel); ++i) {
        dstTile.data()[i] = src0Tile.data()[i];
    }
    if constexpr (src0eqdst) {
        TROWEXPANDMUL(dstTile, dstTile, src1Tile);
    } else {
        TROWEXPANDMUL(dstTile, src1Tile, dstTile);
    }
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
AICORE void runTRowExpandMul2(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t src1Col = 32 / sizeof(T);
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRow, validCol>, Stride<1, 1, 1, validCol, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, src1Col>, Stride<1, 1, 1, src1Col, 1>>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, src1Col, BLayout::RowMajor, -1, -1>;

    TileDataDst src0Tile(validRow, validCol);
    TileDataDst dstTile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, src1Col);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    std::fill_n(out, static_cast<size_t>(Row) * static_cast<size_t>(Col), static_cast<T>(0));
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    for (size_t i = 0; i < static_cast<size_t>(TileDataDst::Numel); ++i) {
        dstTile.data()[i] = src0Tile.data()[i];
    }
    if constexpr (src0eqdst) {
        TROWEXPANDMUL(dstTile, dstTile, src1Tile);
    } else {
        TROWEXPANDMUL(dstTile, src1Tile, dstTile);
    }
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandMul(T *out, T *src0, T *src1, void *stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTRowExpandMul<half, validRow, validCol, Row, Col, src0eqdst>((half *)(out), (half *)(src0), (half *)(src1));
    } else {
        runTRowExpandMul<T, validRow, validCol, Row, Col, src0eqdst>(out, src0, src1);
    }
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandMul2(T *out, T *src0, T *src1, void *stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTRowExpandMul2<half, validRow, validCol, Row, Col, src0eqdst>((half *)(out), (half *)(src0), (half *)(src1));
    } else {
        runTRowExpandMul2<T, validRow, validCol, Row, Col, src0eqdst>(out, src0, src1);
    }
}

template void launchTRowExpandMul<float, 16, 16, 16, 16, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul<float, 16, 16, 32, 32, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul<aclFloat16, 16, 16, 16, 16, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void launchTRowExpandMul<aclFloat16, 16, 16, 32, 32, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void launchTRowExpandMul<float, 1, 16384, 1, 16384, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul<float, 2048, 1, 2048, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul2<float, 16, 16, 16, 16, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul2<float, 16, 16, 32, 32, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul2<aclFloat16, 16, 16, 16, 16, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void launchTRowExpandMul2<aclFloat16, 16, 16, 32, 32, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
template void launchTRowExpandMul2<float, 1, 16384, 1, 16384, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul2<float, 2048, 1, 2048, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul<float, 16, 16, 16, 16, false>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandMul2<float, 16, 16, 16, 16, false>(float *out, float *src0, float *src1, void *stream);
