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

using namespace pto;

template <typename T, typename TIdx, int axis, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols, typename LaunchFn>
AICORE void runTARGREDUCEOP(__gm__ T __out__ *out, __gm__ T __in__ *src, LaunchFn fn)
{
    using GlobShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalMat = GlobalTensor<T, GlobShapeDim5, GlobStridDim5>;

    using TileMatDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileMatSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    constexpr int dstKTRows = axis == 0 ? 1 : kTRows;
    constexpr int dstKTCols = axis == 0 ? kTCols : 1;
    TileMatSrc srcTile(kTRows, kTCols);
    TileMatSrc tmpTile(kTRows, kTCols);
    TileMatDst dstTile(dstKTRows, dstKTCols);

    GlobalMat srcGlobal(src, GlobShapeDim5(kTRows, kTCols), GlobStridDim5(iRow, iCol));
    GlobalMat dstGlobal(out, GlobShapeDim5(dstKTRows, dstKTCols), GlobStridDim5(oRow, oCol));

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, iRow * iCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
    fn(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMAX(T *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<half, TIdx, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half *)(out), (half *)(src),
            [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMAX(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<T, TIdx, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, tmp, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMAX(dst, src, tmp); });
    }
}

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMIN(T *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, TIdx, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half *)(out), (half *)(src),
            [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMIN(dst, src, tmp); });
    } else {
        runTROWEXPANDOP<T, TIdx, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, tmp, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMIN(dst, src, tmp); });
    }
}

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMAX(T *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<half, TIdx, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half *)(out), (half *)(src),
            [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMAX(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<T, TIdx, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, tmp, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMAX(dst, src, tmp); });
    }
}

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMIN(T *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, TIdx, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half *)(out), (half *)(src),
            [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMIN(dst, src, tmp); });
    } else {
        runTROWEXPANDOP<T, TIdx, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, tmp, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMIN(dst, src, tmp); });
    }
}


template void LaunchTROWEXPANDDIV<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDDIV<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDMUL<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMUL<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDSUB<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDSUB<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDADD<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDADD<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDMAX<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMAX<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDMIN<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMIN<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDEXPDIF<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDEXPDIF<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWEXPANDDIV<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMUL<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDSUB<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDADD<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMAX<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDMIN<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDEXPDIF<float, 16, 16, 32, 32, 64, 64>(float *out, float *src, void *stream);
