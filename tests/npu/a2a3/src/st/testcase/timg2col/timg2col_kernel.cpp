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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, typename U, uint32_t FMN, uint32_t FMC1, uint32_t FMH, uint32_t FMW, uint32_t FMC0,
          uint32_t FTC1, uint32_t FTH, uint32_t FTW, uint32_t FTN, uint32_t FTC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
          uint8_t padTop = 0, uint8_t padBottom = 0, uint8_t padLeft = 0, uint8_t padRight = 0>
AICORE inline void runTIMG2COL(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (FMH + padTop + padBottom - dilationH * (FTH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (FMW + padLeft + padRight - dilationW * (FTW - 1) - 1) / strideW + 1;
    constexpr uint32_t M = FMN * heightOut * widthOut;
    constexpr uint32_t N = FTN;
    constexpr uint32_t K = FTC1 * FTC0 * FTH * FTW;

    constexpr int gStrideSrc0[5] = {FMC1 * FMH * FMW * FMC0, FMH * FMW * FMC0, FMW * FMC0, FMC0, 1};
    using ShapeDim5Src0 = pto::Shape<FMN, FMC1, FMH, FMW, FMC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {FTH * FTW * FTN * FTC0, FTW * FTN * FTC0, FTN * FTC0, FTC0, 1};
    using ShapeDim5Src1 = pto::Shape<FTC1, FTH, FTW, FTN, FTC0>;
    using StridDim5Src1 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    constexpr int bufferSizeA = FMN * FMC1 * FMH * FMW * FMC0 * sizeof(U);
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<FMN, FMC1, FMH, FMW, FMC0>>;
    TileData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = FTC1 * FTH * FTW * FTN * FTC0 * sizeof(T);
    using TileData =
        ConvTile<TileType::Mat, U, bufferSize, Layout::FRACTAL_Z, pto::ConvTileShape<FTC1, FTH, FTW, FTN, FTC0>>;
    TileData bMatTile;
    static_assert(bMatTile.totalDimCount == 5);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<U, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);


    Img2colTileConfig convcfg;
    convcfg.channelSize = FMC1 * FMC0;
    convcfg.dilationH = dilationH;
    convcfg.dilationW = dilationW;
    convcfg.filterH = FTH;
    convcfg.filterW = FTW;
    convcfg.fmapH = FMH;
    convcfg.fmapW = FMW;
    convcfg.strideH = strideH;
    convcfg.strideW = strideW;
    convcfg.transpose = false;
    convcfg.padList[0] = padTop;
    convcfg.padList[1] = padBottom;
    convcfg.padList[2] = padLeft;
    convcfg.padList[3] = padRight;
    convcfg.padValue = 0;

    TSETFMATRIX<SetFmatrixMode::FMATRIX_A_MANUAL>(convcfg);
    TIMG2COL<LeftTile, TileMatAData, SetFmatrixMode::FMATRIX_A_MANUAL>(aTile, aMatTile, 0, 0, convcfg);
    TMOV(bTile, bMatTile, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}


extern "C" __global__ AICORE void launchTIMG2COl_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 1;
    constexpr uint32_t FMC1 = 2;
    constexpr uint32_t FMH = 4;
    constexpr uint32_t FMW = 16;
    constexpr uint32_t FMC0 = 16;

    constexpr uint32_t FTC1 = 2;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 16;


    runTIMG2COL<float, half, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}



template <int32_t tilingKey>
void launchTIMG2COl(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTIMG2COl_1<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTIMG2COl<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
