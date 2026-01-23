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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, typename U, uint32_t FMN, uint32_t FMC1, uint32_t FMH, uint32_t FMW, uint32_t FMC0,
          uint32_t FTC1, uint32_t FTH, uint32_t FTW, uint32_t FTN, uint32_t FTC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
          uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COL(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (FMH + padTop + padBottom - dilationH * (FTH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (FMW + padLeft + padRight - dilationW * (FTW - 1) - 1) / strideW + 1;
    constexpr uint32_t M = FMN * heightOut * widthOut;
    constexpr uint32_t SingleCoreM = heightOut * widthOut;
    constexpr uint32_t N = FTN;
    constexpr uint32_t K = FTC1 * FTC0 * FTH * FTW;

    constexpr int gStrideSrc0[5] = {FMC1 * FMH * FMW * FMC0, FMH * FMW * FMC0, FMW * FMC0, FMC0, 1};
    using ShapeDim5Src0 = pto::Shape<FMN, FMC1, FMH, FMW, FMC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {FTH * FTW * FTN * FTC0, FTW * FTN * FTC0, FTN * FTC0, FTC0, 1};
    using ShapeDim5Src1 = pto::Shape<FTC1, FTH, FTW, FTN, FTC0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc0 src0Global(src0);
    GlobalDataOut dstGlobal(out);

    constexpr int bufferSizeA = FMN * FMC1 * FMH * FMW * FMC0 * sizeof(U);
    using TileMatADataSplit =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<1, FMC1, FMH, FMW, FMC0>>;
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<FMN, FMC1, FMH, FMW, FMC0>>;
    TileMatADataSplit a1MatTile;
    TileMatAData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(a1MatTile, 0x0);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = FTC1 * FTH * FTW * FTN * FTC0 * sizeof(T);
    using TileMatBData =
        ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z, pto::ConvTileShape<FTC1, FTH, FTW, FTN, FTC0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 5);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, M, K, M, K>;
    using LeftTileSingleCore = TileLeft<U, SingleCoreM, K, SingleCoreM, K>;
    using RightTile = TileRight<U, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;
    LeftTile aTile;
    LeftTileSingleCore a1Tile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(a1Tile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    Img2colTileConfig convcfg;
    convcfg.channelSize = FMC1 * FMC0;
    convcfg.dilationW = dilationW;
    convcfg.dilationH = dilationH;
    convcfg.filterH = FTH;
    convcfg.filterW = FTW;
    convcfg.fmapH = FMH;
    convcfg.fmapW = FMW;
    convcfg.strideW = strideW;
    convcfg.strideH = strideH;
    convcfg.padList[0] = padTop;
    convcfg.padList[1] = padBottom;
    convcfg.padList[2] = padLeft;
    convcfg.padList[3] = padRight;
    convcfg.padValue = 0;
    convcfg.transpose = false;

    TSETFMATRIX<SetFmatrixMode::FMATRIX_A_MANUAL>(convcfg);
    for (int i = 0; i < FMN; i++) {
        TIMG2COL<LeftTileSingleCore, TileMatADataSplit>(a1Tile, a1MatTile, 0, 0, convcfg);
        TASSIGN(a1MatTile, (i + 1) * FMC1 * FMH * FMW * FMC0 * sizeof(U));
        TASSIGN(a1Tile, (i + 1) * SingleCoreM * K * sizeof(U));
    }
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}
template <typename T, typename U, uint32_t FMN, uint32_t FMC1, uint32_t FMH, uint32_t FMW, uint32_t FMC0,
          uint32_t FTC1, uint32_t FTH, uint32_t FTW, uint32_t FTN, uint32_t FTC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
          uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COLSplit(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (FMH + padTop + padBottom - dilationH * (FTH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (FMW + padLeft + padRight - dilationW * (FTW - 1) - 1) / strideW + 1;

    constexpr uint32_t validM = FMN * heightOut * widthOut;
    constexpr uint32_t validSingleCoreM = heightOut * widthOut;
    constexpr uint32_t validN = FTN;
    constexpr uint32_t validK = FTC1 * FTC0 * FTH * FTW;

    constexpr int baseK = 64;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<uint32_t>(validM, 16);
    constexpr int SingleCoreM = CeilAlign<uint32_t>(validSingleCoreM, 16);
    constexpr int N = CeilAlign<uint32_t>(validN, blockAlign);
    constexpr int K = CeilAlign<uint32_t>(validK, baseK);

    constexpr int gStrideSrc0[5] = {FMC1 * FMH * FMW * FMC0, FMH * FMW * FMC0, FMW * FMC0, FMC0, 1};
    using ShapeDim5Src0 = pto::Shape<FMN, FMC1, FMH, FMW, FMC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {FTH * FTW * FTN * FTC0, FTW * FTN * FTC0, FTN * FTC0, FTC0, 1};
    using ShapeDim5Src1 = pto::Shape<FTC1, FTH, FTW, FTN, FTC0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                            pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    constexpr int bufferSizeA = FMN * FMC1 * FMH * FMW * FMC0 * sizeof(U);
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<FMN, FMC1, FMH, FMW, FMC0>>;
    using TileMatADataSplit =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0, pto::ConvTileShape<1, FMC1, FMH, FMW, FMC0>>;
    TileMatAData aMatTile;
    TileMatADataSplit a1MatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(a1MatTile, 0x0);

    constexpr int bufferSizeB = FTC1 * FTH * FTW * FTN * FTC0 * sizeof(T);
    using TileMatBData =
        ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z, pto::ConvTileShape<FTC1, FTH, FTW, FTN, FTC0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 5);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, M, baseK, validM, baseK>;
    using LeftTileSingleCore = TileLeft<U, SingleCoreM, baseK, validSingleCoreM, baseK>;
    using RightTile = TileRight<U, baseK, N, baseK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using AccTileSplit = TileAcc<T, SingleCoreM, N, validSingleCoreM, validN>;
    LeftTile aTile;
    LeftTileSingleCore a1Tile;
    RightTile bTile;
    AccTile cTile;
    AccTileSplit c1Tile;
    TASSIGN(aTile, 0x0);
    TASSIGN(a1Tile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(c1Tile, 0x0);

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
    constexpr int iter = K / baseK;
    for (int i = 0; i < FMN; i++) {
        for (int i = 0; i < iter; i++) {
            TIMG2COL<LeftTileSingleCore, TileMatADataSplit>(a1Tile, a1MatTile, 0, i * baseK, convcfg);
            TEXTRACT(bTile, bMatTile, i * baseK, 0);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (i == 0) {
                TMATMUL(c1Tile, a1Tile, bTile);
            } else {
                TMATMUL_ACC(c1Tile, c1Tile, a1Tile, bTile);
            }
            pipe_barrier(PIPE_ALL);
        }
        TASSIGN(a1MatTile, (i + 1) * FMC1 * FMH * FMW * FMC0 * sizeof(U));
        TASSIGN(a1Tile, 0x00);
        TASSIGN(c1Tile, (i + 1) * SingleCoreM * N * sizeof(T));
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTIMG2COL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
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

    runTIMG2COL<float, bfloat16_t, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 1;
    constexpr uint32_t FMC1 = 4;
    constexpr uint32_t FMH = 4;
    constexpr uint32_t FMW = 16;
    constexpr uint32_t FMC0 = 16;

    constexpr uint32_t FTC1 = 4;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 16;

    runTIMG2COL<float, half, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0, 2, 1>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 1;
    constexpr uint32_t FMC1 = 4;
    constexpr uint32_t FMH = 8;
    constexpr uint32_t FMW = 16;
    constexpr uint32_t FMC0 = 8;

    constexpr uint32_t FTC1 = 4;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 8;

    runTIMG2COL<float, float, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0, 1, 1, 2, 2>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 1;
    constexpr uint32_t FMC1 = 1;
    constexpr uint32_t FMH = 8;
    constexpr uint32_t FMW = 16;
    constexpr uint32_t FMC0 = 32;

    constexpr uint32_t FTC1 = 1;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 32;


    runTIMG2COL<int32_t, int8_t, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 2;
    constexpr uint32_t FMC1 = 4;
    constexpr uint32_t FMH = 16;
    constexpr uint32_t FMW = 64;
    constexpr uint32_t FMC0 = 16;

    constexpr uint32_t FTC1 = 4;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 16;

    runTIMG2COLSplit<float, bfloat16_t, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0, 2, 2, 2, 2, 1, 2, 1, 2>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 4;
    constexpr uint32_t FMC1 = 4;
    constexpr uint32_t FMH = 8;
    constexpr uint32_t FMW = 16;
    constexpr uint32_t FMC0 = 16;

    constexpr uint32_t FTC1 = 4;
    constexpr uint32_t FTH = 3;
    constexpr uint32_t FTW = 3;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 16;

    runTIMG2COLSplit<float, half, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 2;
    constexpr uint32_t FMC1 = 2;
    constexpr uint32_t FMH = 16;
    constexpr uint32_t FMW = 32;
    constexpr uint32_t FMC0 = 8;

    constexpr uint32_t FTC1 = 2;
    constexpr uint32_t FTH = 4;
    constexpr uint32_t FTW = 4;
    constexpr uint32_t FTN = 16;
    constexpr uint32_t FTC0 = 8;

    runTIMG2COLSplit<float, float, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0, 1, 1, 2, 2>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t FMN = 1;
    constexpr uint32_t FMC1 = 2;
    constexpr uint32_t FMH = 32;
    constexpr uint32_t FMW = 64;
    constexpr uint32_t FMC0 = 32;

    constexpr uint32_t FTC1 = 2;
    constexpr uint32_t FTH = 2;
    constexpr uint32_t FTW = 2;
    constexpr uint32_t FTN = 64;
    constexpr uint32_t FTC0 = 32;

    runTIMG2COLSplit<int32_t, int8_t, FMN, FMC1, FMH, FMW, FMC0, FTC1, FTH, FTW, FTN, FTC0, 2, 2, 2, 2, 1, 1, 1, 0>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

template <int32_t tilingKey>
void launchTIMG2COL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTIMG2COL_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTIMG2COL_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTIMG2COL_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTIMG2COL_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTIMG2COL_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTIMG2COL_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTIMG2COL_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTIMG2COL_8<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTIMG2COL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
