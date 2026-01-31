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

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}
template <typename T, typename U, uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0,
          uint32_t filterDim3, uint32_t filterDim2, uint32_t filterDim1, uint32_t filterDim0,
          uint32_t filterH, uint32_t filterW, uint32_t outC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
          uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COLOrigin(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;

    constexpr uint32_t validM = fmapN * heightOut * widthOut;
    constexpr uint32_t validN = filterDim2 * filterDim1;
    constexpr uint32_t validK = filterDim3 * filterDim0;

    constexpr int baseK = 64;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<uint32_t>(validM, 16);
    constexpr int N = CeilAlign<uint32_t>(validN, blockAlign);
    constexpr int K = CeilAlign<uint32_t>(validK, baseK);

    constexpr int gStrideSrc0[5] = {fmapC1 * fmapH * fmapW * fmapC0, fmapH * fmapW * fmapC0, fmapW * fmapC0, fmapC0, 1};
    using ShapeDim5Src0 = pto::Shape<fmapN, fmapC1, fmapH, fmapW, fmapC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {filterDim3 * filterDim2 * filterDim1 * filterDim0,
                                    filterDim2 * filterDim1 * filterDim0, filterDim1 * filterDim0, filterDim0, 1};
    using ShapeDim5Src1 = pto::Shape<1, filterDim3, filterDim2, filterDim1, filterDim0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    constexpr int bufferSizeA = sizeof(U) * fmapN * fmapC1 * fmapH * fmapW * fmapC0;
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0,
                 pto::ConvTileShape<fmapN, fmapC1, fmapH, fmapW, fmapC0>>;
    TileMatAData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = filterDim3 * filterDim2 * filterDim1 * filterDim0 * sizeof(T);
    using TileMatBData =
        ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                 pto::ConvTileShape<filterDim3, filterDim2, filterDim1, filterDim0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 4);
    TASSIGN(bMatTile, 0x40000);

    using LeftTile = TileLeft<U, 2 * M, 2 * baseK, validM, baseK>;    //test compact mode works properly
    using RightTile = TileRight<U, 2 * baseK, 2 * N, baseK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
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

    Img2colTileConfig<U> convcfg;
    convcfg.channelSize = fmapC1 * fmapC0;
    convcfg.dilationH = dilationH;
    convcfg.dilationW = dilationW;
    convcfg.filterH = filterH;
    convcfg.filterW = filterW;
    convcfg.fmapH = fmapH;
    convcfg.fmapW = fmapW;
    convcfg.strideH = strideH;
    convcfg.strideW = strideW;
    convcfg.transpose = false;
    convcfg.padList[0] = padLeft;
    convcfg.padList[1] = padRight;
    convcfg.padList[2] = padTop;
    convcfg.padList[3] = padBottom;
    convcfg.padValue = 0;

    TSETFMATRIX<SetFmatrixMode::FMATRIX_A_MANUAL, U>(convcfg);
    constexpr int iter = K / baseK;
    for (int i = 0; i < iter; i++) {
        TIMG2COL<LeftTile, TileMatAData, SetFmatrixMode::FMATRIX_A_MANUAL, U>(aTile, aMatTile, 0, i * baseK, convcfg);
        TEXTRACT(bTile, bMatTile, i * baseK, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
        pipe_barrier(PIPE_ALL);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    if constexpr (std::is_same_v<T, int32_t>) {
        using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                                pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    } else {
        using GlobalDataOut = GlobalTensor<T, pto::Shape<1, filterDim2 * filterDim1 / outC0, heightOut, widthOut, outC0>,
            pto::Stride<filterDim2 * filterDim1 * heightOut * widthOut, heightOut * widthOut * outC0, widthOut * outC0, outC0, 1>,
            Layout::NC1HWC0>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    }
}
template <typename T, typename U, uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0,
          uint32_t filterDim3, uint32_t filterDim2, uint32_t filterDim1, uint32_t filterDim0,
          uint32_t filterH, uint32_t filterW, uint32_t outC0,
          uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
          uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1>
AICORE inline void runTIMG2COLFractalZ4D(__gm__ T *out, __gm__ U *src0, __gm__ U *src1)
{
    constexpr uint32_t heightOut = (fmapH + padTop + padBottom - dilationH * (filterH - 1) - 1) / strideH + 1;
    constexpr uint32_t widthOut = (fmapW + padLeft + padRight - dilationW * (filterW - 1) - 1) / strideW + 1;

    constexpr uint32_t validM = fmapN * heightOut * widthOut;
    constexpr uint32_t validN = filterDim2 * filterDim1;
    constexpr uint32_t validK = filterDim3 * filterDim0;

    constexpr uint32_t baseK = 64;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseN = 256;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<uint32_t>(validM, baseM);
    constexpr int N = CeilAlign<uint32_t>(validN, blockAlign);
    constexpr int K = CeilAlign<uint32_t>(validK, baseK);

    constexpr int gStrideSrc0[5] = {fmapC1 * fmapH * fmapW * fmapC0, fmapH * fmapW * fmapC0, fmapW * fmapC0, fmapC0, 1};
    using ShapeDim5Src0 = pto::Shape<fmapN, fmapC1, fmapH, fmapW, fmapC0>;
    using StridDim5Src0 = pto::Stride<gStrideSrc0[0], gStrideSrc0[1], gStrideSrc0[2], gStrideSrc0[3], gStrideSrc0[4]>;
    using GlobalDataSrc0 = GlobalTensor<U, ShapeDim5Src0, StridDim5Src0, Layout::NC1HWC0>;

    constexpr int gStrideSrc1[5] = {filterDim3 * filterDim2 * filterDim1 * filterDim0,
                                    filterDim2 * filterDim1 * filterDim0, filterDim1 * filterDim0, filterDim0, 1};
    using ShapeDim5Src1 = pto::Shape<1, filterDim3, filterDim2, filterDim1, filterDim0>;
    using StridDim5Src1 = pto::Stride<gStrideSrc1[0], gStrideSrc1[1], gStrideSrc1[2], gStrideSrc1[3], gStrideSrc1[4]>;
    using GlobalDataSrc1 = GlobalTensor<U, ShapeDim5Src1, StridDim5Src1, Layout::FRACTAL_Z>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                            pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataOut dstGlobal(out);
    constexpr int bufferSizeA = sizeof(U) * fmapN * fmapC1 * fmapH * fmapW * fmapC0;
    using TileMatAData =
        ConvTile<TileType::Mat, U, bufferSizeA, Layout::NC1HWC0,
                 pto::ConvTileShape<fmapN, fmapC1, fmapH, fmapW, fmapC0>>;
    TileMatAData aMatTile;
    static_assert(aMatTile.totalDimCount == 5);
    TASSIGN(aMatTile, 0x0);

    constexpr int bufferSizeB = filterDim3 * filterDim2 * filterDim1 * filterDim0 * sizeof(T);
    using TileMatBData =
        ConvTile<TileType::Mat, U, bufferSizeB, Layout::FRACTAL_Z,
                 pto::ConvTileShape<filterDim3, filterDim2, filterDim1, filterDim0>>;
    TileMatBData bMatTile;
    static_assert(bMatTile.totalDimCount == 4);
    TASSIGN(bMatTile, 0x50000);

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;    //test compact mode works properly
    using RightTile = TileRight<U, baseK, N, baseK, validN>;
    using AccTile = TileAcc<T, baseM, N, baseM, validN>;
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

    Img2colTileConfig<U> convcfg;
    convcfg.channelSize = fmapC1 * fmapC0;
    convcfg.dilationH = dilationH;
    convcfg.dilationW = dilationW;
    convcfg.filterH = filterH;
    convcfg.filterW = filterW;
    convcfg.fmapH = fmapH;
    convcfg.fmapW = fmapW;
    convcfg.strideH = strideH;
    convcfg.strideW = strideW;
    convcfg.transpose = false;
    convcfg.padList[0] = padLeft;
    convcfg.padList[1] = padRight;
    convcfg.padList[2] = padTop;
    convcfg.padList[3] = padBottom;
    convcfg.padValue = 0;

    TSETFMATRIX<SetFmatrixMode::FMATRIX_A_MANUAL, U>(convcfg);
    constexpr int iter = K / baseK;
    constexpr int Miter = M / baseM;
    for (int j = 0; j < Miter; j++) {
        for (int i = 0; i < iter; i++) {
            TIMG2COL<LeftTile, TileMatAData, SetFmatrixMode::FMATRIX_A_MANUAL, U>(aTile, aMatTile, j * baseM, i * baseK, convcfg);
            TEXTRACT(bTile, bMatTile, i * baseK, 0);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (i == 0) {
                TMATMUL(cTile, aTile, bTile);
            } else {
                TMATMUL_ACC(cTile, cTile, aTile, bTile);
            }
            pipe_barrier(PIPE_ALL);
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        GlobalDataOut dstGlobal(out + j * baseM * N);
        TSTORE(dstGlobal, cTile);
        pipe_barrier(PIPE_ALL);
    }
    out = dstGlobal.data();
}
extern "C" __global__ AICORE void launchTIMG2COL_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 33;
    constexpr uint32_t fmapW = 63;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterDim3 = 36;
    constexpr uint32_t filterDim2 = 4;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 16;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;

    constexpr uint32_t outC0 = 16;

    runTIMG2COLFractalZ4D<float, bfloat16_t, fmapN, fmapC1, fmapH, fmapW, fmapC0,
                      filterDim3, filterDim2, filterDim1, filterDim0, filterH, filterW, outC0, 2, 2, 2, 2, 0, 0, 0, 0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTIMG2COL_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 4;
    constexpr uint32_t fmapH = 36;
    constexpr uint32_t fmapW = 32;
    constexpr uint32_t fmapC0 = 16;

    constexpr uint32_t filterDim3 = 36;
    constexpr uint32_t filterDim2 = 4;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 16;
    constexpr uint32_t filterH = 3;
    constexpr uint32_t filterW = 3;

    constexpr uint32_t outC0 = 16;

    runTIMG2COLFractalZ4D<float, half, fmapN, fmapC1, fmapH, fmapW, fmapC0,
                      filterDim3, filterDim2, filterDim1, filterDim0, filterH, filterW, outC0, 2, 2, 2, 2, 1, 2, 2, 1>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTIMG2COL_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t fmapN = 1;
    constexpr uint32_t fmapC1 = 2;
    constexpr uint32_t fmapH = 36;
    constexpr uint32_t fmapW = 38;
    constexpr uint32_t fmapC0 = 8;

    constexpr uint32_t filterDim3 = 32;
    constexpr uint32_t filterDim2 = 2;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 8;
    constexpr uint32_t filterH = 4;
    constexpr uint32_t filterW = 4;

    constexpr uint32_t outC0 = 8;

    runTIMG2COLFractalZ4D<float, float, fmapN, fmapC1, fmapH, fmapW, fmapC0,
                      filterDim3, filterDim2, filterDim1, filterDim0, filterH, filterW, outC0, 1, 1, 2, 2, 1, 2, 3, 0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

template <int32_t tilingKey>
void launchTIMG2COL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 9) {
        launchTIMG2COL_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTIMG2COL_10<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTIMG2COL_11<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTIMG2COL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTIMG2COL<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
