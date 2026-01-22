/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint16_t BIAS_ALIGN = 64;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}


template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL_GEMV_CLOSE(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 = GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    // clear l1 buffer which exceed the valid shape
    TFILLPAD(aMatTile, aMatTile);

    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 = GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMULSplitK(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int BASEK = 32;

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, BASEK);

    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, BASEK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, BASEK, validN>,
        pto::Stride<1 * BASEK * validN, 1 * BASEK * validN, BASEK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<validN, validN, validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, U, M, BASEK, BLayout::ColMajor, validM, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    using LeftTile = TileLeft<U, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<S, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + validN * i * BASEK);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL_GEMV_OPEN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 = GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatADataGemv =
        Tile<TileType::Mat, U, 1, K, BLayout::ColMajor, 1, validK, SLayout::RowMajor, 32>; // [1, K] gemv layout
    
    // constexpr int reshpeM = CeilDivision(validK, 16);
    // assert  rows  must be divisible by inner box rows   512 -> inner row = 16
    // constexpr int reshpeMAlign = CeilAlign<int>(reshpeM, 16);
    // using TileMatAData = Tile<TileType::Mat, U, reshpeMAlign, 16, BLayout::ColMajor, reshpeM, 16, SLayout::RowMajor, 512>;

    // 需要申请512B对齐的静态shape (validK * sizeof(T) + 511)/512 * 512  / sizeof(T)
    // 不512B对齐好像也没关系？
    // textract仅仅支持src为NZ格式的，所以aMatTile需要是NZ的 , 静态shape对齐到32B就可以

    using TileMatAData =Tile<TileType::Mat, U, 1, K, BLayout::ColMajor, 1, validK, SLayout::RowMajor, 32>;  // innerRow = 1, innerCol = 16
    // mx的 innerRow = 16, innerCol = 2，所以会被assert
    // using TileMatAData = Tile<TileType::Mat, U, 1, K, BLayout::ColMajor, 1, validK, SLayout::RowMajor, 32>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, validN>;

    // TileLeftVector语法糖，用户输入1*k 在语法糖内部转成 [K/16, 16]
    // L0A上如果是K*1呢？  ...
    using LeftTile = TileLeftVector<U, 1, K, 1, validK>;

    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatADataGemv aMatTileGemv;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(biasDataTile, 0x40000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /******************************TLOAD*****************************/
    
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    // TRESHAPE assert : 原始的和后面的total size必须一致
    // TRESHAPE(aMatTile, aMatTileGemv);  // reshape [1, K] -> [K/16, 16] as aTile is gemv layout before TEXTRACT
    //  1 * 16   -> rows/innerrows == 0 
    
    TEXTRACT(aTile, aMatTile, 0, 0);
    TMOV(bTile, bMatTile);

    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    if constexpr (isBias) {
        TGEMV_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        // TMATMUL(cTile, aTile, bTile);
        TGEMV(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 31, 120, 58, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 65, 90, 89, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMULSplitK<float, half, half, float, 5, 75, 11, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL_GEMV_CLOSE<float, half, half, float, 1, 256, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL_GEMV_OPEN<float, half, half, float, 1, 16, 32, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL_GEMV_OPEN<float, half, half, float, 1, 512, 32, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    }
}

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 26, 100, 94, true><<<1, nullptr, stream>>>(reinterpret_cast<float *>(out),
            reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<float, half, half, float, 101, 288, 67, true><<<1, nullptr, stream>>>(reinterpret_cast<float *>(out),
            reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<float, float, float, float, 15, 16, 15, true><<<1, nullptr, stream>>>(reinterpret_cast<float *>(out),
            reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 55, 127, 29, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 11, 402, 30, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                reinterpret_cast<bfloat16_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 150, 89, 50, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 7) {
        RunTMATMULSplitK<int32_t, int8_t, int8_t, int32_t, 135, 64, 88, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);