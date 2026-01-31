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

constexpr uint16_t BLOCK_CUBE_M_N = 16;

template <bool use5d, int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
    int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmNz2nd(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;

    using GlobalDataSrc0 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    // using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;
    using GlobalDataOut = std::conditional_t< use5d,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <bool use5d, int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
    int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmScalarNz2nd(
    __gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1, float scalarQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;

    using GlobalDataSrc0 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    // using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;
    using GlobalDataOut = std::conditional_t< use5d,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    uint64_t preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalarQuant));
    if (sizeof(dstDataType) == 1) {
        constexpr bool sign = (std::is_same_v<dstDataType, int8_t>) ? true : false;
        preQuantScalar = (preQuantScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;

    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile, preQuantScalar);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, preQuantScalar);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <bool use5d, int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
    int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmVectorNz2nd(
    __gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1, __gm__ uint64_t *quantTensor)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;
    constexpr int alignScalingN = ((validN * sizeof(uint64_t) + 127) / 128) * 128 / sizeof(uint64_t);

    using GlobalDataSrc0 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, alignScalingN>,
        pto::Stride<alignScalingN, alignScalingN, alignScalingN, alignScalingN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    // using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;
    using GlobalDataOut = std::conditional_t< use5d,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(quantTensor);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<TileType::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);
    uint32_t bMatSize = K * N * sizeof(srcDataType);
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(validN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);
    TASSIGN(scalingMatTile, aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    ScalingTile scalingTile(validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(scalingMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(scalingTile, scalingMatTile);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum>(dstGlobal, cTile, scalingTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, scalingTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmNz2nd<false, 1, float, float, float, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, 16>
            <<<1, nullptr, stream>>>(
                reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmNz2nd<false, 0, float, float, float, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmNz2nd<false, 0, float, float, float, 1, 1, 1, 117, 97, 1, 1, 1, 117, 97, 117, 97, 71, 1>
            <<<1, nullptr, stream>>>(
                reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmScalarNz2nd<false, 0, int32_t, half, int8_t, 1, 1, 1, 64, 64, 1, 1, 1, 64, 64, 64, 64, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmScalarNz2nd<false, 0, int32_t, int8_t, int8_t, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 26>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmScalarNz2nd<false, 0, float, int8_t, half, 1, 1, 1, 77, 34, 1, 1, 1, 77, 34, 77, 34, 81, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1), scalarQuant);
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmVectorNz2nd<false, 0, int32_t, half, int8_t, 1, 1, 1, 55, 88, 1, 1, 1, 55, 88, 55, 88, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmVectorNz2nd<false, 0, int32_t, int8_t, int8_t, 1, 1, 1, 34, 85, 1, 1, 1, 34, 85, 34, 85, 19>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmVectorNz2nd<false, 0, float, int8_t, half, 1, 1, 1, 85, 77, 1, 1, 1, 85, 77, 85, 77, 66, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    }
}

template void LaunchTStoreAcc2gmNz2nd<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gmScalarNz2nd<1>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<2>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);


template void LaunchTStoreAcc2gmVectorNz2nd<1>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<2>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);


template void LaunchTStoreAcc2gmNz2nd<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmScalarNz2nd<21>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);
template void LaunchTStoreAcc2gmVectorNz2nd<21>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);