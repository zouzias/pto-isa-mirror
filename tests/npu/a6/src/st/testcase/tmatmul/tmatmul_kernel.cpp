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

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMUL_DN(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(AType);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, 1, validM>,
                     pto::Layout::DN>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, 1, validK>,
                     pto::Layout::DN>;
    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(aTile, aMatTile, 0, 0);
    TEXTRACT(bTile, bMatTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMUL_ND(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(AType);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                     pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(aTile, aMatTile, 0, 0);
    TEXTRACT(bTile, bMatTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <>
void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, half, 31, 96, 47><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<int32_t, int8_t, int8_t, 65, 90, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<15>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int4b_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int4b_t *>(src1));
}
template <>
void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, float, float, 16, 32, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
}

template <>
void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, half, 1, 256, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, half, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int8_t, 96, 128, 65><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, float, float, 33, 63, 31><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
}

template <>
void LaunchTMATMUL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, half, 2, 80, 48><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, half, 127, 33, 95><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<int32_t, int8_t, int8_t, 17, 33, 31><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, float, float, 63, 31, 15><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
}

template <>
void LaunchTMATMUL<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, half, 95, 33, 79><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int8_t, 129, 95, 33><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, float, float, 47, 29, 25><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
}

template <>
void LaunchTMATMUL<16>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int4b_t, 96, 128, 65><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<17>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int4b_t, 129, 95, 33><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<18>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int4b_t, 17, 33, 31><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<19>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<int32_t, int8_t, int4b_t, 2, 80, 48><<<1, nullptr, stream>>>(
        reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<20>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int8_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int8_t, 96, 128, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int8_t, 129, 95, 63><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, int8_t, 65, 90, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<24>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int8_t, 2, 90, 31><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<25>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, half, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<26>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, half, 95, 33, 79><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<27>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, half, 127, 33, 95><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
}

template <>
void LaunchTMATMUL<28>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, float8_e4m3_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<float8_e4m3_t *>(src1));
}

template <>
void LaunchTMATMUL<29>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, float8_e4m3_t, 127, 64, 95><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<float8_e4m3_t *>(src1));
}

template <>
void LaunchTMATMUL<30>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, bfloat16_t, float8_e4m3_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<float8_e4m3_t *>(src1));
}

template <>
void LaunchTMATMUL<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, bfloat16_t, float8_e4m3_t, 127, 64, 95><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<float8_e4m3_t *>(src1));
}

template <>
void LaunchTMATMUL<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, bfloat16_t, int8_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, bfloat16_t, int8_t, 65, 90, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0), reinterpret_cast<int8_t *>(src1));
}

template <>
void LaunchTMATMUL<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int4b_t, 64, 64, 64><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<35>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int4b_t, 65, 90, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<36>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int4b_t, 96, 128, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<37>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_ND<float, half, int4b_t, 129, 95, 63><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<38>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, int4b_t, 65, 90, 89><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}

template <>
void LaunchTMATMUL<39>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    RunTMATMUL_DN<float, half, int4b_t, 127, 95, 63><<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<int4b_t *>(src1));
}
