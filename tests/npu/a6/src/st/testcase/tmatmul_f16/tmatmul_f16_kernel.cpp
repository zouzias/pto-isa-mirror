/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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

// MMAD.f16f32: A[half] x B[half] -> C[float].
// Dedicated testcase for the fp16 x fp16 -> fp32 cube matmul (no microscaling).
// Same data path as the tmatmul testcase but focused solely on the f16f32
// combo, exercising both ND and DN GM layouts with aligned/partial/GEMV-like
// shapes.
template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, typename GlobalDataSrc0,
          typename GlobalDataSrc1>
AICORE inline void RunTMATMULF16Impl(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(AType); // half -> 16
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<1, 1, 1, validM, validN>,
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

// ND (row-major) GM layout: stride last-dim = 1.
template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMULF16_ND(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    RunTMATMULF16Impl<OutType, AType, BType, validM, validK, validN, GlobalDataSrc0, GlobalDataSrc1>(out, src0, src1);
}

// DN (transposed) GM layout: stride last-dim = the *other* extent (M for A, K for B).
template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMULF16_DN(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, 1, validM>, pto::Layout::DN>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, 1, validK>, pto::Layout::DN>;
    RunTMATMULF16Impl<OutType, AType, BType, validM, validK, validN, GlobalDataSrc0, GlobalDataSrc1>(out, src0, src1);
}

template <int32_t tilingKey>
void LaunchTMATMULF16(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

#define DEFINE_TMATMUL_F16_LAUNCH_ND(KEY, M, K, N)                                                     \
    template <>                                                                                        \
    void LaunchTMATMULF16<KEY>(uint8_t * out, uint8_t * src0, uint8_t * src1, void* stream)            \
    {                                                                                                  \
        RunTMATMULF16_ND<float, half, half, M, K, N><<<1, nullptr, stream>>>(                          \
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1)); \
    }

#define DEFINE_TMATMUL_F16_LAUNCH_DN(KEY, M, K, N)                                                     \
    template <>                                                                                        \
    void LaunchTMATMULF16<KEY>(uint8_t * out, uint8_t * src0, uint8_t * src1, void* stream)            \
    {                                                                                                  \
        RunTMATMULF16_DN<float, half, half, M, K, N><<<1, nullptr, stream>>>(                          \
            reinterpret_cast<float*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1)); \
    }

// --- ND (row-major) cases ---
DEFINE_TMATMUL_F16_LAUNCH_ND(1, 16, 16, 16)      // minimal fractal
DEFINE_TMATMUL_F16_LAUNCH_ND(2, 64, 64, 64)      // aligned, 1 fractal
DEFINE_TMATMUL_F16_LAUNCH_ND(3, 128, 128, 128)   // aligned, multi-fractal
DEFINE_TMATMUL_F16_LAUNCH_ND(4, 17, 33, 31)      // partial, unaligned (just over 16)
DEFINE_TMATMUL_F16_LAUNCH_ND(5, 95, 33, 79)      // partial, mixed
DEFINE_TMATMUL_F16_LAUNCH_ND(6, 127, 96, 95)     // partial, near power-of-2
DEFINE_TMATMUL_F16_LAUNCH_ND(7, 1, 256, 64)      // GEMV-like (M=1)
DEFINE_TMATMUL_F16_LAUNCH_ND(8, 2, 80, 48)       // small M, unaligned K
DEFINE_TMATMUL_F16_LAUNCH_ND(9, 128, 128, 256)   // large K (acc stays 1 fractal)
DEFINE_TMATMUL_F16_LAUNCH_ND(10, 129, 95, 33)    // partial, M>128

// --- DN (transposed) cases ---
DEFINE_TMATMUL_F16_LAUNCH_DN(11, 31, 96, 47)     // partial
DEFINE_TMATMUL_F16_LAUNCH_DN(12, 127, 33, 95)    // partial
DEFINE_TMATMUL_F16_LAUNCH_DN(13, 64, 64, 64)     // aligned
DEFINE_TMATMUL_F16_LAUNCH_DN(14, 1, 256, 64)     // GEMV-like (M=1)
DEFINE_TMATMUL_F16_LAUNCH_DN(15, 65, 90, 89)     // partial, just over 64

#undef DEFINE_TMATMUL_F16_LAUNCH_DN
#undef DEFINE_TMATMUL_F16_LAUNCH_ND
