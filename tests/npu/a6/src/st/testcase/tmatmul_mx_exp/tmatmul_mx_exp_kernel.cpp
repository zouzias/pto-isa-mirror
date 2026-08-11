/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MMAD_MX experiment: isolate which aspect of E4M3E2M1 the dav-920r1 sim rejects.
//
// Known results:
//   tmatmul_mx_e1m2  (fp4_e1m2x2 A + fp4_e1m2x2 B, MX_A_ZZ/MX_B_NN e8m0) -> PASS
//   tmatmul_mx_hif4  (hif4 A + hif4 B, HIF4_A_ZZ/HIF4_B_NN)             -> PASS
//   tmatmul_mx_e4m3e2m1 (fp8_e4m3 A + fp4_e2m1 B, MX_A_ZZ/MX_B_NN e8m0) -> FAIL all-NaN
//                        (cube_invld_input, XM=XN=XD=0; persists with neutral A-scale)
//
// This test runs 4 cases at 128x128x128 ND to localize the failure:
//   E1: fp8_e4m3   x fp8_e4m3   — does fp8 A work in ANY MX context (pure fp8)?
//   E2: fp4_e2m1   x fp8_e4m3   — does fp8 on the B side work? (swap of failing case)
//   E3: fp8_e4m3   x fp4_e2m1   — failing case with BOTH scales neutral (all 127)
//   E4: fp4_e2m1   x fp4_e2m1   — pure e2m1 baseline (should pass; sanity)
//
// Interpretation guide:
//   E4 PASS + E1 FAIL          -> fp8 A is broken in MX on this sim (regardless of B)
//   E4 PASS + E1 PASS + E2 PASS-> the fp8-A + fp4-B MIXING specifically breaks
//   E4 PASS + E2 FAIL          -> fp8 on B side also breaks (fp8 anywhere in MX)
//   E3 still NaN               -> structural (not scale values on either side)

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t MX_SCALE_GROUP = 32;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

// Generic MX matmul kernel: AType x BType -> float, with e8m0 per-32-K scales
// on both sides (MX_A_ZZ for A, MX_B_NN for B). Same data path as
// tmatmul_mx_e4m3e2m1 / tmatmul_mx_e1m2, just parameterized over both dtypes.
template <typename AType, typename BType, int validM, int validK, int validN>
AICORE inline void RunMxExpImpl(__gm__ float* out, __gm__ AType* aData, __gm__ uint8_t* aScale,
                               __gm__ BType* bData, __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / MX_SCALE_GROUP;

    using TileMatA = Tile<
        TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleK, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleK, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::MX_B_NN>;
    using GlobalDataOut = GlobalTensor<
        float, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);
    GlobalDataOut outGm(out);

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleK, validM, scaleK>;
    using RightScaleTile = TileRightScale<uint8_t, scaleK, N, scaleK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0;
    RightTile bl0;
    LeftScaleTile aScaleL0;
    RightScaleTile bScaleL0;
    AccTile cTile;
    TASSIGN(al0, 0x0u);
    TASSIGN(bl0, 0x0u);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);
    TEXTRACT(bl0, bMatTile, 0, 0);
    TEXTRACT(bScaleL0, bScaleTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(outGm, cTile);
}

// E5 variant: same as RunMxExpImpl but SKIPS TSTORE entirely (fixpipe removed).
// Purpose: verify whether cube_invld_input still fires on MMAD_MX when no
// fixpipe (FIX_L0C_TO_DST) instruction follows. If it still fires, the cube
// itself rejects the E4M3E2M1 instruction — independent of fixpipe.
template <typename AType, typename BType, int validM, int validK, int validN>
AICORE inline void RunMxExpNoStoreImpl(__gm__ float* out, __gm__ AType* aData, __gm__ uint8_t* aScale,
                                      __gm__ BType* bData, __gm__ uint8_t* bScale)
{
    (void)out; // no store — output buffer unused
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / MX_SCALE_GROUP;

    using TileMatA = Tile<
        TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleK, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleK, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::MX_B_NN>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleK, validM, scaleK>;
    using RightScaleTile = TileRightScale<uint8_t, scaleK, N, scaleK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0;
    RightTile bl0;
    LeftScaleTile aScaleL0;
    RightScaleTile bScaleL0;
    AccTile cTile;
    TASSIGN(al0, 0x0u);
    TASSIGN(bl0, 0x0u);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);
    TEXTRACT(bl0, bMatTile, 0, 0);
    TEXTRACT(bScaleL0, bScaleTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    // MMAD_MX only — NO TSTORE, NO fixpipe. Inspect the sim log for
    // cube_invld_input: if it fires here, the cube rejects E4M3E2M1 upstream
    // of fixpipe.
    TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);
}

template <typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunMxExp(__gm__ float* out, __gm__ AType* aData, __gm__ uint8_t* aScale,
                                __gm__ BType* bData, __gm__ uint8_t* bScale)
{
    RunMxExpImpl<AType, BType, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

// E5: MMAD_MX only, no TSTORE (fixpipe removed). Same args signature so the
// host Launch path is uniform; the output buffer is allocated but unused.
template <typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunMxExpNoStore(__gm__ float* out, __gm__ AType* aData, __gm__ uint8_t* aScale,
                                       __gm__ BType* bData, __gm__ uint8_t* bScale)
{
    RunMxExpNoStoreImpl<AType, BType, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

namespace TmatmulMxExp {

template <int caseId>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);

#define DEFINE_EXP_LAUNCH(ID, A_T, B_T, M, K, N)                                                         \
    template <>                                                                                          \
    void Launch<ID>(uint8_t * out, uint8_t * aData, uint8_t * aScale, uint8_t * bData, uint8_t * bScale, \
                    void* stream)                                                                        \
    {                                                                                                    \
        RunMxExp<A_T, B_T, M, K, N><<<1, nullptr, stream>>>(                                            \
            reinterpret_cast<float*>(out), reinterpret_cast<A_T*>(aData), reinterpret_cast<uint8_t*>(aScale), \
            reinterpret_cast<B_T*>(bData), reinterpret_cast<uint8_t*>(bScale));                          \
    }

// E1: fp8_e4m3 x fp4_e2m1, varied scales (failing baseline — reproduces NaN)
DEFINE_EXP_LAUNCH(1, float8_e4m3_t, float4_e2m1x2_t, 128, 128, 128)
// E2: fp8_e4m3 x fp4_e2m1, neutral A-scale (isolates A-scale values)
DEFINE_EXP_LAUNCH(2, float8_e4m3_t, float4_e2m1x2_t, 128, 128, 128)
// E3: fp8_e4m3 x fp4_e2m1, neutral B-scale (isolates B-scale values)
DEFINE_EXP_LAUNCH(3, float8_e4m3_t, float4_e2m1x2_t, 128, 128, 128)
// E4: fp4_e1m2x2 x fp4_e1m2x2, varied scales (passing baseline — harness sanity)
DEFINE_EXP_LAUNCH(4, float4_e1m2x2_t, float4_e1m2x2_t, 128, 128, 128)

// E5: fp8_e4m3 x fp4_e2m1, NO TSTORE (fixpipe removed) — check cube_invld_input
template <>
void Launch<5>(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream)
{
    RunMxExpNoStore<float8_e4m3_t, float4_e2m1x2_t, 128, 128, 128><<<1, nullptr, stream>>>(
        reinterpret_cast<float*>(out), reinterpret_cast<float8_e4m3_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
        reinterpret_cast<float4_e2m1x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
}

#undef DEFINE_EXP_LAUNCH
} // namespace TmatmulMxExp
