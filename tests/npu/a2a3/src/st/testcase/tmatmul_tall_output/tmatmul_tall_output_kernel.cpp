/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * FP32 TMATMUL whose result tile is TALLER THAN WIDE (validN < validM), cube-only.
 *
 * The existing `tmatmul` ST case never exercises this: every one of its shapes has
 * validN >= validM (31x120x58, 65x90x89, 5x75x11, 1x256x64, 1x16x32, 1x200x32, 16x32x64 and
 * the bias variants), so a result tile with fewer columns than rows was untested.
 *
 * All cases here PASS. That is the point: the cube loads both operands itself, so no
 * cross-core FIFO is involved, and TMATMUL/TSTORE handle a tall result correctly. The
 * companion `tmatmul_tall_output_mix` runs the same shapes with the operands pushed through
 * a cross-core pipe, which is where a tall result goes wrong.
 */

#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, int validM, int validK, int validN>
__global__ AICORE void RunTallStore(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        T, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;

    using GlobalDataOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, T, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, T, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<T, M, K, validM, validK>;
    using RightTile = TileRight<T, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

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

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

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
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMULTallOutput(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream)
{
    if constexpr (tilingKey == 1) {
        // TALL: 64x64 @ 64x32 -> 64x32.
        RunTallStore<float, 64, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1));
    } else if constexpr (tilingKey == 2) {
        // SQUARE control: identical code path.
        RunTallStore<float, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1));
    } else if constexpr (tilingKey == 3) {
        // TALL at a second shape, to show it is not specific to 64x32.
        RunTallStore<float, 32, 32, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(src0), reinterpret_cast<float*>(src1));
    }
}

template void LaunchTMATMULTallOutput<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMULTallOutput<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTMATMULTallOutput<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
