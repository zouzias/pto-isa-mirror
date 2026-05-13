/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * moe_kernel.cpp - kernel-side scaffold for mani_moe.
 *
 * Current simple contract, matching scripts/gen_data.py and main.cpp:
 *   X        [kT, kH]       fp16
 *   WRouter  [kH, kE]       fp16
 *   W1       [kE, kH, kF]   fp16
 *   W2       [kE, kF, kH]   fp16
 *
 *   logits   [kT, kE]       fp32
 *   expertId [kT]           uint32
 *   Z        [kT, kH]       fp32
 *
 * Router GEMM stage only:
 *   logits = X @ WRouter
 *
 * The TopK stage lives in moe_topk_kernel.cpp and is compiled for the vec
 * target. This file stays cube-only because it uses TMATMUL.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace mani_moe_cfg {

constexpr unsigned kT = 256;
constexpr unsigned kH = 64;
constexpr unsigned kF = 64;
constexpr unsigned kE = 16;
constexpr unsigned kTileM = 128;

static_assert(kT % kTileM == 0, "mani_moe v1 expects kT to be a multiple of kTileM.");

}  // namespace mani_moe_cfg

template <typename TIn, typename TWeight, typename TAcc>
__global__ AICORE void runRouterGemm(__gm__ uint8_t *logits_raw,
                                     __gm__ uint8_t *x_raw,
                                     __gm__ uint8_t *w_router_raw)
{
    using namespace mani_moe_cfg;

    __gm__ TAcc *logits = reinterpret_cast<__gm__ TAcc *>(logits_raw);
    __gm__ TIn *x = reinterpret_cast<__gm__ TIn *>(x_raw);
    __gm__ TWeight *wRouter = reinterpret_cast<__gm__ TWeight *>(w_router_raw);

    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(TIn);
    constexpr int M = ((kTileM + 15) / 16) * 16;
    constexpr int K = ((kH + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((kE + blockAlign - 1) / blockAlign) * blockAlign;

    using GlobalDataX =
        GlobalTensor<TIn, Shape<1, 1, 1, kTileM, kH>,
                     Stride<1 * kTileM * kH, 1 * kTileM * kH, kTileM * kH, kH, 1>>;
    using GlobalDataW =
        GlobalTensor<TWeight, Shape<1, 1, 1, kH, kE>,
                     Stride<1 * kH * kE, 1 * kH * kE, kH * kE, kE, 1>>;
    using GlobalDataLogits =
        GlobalTensor<TAcc, Shape<1, 1, 1, kTileM, kE>,
                     Stride<1 * kTileM * kE, 1 * kTileM * kE, kTileM * kE, kE, 1>>;

    using XMatTile = Tile<TileType::Mat, TIn, M, K, BLayout::ColMajor, kTileM, kH, SLayout::RowMajor, 512>;
    using WMatTile = Tile<TileType::Mat, TWeight, K, N, BLayout::ColMajor, kH, kE, SLayout::RowMajor, 512>;
    using XLeftTile = TileLeft<TIn, M, K, kTileM, kH>;
    using WRightTile = TileRight<TWeight, K, N, kH, kE>;
    using LogitsTile = TileAcc<TAcc, M, N, kTileM, kE>;

    XMatTile xMatTile;
    WMatTile wMatTile;
    XLeftTile xTile;
    WRightTile wRouterTile;
    LogitsTile logitsTile;

    GlobalDataW wGlobal(wRouter);

    for (unsigned m0 = 0; m0 < kT; m0 += kTileM) {
        GlobalDataX xGlobal(x + static_cast<size_t>(m0) * kH);
        GlobalDataLogits logitsGlobal(logits + static_cast<size_t>(m0) * kE);

        TLOAD(xMatTile, xGlobal);
        TLOAD(wMatTile, wGlobal);
        TMOV(xTile, xMatTile);
        TMOV(wRouterTile, wMatTile);
        TMATMUL(logitsTile, xTile, wRouterTile);
        TSTORE(logitsGlobal, logitsTile);
    }
}

template <typename TIn, typename TWeight, typename TAcc>
void launchRouterGemm(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream)
{
    runRouterGemm<TIn, TWeight, TAcc><<<1, nullptr, stream>>>(logits, x, w_router);
}

template void launchRouterGemm<half, half, float>(uint8_t *logits, uint8_t *x, uint8_t *w_router, void *stream);

extern "C" void launchManiMoeRouterGemmFp16(uint8_t *logits_fp32,
                                            uint8_t *x_fp16,
                                            uint8_t *w_router_fp16,
                                            void *stream)
{
    launchRouterGemm<half, half, float>(logits_fp32, x_fp16, w_router_fp16, stream);
}
