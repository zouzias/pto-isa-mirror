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
 * This file intentionally wires the template/kernel/launcher shape only.
 * The computation body is left for manual implementation.
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

template <typename TIn, typename TWeight, typename TAcc, typename TIdx>
__global__ AICORE void runManiMoe(__gm__ uint8_t *z_raw,
                                  __gm__ uint8_t *logits_raw,
                                  __gm__ uint8_t *expert_id_raw,
                                  __gm__ uint8_t *x_raw,
                                  __gm__ uint8_t *w_router_raw,
                                  __gm__ uint8_t *w1_raw,
                                  __gm__ uint8_t *w2_raw)
{
    using namespace mani_moe_cfg;

    __gm__ TAcc *z = reinterpret_cast<__gm__ TAcc *>(z_raw);
    __gm__ TAcc *logits = reinterpret_cast<__gm__ TAcc *>(logits_raw);
    __gm__ TIdx *expertId = reinterpret_cast<__gm__ TIdx *>(expert_id_raw);
    __gm__ TIn *x = reinterpret_cast<__gm__ TIn *>(x_raw);
    __gm__ TWeight *wRouter = reinterpret_cast<__gm__ TWeight *>(w_router_raw);
    __gm__ TWeight *w1 = reinterpret_cast<__gm__ TWeight *>(w1_raw);
    __gm__ TWeight *w2 = reinterpret_cast<__gm__ TWeight *>(w2_raw);

    // TODO(mani):
    //   1. Router:
    //        logits[t, e] = sum_h x[t, h] * wRouter[h, e]
    //        expertId[t] = argmax_e logits[t, e]
    //
    //   2. Expert FFN:
    //        hidden[f] = relu(sum_h x[t, h] * w1[expertId[t], h, f])
    //        z[t, h]   = sum_f hidden[f] * w2[expertId[t], f, h]
    //
    //   Existing implementations worth copying from in small pieces:
    //     - moe_router_top1: router GEMM + TROWARGMAX skeleton.
    //     - moe_segmented_gemm_relu: one expert GEMM + fused ReLU pattern.
    //     - moe_segmented_ffn_top1: split FFN1/FFN2 pattern.
    //
    //   Auto-mode reminder:
    //     if a nested loop or a complex load/compute/store pattern compiles
    //     but fails/hangs, also suspect auto-sync. A temporary #pragma unroll
    //     on the relevant loop may help the compiler recognize the pattern.
    (void)kT;
    (void)kH;
    (void)kF;
    (void)kE;
    (void)kTileM;
    (void)z;
    (void)logits;
    (void)expertId;
    (void)x;
    (void)wRouter;
    (void)w1;
    (void)w2;
}

template <typename TIn, typename TWeight, typename TAcc, typename TIdx>
void launchManiMoe(uint8_t *z,
                   uint8_t *logits,
                   uint8_t *expert_id,
                   uint8_t *x,
                   uint8_t *w_router,
                   uint8_t *w1,
                   uint8_t *w2,
                   void *stream)
{
    runManiMoe<TIn, TWeight, TAcc, TIdx><<<1, nullptr, stream>>>(z, logits, expert_id, x, w_router, w1, w2);
}

template void launchManiMoe<half, half, float, uint32_t>(uint8_t *z,
                                                         uint8_t *logits,
                                                         uint8_t *expert_id,
                                                         uint8_t *x,
                                                         uint8_t *w_router,
                                                         uint8_t *w1,
                                                         uint8_t *w2,
                                                         void *stream);

extern "C" void launchManiMoeFp16(uint8_t *z_fp32,
                                  uint8_t *logits_fp32,
                                  uint8_t *expert_id_u32,
                                  uint8_t *x_fp16,
                                  uint8_t *w_router_fp16,
                                  uint8_t *w1_fp16,
                                  uint8_t *w2_fp16,
                                  void *stream)
{
    launchManiMoe<half, half, float, uint32_t>(z_fp32, logits_fp32, expert_id_u32, x_fp16, w_router_fp16, w1_fp16,
                                               w2_fp16, stream);
}
