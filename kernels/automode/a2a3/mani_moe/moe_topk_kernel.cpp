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
 * moe_topk_kernel.cpp - vec target TopK stage for mani_moe.
 *
 * This TU intentionally includes the existing topk kernel implementation and
 * is compiled with dav-c220-vec. The router GEMM stage is compiled separately
 * in moe_kernel.cpp with dav-c220-cube.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "../topk/topk_kernel.cpp"

using namespace pto;

namespace mani_moe_topk_cfg {

constexpr unsigned kT = 256;
constexpr unsigned kE = 16;
constexpr unsigned kTopKScratch = 8;

}  // namespace mani_moe_topk_cfg

template <typename TIdx>
__global__ AICORE void runMakeTopkIdx(__gm__ uint8_t *idx_raw)
{
    using namespace mani_moe_topk_cfg;

    using IdxTile = Tile<TileType::Vec, TIdx, 1, kE, BLayout::RowMajor, -1, -1>;
    using IdxGlobal = GlobalTensor<TIdx, Shape<1, 1, 1, 1, kE>, Stride<1, 1, 1, kE, 1>>;

    __gm__ TIdx *idx = reinterpret_cast<__gm__ TIdx *>(idx_raw);
    IdxTile idxTile(1, kE);
    IdxGlobal idxGlobal(idx);

    TCI<IdxTile, TIdx, 0>(idxTile, 0);
    TSTORE(idxGlobal, idxTile);
}

template <typename TIdx>
void launchMakeTopkIdx(uint8_t *idx, void *stream)
{
    runMakeTopkIdx<TIdx><<<1, nullptr, stream>>>(idx);
}

template <typename TIdx>
__global__ AICORE void runExtractTop1Idx(__gm__ uint8_t *expert_id_raw, __gm__ uint8_t *topk_out_idx_raw)
{
    using namespace mani_moe_topk_cfg;

    __gm__ TIdx *expertId = reinterpret_cast<__gm__ TIdx *>(expert_id_raw);
    __gm__ TIdx *topkOutIdx = reinterpret_cast<__gm__ TIdx *>(topk_out_idx_raw);

    for (unsigned row = 0; row < kT; ++row) {
        expertId[row] = topkOutIdx[row * kTopKScratch];
    }
}

template <typename TIdx>
void launchExtractTop1Idx(uint8_t *expert_id, uint8_t *topk_out_idx, void *stream)
{
    runExtractTop1Idx<TIdx><<<1, nullptr, stream>>>(expert_id, topk_out_idx);
}

template void launchMakeTopkIdx<uint32_t>(uint8_t *idx, void *stream);
template void launchExtractTop1Idx<uint32_t>(uint8_t *expert_id, uint8_t *topk_out_idx, void *stream);

extern "C" void launchManiMoeTopkFp32(uint8_t *topk_values_fp32,
                                      uint8_t *expert_id_u32,
                                      uint8_t *logits_fp32,
                                      uint8_t *topk_idx_u32,
                                      uint8_t *topk_out_idx_u32,
                                      void *stream)
{
    launchMakeTopkIdx<uint32_t>(topk_idx_u32, stream);
    RunTopk<float, mani_moe_topk_cfg::kT, mani_moe_topk_cfg::kE, mani_moe_topk_cfg::kTopKScratch>
        <<<1, nullptr, stream>>>(topk_values_fp32, topk_out_idx_u32, logits_fp32, topk_idx_u32);
    launchExtractTop1Idx<uint32_t>(expert_id_u32, topk_out_idx_u32, stream);
}
