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
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"

static constexpr size_t kCcuCount = 1024;

using ShapeS = pto::Shape<1, 1, 1, 1, kCcuCount>;
using StrideS = pto::Stride<kCcuCount, kCcuCount, kCcuCount, kCcuCount, 1>;
using GmType = pto::GlobalTensor<float, ShapeS, StrideS, pto::Layout::ND>;
using TileT = pto::Tile<pto::TileType::Vec, float, 1, kCcuCount, pto::BLayout::RowMajor, -1, -1>;

// HostManaged path: host already wrote input; AIV only triggers CKE.
__global__ __aicore__ void tall_gather_ccu_trigger_kernel(uint64_t ckeVA, uint32_t mask)
{
    if (get_block_idx() != 0)
        return;

    GmType srcGm;
    GmType dstGm;
    TileT tile;

    pto::comm::CcuTriggerContext ctx{ckeVA, mask, /*selfIdx=*/0, pto::comm::CcuInputSource::HostManaged};

    pto::comm::TALL_GATHER<pto::comm::CollEngine::CCU>(srcGm, dstGm, tile, ctx);
}

extern "C" __attribute__((visibility("default"))) int tall_gather_ccu_trigger_launch(void *stream, uint64_t ckeVA,
                                                                                      uint32_t mask)
{
    tall_gather_ccu_trigger_kernel<<<1, nullptr, stream>>>(ckeVA, mask);
    return 0;
}

// AivStored path: AIV fills tile with fillValue then IMPL TSTOREs to srcGlobalData + triggers.
__global__ __aicore__ void tall_gather_ccu_fused_kernel(__gm__ float *inputVa, __gm__ float *outputVa,
                                                        uint32_t selfIdx, int nranks, float fillValue, uint64_t ckeVA,
                                                        uint32_t mask)
{
    if (get_block_idx() != 0)
        return;

    TileT srcTile(1, kCcuCount);
    TASSIGN(srcTile, 0x0);

    GmType inputGm(inputVa);
    GmType outputGm(outputVa);

    TEXPANDS(srcTile, fillValue);

    pto::comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, pto::comm::CcuInputSource::AivStored};

    pto::comm::TALL_GATHER<pto::comm::CollEngine::CCU>(inputGm, outputGm, srcTile, ctx);
}

extern "C" __attribute__((visibility("default"))) int tall_gather_ccu_fused_launch(void *stream, uint64_t inputVa,
                                                                                    uint64_t outputVa,
                                                                                    uint32_t selfIdx, int nranks,
                                                                                    float fillValue, uint64_t ckeVA,
                                                                                    uint32_t mask)
{
    tall_gather_ccu_fused_kernel<<<1, nullptr, stream>>>(reinterpret_cast<__gm__ float *>(inputVa),
                                                          reinterpret_cast<__gm__ float *>(outputVa), selfIdx, nranks,
                                                          fillValue, ckeVA, mask);
    return 0;
}
