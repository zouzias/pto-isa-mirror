/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H
#define DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H

#include <pto/pto-inst.hpp>

#include "utils/const_args.hpp"

constexpr uint16_t kGmm2CombineCvReadyFlagBase = 3U;
constexpr uint16_t kGmm2CombineCvFreeFlagBase =
    kGmm2CombineCvReadyFlagBase + kMegaMoeGmm2CombineCvFifoDepth;
constexpr uint32_t kGmm2CombineCvTileRows = kMegaMoeGmm2CombineCvTileRows;
constexpr uint32_t kGmm2CombineCvTileCols = kMegaMoeGmm2CombineCvTileCols;
constexpr uint32_t kGmm2CombineCvFifoDepth = kMegaMoeGmm2CombineCvFifoDepth;
constexpr uint32_t kGmm2CombineCvSlotBytes = kMegaMoeGmm2CombineCvSlotBytes;
constexpr uint32_t kGmm2CombineCvBufferOffset = 0U;
constexpr uint32_t kGmm2CombineCvBufferBytes = kGmm2CombineCvFifoDepth * kGmm2CombineCvSlotBytes;
constexpr uint32_t kGmm2CombineMetadataOffset = kGmm2CombineCvBufferOffset + kGmm2CombineCvBufferBytes;
constexpr uint32_t kGmm2CombineMetadataSlotBytes = kMegaMoeGmm2CombineMetadataSlotBytes;
constexpr uint32_t kGmm2CombineMetadataBytes = kGmm2CombineCvFifoDepth * kGmm2CombineMetadataSlotBytes;
constexpr uint32_t kGmm2CombineRequiredUbBytes = kMegaMoeGmm2CombineRequiredUbBytes;

struct Gmm2CombineCvPipe {
    struct Endpoint {
        uint32_t tileIndex = 0U;
    };

    Endpoint prod;
    Endpoint cons;
};

using Gmm2CombineCvTile =
    pto::Tile<pto::TileType::Vec, bfloat16_t, kGmm2CombineCvTileRows, kGmm2CombineCvTileCols,
              pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::NoneBox>;

static_assert(kGmm2CombineCvSlotBytes == 64U * 1024U);
static_assert(kGmm2CombineCvBufferBytes == 192U * 1024U);
static_assert(kGmm2CombineRequiredUbBytes <= A5_MAIN_UB_SIZE);
static_assert(kGmm2CombineCvFreeFlagBase + kGmm2CombineCvFifoDepth <= 16U);
static_assert(kGmm2CombineCvFreeFlagBase + kGmm2CombineCvFifoDepth - 1U +
                  kMegaMoeFixedSecondAivSubblockFlagOffset <
              32U);

AICORE inline uint16_t Gmm2CombineAiv1Flag(uint16_t logicalFlag)
{
    return static_cast<uint16_t>(logicalFlag + kMegaMoeFixedSecondAivSubblockFlagOffset);
}

AICORE inline uint16_t Gmm2CombineReadyFlag(uint32_t slot)
{
    return static_cast<uint16_t>(kGmm2CombineCvReadyFlagBase + slot);
}

AICORE inline uint16_t Gmm2CombineFreeFlag(uint32_t slot)
{
    return static_cast<uint16_t>(kGmm2CombineCvFreeFlagBase + slot);
}

AICORE inline uint64_t Gmm2CombineSlotOffset(uint32_t tileIndex)
{
    return kGmm2CombineCvBufferOffset +
           static_cast<uint64_t>(tileIndex % kGmm2CombineCvFifoDepth) * kGmm2CombineCvSlotBytes;
}

AICORE inline void Gmm2CombineProducerAllocate(Gmm2CombineCvPipe &pipe)
{
    if (pipe.prod.tileIndex >= kGmm2CombineCvFifoDepth) {
        const uint32_t slot = pipe.prod.tileIndex % kGmm2CombineCvFifoDepth;
        wait_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(Gmm2CombineFreeFlag(slot)));
    }
}

AICORE inline void Gmm2CombineProducerRecord(Gmm2CombineCvPipe &pipe)
{
    const uint32_t slot = pipe.prod.tileIndex % kGmm2CombineCvFifoDepth;
    set_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(Gmm2CombineReadyFlag(slot)));
    ++pipe.prod.tileIndex;
}

AICORE inline void Gmm2CombineProducerDrainWave(Gmm2CombineCvPipe &pipe)
{
    const uint32_t outstanding =
        pipe.prod.tileIndex < kGmm2CombineCvFifoDepth ? pipe.prod.tileIndex : kGmm2CombineCvFifoDepth;
    for (uint32_t slot = 0U; slot < outstanding; ++slot) {
        wait_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(Gmm2CombineFreeFlag(slot)));
    }
    pipe.prod.tileIndex = 0U;
}

AICORE inline void Gmm2CombineConsumerWait(Gmm2CombineCvPipe &pipe)
{
    const uint32_t slot = pipe.cons.tileIndex % kGmm2CombineCvFifoDepth;
    wait_intra_block(PIPE_MTE3, Gmm2CombineReadyFlag(slot));
}

AICORE inline void Gmm2CombineConsumerRelease(Gmm2CombineCvPipe &pipe)
{
    const uint32_t slot = pipe.cons.tileIndex % kGmm2CombineCvFifoDepth;
    set_intra_block(PIPE_MTE3, Gmm2CombineFreeFlag(slot));
    ++pipe.cons.tileIndex;
}

AICORE inline void Gmm2CombineConsumerFinishWave(Gmm2CombineCvPipe &pipe)
{
    pipe.cons.tileIndex = 0U;
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H
