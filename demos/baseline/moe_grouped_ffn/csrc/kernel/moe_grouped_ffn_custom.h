/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOE_GROUPED_FFN_CUSTOM_H
#define PTO_MOE_GROUPED_FFN_CUSTOM_H

#include <cstddef>
#include <cstdint>

constexpr int kMoeHiddenSize = 4096;
constexpr int kMoeInterSize = 4096;
constexpr int kMoeBaseM = 64;
constexpr int kMoeBaseK = 64;
constexpr int kMoeBaseN = 256;
constexpr int kMoeNumNTiles = kMoeInterSize / kMoeBaseN;
constexpr int kMoeFifoDepth = 2;
constexpr int kMoeStepKa = 4;
constexpr int kMoeStepKb = 4;
constexpr uint32_t kMoeWorkspaceAlignBytes = 512;
constexpr uint32_t kMoeCvCommSlotBytes = 512;
constexpr uint32_t kMoeCvMaxSlots = 25;
constexpr uint32_t kMoeVecSubBlockCount = 2;
constexpr int kMoeVecSubBlockRows = kMoeBaseM / static_cast<int>(kMoeVecSubBlockCount);
constexpr std::size_t kMoeWeightTileBytes = static_cast<std::size_t>(kMoeBaseK) * kMoeBaseN * sizeof(uint16_t);
constexpr std::size_t kMoeVecTileBytes =
    static_cast<std::size_t>(kMoeBaseM) * kMoeBaseN * sizeof(float);
constexpr std::size_t kMoeVecSubTileBytes =
    static_cast<std::size_t>(kMoeVecSubBlockRows) * kMoeBaseN * sizeof(float);
constexpr std::size_t kMoeVecTileElems = static_cast<std::size_t>(kMoeBaseM) * kMoeBaseN;
constexpr std::size_t kMoeMaxVecUbBytes = 192U * 1024U;
// gate tile + scratch tile + recip tile
constexpr std::size_t kMoeVecResidentTileCount = 3;
constexpr std::size_t kMoeVecResidentBytes = kMoeVecResidentTileCount * kMoeVecSubTileBytes;

constexpr std::size_t RoundUpMoeWorkspace(std::size_t bytes)
{
    return ((bytes + kMoeWorkspaceAlignBytes - 1) / kMoeWorkspaceAlignBytes) * kMoeWorkspaceAlignBytes;
}

constexpr uint32_t GetMoeCommSlotCount(uint32_t blockDim)
{
    // Stable-first policy: give each logical block a dedicated projection slot.
    // TSYNC_CVID-based slot reuse can be reintroduced after the wrapped-slot protocol is hardened.
    return blockDim;
}

constexpr bool UseMoeCvComm(uint32_t blockDim)
{
    (void)blockDim;
    // Keep CV comm workspace reserved for future slot reuse, but do not enable it in the stable path.
    // This avoids large-shape slot aliasing across block waves.
    return false;
}

constexpr std::size_t GetMoeProjectionBufferBytes(uint32_t blockDim)
{
    return RoundUpMoeWorkspace(
        static_cast<std::size_t>(GetMoeCommSlotCount(blockDim)) * kMoeFifoDepth * kMoeVecTileBytes);
}

constexpr std::size_t GetMoeCvCommBytes(uint32_t blockDim)
{
    return RoundUpMoeWorkspace(static_cast<std::size_t>(blockDim) * kMoeCvCommSlotBytes);
}

constexpr std::size_t GetMoeUserWorkspaceBytes(uint32_t blockDim)
{
    return 2 * GetMoeProjectionBufferBytes(blockDim) + GetMoeCvCommBytes(blockDim);
}

static_assert(kMoeHiddenSize % kMoeBaseK == 0, "kMoeHiddenSize must be divisible by kMoeBaseK");
static_assert(kMoeInterSize % kMoeBaseN == 0, "kMoeInterSize must be divisible by kMoeBaseN");
static_assert(kMoeNumNTiles > 0, "kMoeInterSize must produce at least one N tile");
static_assert(kMoeBaseM % static_cast<int>(kMoeVecSubBlockCount) == 0,
              "kMoeBaseM must divide evenly across vector subblocks");
static_assert(kMoeVecResidentBytes <= kMoeMaxVecUbBytes, "Vec tile UB allocation exceeds 192KB");
static_assert(kMoeNumNTiles >= kMoeFifoDepth,
              "N-tile count must be at least kMoeFifoDepth for drain correctness");
static_assert((kMoeFifoDepth & (kMoeFifoDepth - 1)) == 0,
              "kMoeFifoDepth must be a power of two for efficient modulo");

#endif // PTO_MOE_GROUPED_FFN_CUSTOM_H
