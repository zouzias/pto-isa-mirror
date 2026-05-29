/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_KERNEL_LAUNCHERS_H_
#define DISPATCH_COMBINE_TILE_KERNEL_LAUNCHERS_H_

#include "moe_dispatch_combine_a8w8_runtime_types.hpp"
#include "moe_dispatch_combine_a8w8_types.hpp"

#include <cstdint>

namespace dispatch_combine_tile {

// Fixed host launcher ABI. Task 1 adds the device launch expressions without
// changing this order: shape, rank, data pointers, peerWindow, hcclCtx,
// workspace, stream, launch block count.
void LaunchDispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *inputA,
                                       uint8_t *expertIdx, uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace,
                                       void *stream, uint32_t launchBlockCount);

void LaunchDispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2Int8Dispatch(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                          uint8_t *inputA, uint8_t *expertIdx, uint8_t *xActiveMask, uint8_t *peerWindow,
                          uint8_t *hcclCtx, uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2Gmm1Int8(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                      uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2Gmm1Epilogue(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                          uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2ActivationQuant(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                             uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2Gmm2Int8(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                      uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2Gmm2EpilogueAndReturn(moe_dispatch_combine_a8w8::ShapeConfig shape,
                                   moe_dispatch_combine_a8w8::RankConfig rank, uint8_t *peerWindow, uint8_t *hcclCtx,
                                   uint8_t *workspace, void *stream, uint32_t launchBlockCount);

void LaunchM2RestoreOutput(moe_dispatch_combine_a8w8::ShapeConfig shape, moe_dispatch_combine_a8w8::RankConfig rank,
                           uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *workspace, void *stream,
                           uint32_t launchBlockCount);

void LaunchM2MixedSpike(uint8_t *workspace, uint32_t rank, uint32_t aicBlocks, uint32_t aivRatio, void *stream);

void LaunchM2FusedSkeleton(uint8_t *workspace, uint32_t rank, uint32_t aicBlocks, uint32_t aivRatio, void *stream);

void LaunchM2FusedFull(moe_dispatch_combine_a8w8::M2FusedFullLaunchArgs *launchArgsDevice, uint32_t aicBlocks,
                       uint32_t aivRatio, void *stream);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_KERNEL_LAUNCHERS_H_
