/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "common.h"
#include "kernel_launchers.h"

using dispatch_combine_tile::DispatchCombineTileShape;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

// Kernel source contract for later tasks:
//   - Only PTO and C/C++ headers are allowed in this file.
//   - Device payload movement will use pto::GlobalTensor and pto::Tile views.
//   - Cross-rank movement/readiness will use PTO comm primitives.
//   - There are exactly two public device kernel names in this project.

__global__ AICORE void DispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, GM_ADDR inputA,
                                                   GM_ADDR expertIdx, GM_ADDR peerWindow, GM_ADDR hcclCtx,
                                                   GM_ADDR workspace)
{
    (void)shape;
    (void)myRank;
    (void)inputA;
    (void)expertIdx;
    (void)peerWindow;
    (void)hcclCtx;
    (void)workspace;
}

__global__ AICORE void DispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank,
                                                  GM_ADDR expertOutput, GM_ADDR probs, GM_ADDR outputC,
                                                  GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
{
    (void)shape;
    (void)myRank;
    (void)expertOutput;
    (void)probs;
    (void)outputC;
    (void)peerWindow;
    (void)hcclCtx;
    (void)workspace;
}

namespace dispatch_combine_tile {

void LaunchDispatchCombineTileDispatch(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *inputA,
                                       uint8_t *expertIdx, uint8_t *peerWindow, uint8_t *hcclCtx, uint8_t *workspace,
                                       void *stream,
                                       uint32_t launchBlockCount)
{
    DispatchCombineTileDispatch<<<launchBlockCount, nullptr, stream>>>(
        shape, myRank, inputA, expertIdx, peerWindow, hcclCtx, workspace);
}

void LaunchDispatchCombineTileCombine(DispatchCombineTileShape shape, uint32_t myRank, uint8_t *expertOutput,
                                      uint8_t *probs, uint8_t *outputC, uint8_t *peerWindow, uint8_t *hcclCtx,
                                      uint8_t *workspace, void *stream, uint32_t launchBlockCount)
{
    DispatchCombineTileCombine<<<launchBlockCount, nullptr, stream>>>(
        shape, myRank, expertOutput, probs, outputC, peerWindow, hcclCtx, workspace);
}

} // namespace dispatch_combine_tile
