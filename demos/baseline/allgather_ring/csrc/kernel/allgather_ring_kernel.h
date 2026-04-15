/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

// Ring Allgather via TPUT_ASYNC (SDMA engine): N-1 rounds, each round pushes one chunk to the next rank.
bool RunAllgatherRing(int nRanks, int firstRankId, int firstDeviceId);

// Ring Allgather via TPUT (synchronous, MTE2 pipeline): same algorithm, different transport path.
bool RunAllgatherRingSync(int nRanks, int firstRankId, int firstDeviceId);

// Ring Allgather via TPUT_ASYNC + AICORE stage-through (dcci → TLOAD → TSTORE → TPUT_ASYNC).
bool RunAllgatherRingStage(int nRanks, int firstRankId, int firstDeviceId);

// Ring Allgather via TPUT_ASYNC + dcci cache invalidation before reading SDMA-received data.
bool RunAllgatherRingDcci(int nRanks, int firstRankId, int firstDeviceId);

// Diagnostic: TPUT_ASYNC with host-side readback after each round.
bool RunAllgatherRingDiag(int nRanks, int firstRankId, int firstDeviceId);

// Ring Allgather via TPUT_ASYNC with a sleep(delaySec) between rounds.
// Diagnostic: if sleep fixes it, the issue is data arrival timing; if not, it's L2 cache coherency.
bool RunAllgatherRingDelayed(int nRanks, int firstRankId, int firstDeviceId, unsigned int delaySec);
