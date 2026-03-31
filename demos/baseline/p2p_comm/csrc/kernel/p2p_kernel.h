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

// Host-side entry points for the P2P communication demo.
// These are defined in p2p_kernel.cpp (compiled with -xcce) which
// internally launches the __global__ AICORE kernels.

// TPUT demo: Rank 0 sends data to Rank 1 via pto::comm::TPUT.
// Returns true on success.
bool RunTPutDemo(int nRanks, int firstRankId, int firstDeviceId);

// TGET demo: Rank 1 pulls data from Rank 0 via pto::comm::TGET.
// Returns true on success.
bool RunTGetDemo(int nRanks, int firstRankId, int firstDeviceId);
