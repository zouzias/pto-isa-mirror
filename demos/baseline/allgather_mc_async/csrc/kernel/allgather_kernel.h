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

// Multi-core AllGather via TPUT_ASYNC: each AICORE handles one target rank.
bool RunAllgatherPutAsyncMC(int nRanks, int firstRankId, int firstDeviceId);

// Multi-core AllGather via TGET_ASYNC: each AICORE handles one source rank.
bool RunAllgatherGetAsyncMC(int nRanks, int firstRankId, int firstDeviceId);

// Bandwidth sweep: TPUT_ASYNC_MC + TGET_ASYNC_MC across multiple data sizes.
bool RunAllgatherMcAsyncSweep(int nRanks, int firstRankId, int firstDeviceId);
