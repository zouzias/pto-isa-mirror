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

// Bandwidth sweep: single-core Ring + Recursive Doubling across multiple data sizes.
bool RunAllgatherAsyncSweep(int nRanks, int firstRankId, int firstDeviceId);

// Bandwidth sweep: multi-core TPUT_ASYNC_MC + TGET_ASYNC_MC across multiple data sizes.
bool RunAllgatherMcAsyncSweep(int nRanks, int firstRankId, int firstDeviceId);
