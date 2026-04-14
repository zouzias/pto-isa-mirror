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

// Bandwidth sweep covering 4 kernel strategies:
//   TPUT_MC, TGET_MC, RING, REC_DBL
// Each strategy is measured with warmup + timed iterations.
// Output includes host/device latency and bandwidth for each data size.
bool RunAllgatherSyncSweep(int nRanks, int firstRankId, int firstDeviceId);
