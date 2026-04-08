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

// Allgather via TPUT: every rank writes its data to all other ranks (synchronous).
bool RunAllgatherPutSync(int nRanks, int firstRankId, int firstDeviceId);

// Allgather via TGET: every rank pulls data from all other ranks (synchronous).
bool RunAllgatherGetSync(int nRanks, int firstRankId, int firstDeviceId);

// Bandwidth sweep: runs TPUT_SYNC + TGET_SYNC across multiple data sizes (1KB..4MB).
bool RunAllgatherSyncSweep(int nRanks, int firstRankId, int firstDeviceId);
