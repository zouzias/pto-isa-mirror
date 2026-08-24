/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Single-device multi-block FFN GridPipe AllGather driver.

#define FFN_GRID_DRIVER_TITLE "Single-device multi-block FFN GridPipe AllGather demo"
#define FFN_GRID_DRIVER_MODE "hidden AllGather across cols, down split by output H columns"
#define FFN_GRID_DRIVER_PIPE_LABEL "gatherPipeBytes"
#define FFN_GRID_DRIVER_HIDDEN_BYTES FFN_HIDDEN_FULL_BYTES
#define FFN_GRID_DRIVER_VERIFY_LABEL "single-device GridPipe AllGather output shards vs golden"
#define FFN_GRID_DRIVER_SUCCESS "[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS."
#define FFN_GRID_DRIVER_FAILED "[FAILED] Single-device multi-block FFN GridPipe AllGather FAILED."
#define FFN_GRID_DRIVER_LAUNCH launchDistributedFfnGridAllGatherMixedKernel
#define FFN_GRID_DRIVER_ALLOCATE_CHECK()                                                                    \
    do {                                                                                                    \
        if (FFN_MODEL_TILE % FFN_GRID_COLS != 0) {                                                          \
            std::cerr << "[ERROR] AllGather split requires MODEL_TILE divisible by GRID_COLS" << std::endl; \
            return false;                                                                                   \
        }                                                                                                   \
    } while (0)

#include "ffn_grid_driver_inl.hpp"
