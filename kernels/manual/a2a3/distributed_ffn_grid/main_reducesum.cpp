/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Single-device multi-block FFN GridPipe ReduceSum driver.

#define FFN_GRID_DRIVER_TITLE "Single-device multi-block FFN GridPipe demo"
#define FFN_GRID_DRIVER_MODE "row=data-parallel, col=model-parallel on one device; EAST reduce uses local GM windows"
#define FFN_GRID_DRIVER_PIPE_LABEL "reducePipeBytes"
#define FFN_GRID_DRIVER_HIDDEN_BYTES FFN_HIDDEN_BYTES
#define FFN_GRID_DRIVER_VERIFY_LABEL "single-device GridPipe EAST-reduced output vs golden"
#define FFN_GRID_DRIVER_SUCCESS "[SUCCESS] Single-device multi-block FFN GridPipe PASS."
#define FFN_GRID_DRIVER_FAILED "[FAILED] Single-device multi-block FFN GridPipe FAILED."
#define FFN_GRID_DRIVER_LAUNCH launchDistributedFfnGridMixedKernel

#include "ffn_grid_driver_inl.hpp"
