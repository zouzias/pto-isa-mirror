/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_HCCL_CONTEXT_H_
#define DISPATCH_COMBINE_TILE_HCCL_CONTEXT_H_

#include "common.h"

#include <cstdint>

namespace dispatch_combine_tile {

constexpr uint32_t kMaxDispatchCombineTileRanks = 64;

struct HcclDeviceContext {
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[kMaxDispatchCombineTileRanks];
};

struct HcclWindowContext {
    HcclDeviceContext hostDeviceContext;
    void *deviceContext;
    void *localWindowBase;
    void *peerWindow;
    uint64_t peerWindowOffset;
};

// Task 3 adapts kernels/manual/a2a3/gemm_ar HCCL context handling: root-info
// broadcast, MESH/RING windowsIn extraction, device-visible HcclDeviceContext,
// and fixed-offset peerWindow slicing. Host code may use ACL/HCCL/MPI here; no
// symmetric-memory runtime is part of this project.
HcclWindowContext InitHcclWindowContext(const DispatchCombineTileArgs &args, uint32_t myRank);
void DestroyHcclWindowContext(HcclWindowContext *context);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_HCCL_CONTEXT_H_
