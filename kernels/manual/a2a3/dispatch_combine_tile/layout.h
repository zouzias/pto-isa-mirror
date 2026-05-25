/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_LAYOUT_H_
#define DISPATCH_COMBINE_TILE_LAYOUT_H_

#include "common.h"

#include <cstdint>

namespace dispatch_combine_tile {

// Task 2 implements these host-side layout entry points. The formulas must stay
// identical to the offsets used by device views in later PTO kernels.
uint64_t AlignUp(uint64_t value, uint64_t alignment);
WorkspaceLayout ComputeWorkspaceLayout(const DispatchCombineTileShape &shape);
PeerWindowLayout ComputePeerWindowLayout(const DispatchCombineTileShape &shape);
uint64_t EstimateHcclBuffSizeMb(const DispatchCombineTileShape &shape, const PeerWindowLayout &peerWindowLayout);

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_LAYOUT_H_
