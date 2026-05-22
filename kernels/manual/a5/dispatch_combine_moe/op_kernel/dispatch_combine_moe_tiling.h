/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file dispatch_combine_moe_tiling.h
 * \brief
 */

#include <stddef.h>
#include <stdint.h>

#include "token_reorder/routing/moe_init_routing_tiling_common.h"
#include "token_reorder/routing/moe_init_routing_quant_tiling.h"

#ifndef ASCENDC_DISPATCH_COMBINE_MOE_TILING_H
#define ASCENDC_DISPATCH_COMBINE_MOE_TILING_H
struct DispatchCombineMoeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t isTransposeB;
    uint32_t isWeightNz;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t listLen;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t swizzleDirect = -1;
    int32_t swizzleOffset = -1;
    int32_t ubMoveNum = -1;
    int32_t pValue = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
    uint64_t initRoutingQuantTilingKey;
    optiling::MoeInitRoutingQuantTilingData moeInitRoutingQuantTilingData;
};

struct DispatchCombineMoeRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

struct DispatchCombineMoeLaunchConfig {
    uint32_t blockDim = 1;
    uint32_t tilingKey = 0;
    uint64_t workspaceBytes = 0;
};

struct DispatchCombineMoeTilingData {
    DispatchCombineMoeInfo dispatchCombineMoeInfo;
    CoCTiling cocTiling;
    DispatchCombineMoeRuntimeInfo runtimeInfo;
    DispatchCombineMoeLaunchConfig launchConfig;
};
#endif
