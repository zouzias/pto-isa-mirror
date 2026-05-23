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

#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef CONFIG_DISPATCH_COMBINE_MOE_MIX_AIC_BLOCKS
#define CONFIG_DISPATCH_COMBINE_MOE_MIX_AIC_BLOCKS 20U
#endif

#ifndef CONFIG_DISPATCH_COMBINE_MOE_MIX_AIV_RATIO
#define CONFIG_DISPATCH_COMBINE_MOE_MIX_AIV_RATIO 2U
#endif

#ifndef CONFIG_DISPATCH_COMBINE_MOE_AIV_NUM
#define CONFIG_DISPATCH_COMBINE_MOE_AIV_NUM \
    (CONFIG_DISPATCH_COMBINE_MOE_MIX_AIC_BLOCKS * CONFIG_DISPATCH_COMBINE_MOE_MIX_AIV_RATIO)
#endif

#ifndef CONFIG_DISPATCH_COMBINE_MOE_BLOCK_DIM
#define CONFIG_DISPATCH_COMBINE_MOE_BLOCK_DIM \
    (CONFIG_DISPATCH_COMBINE_MOE_MIX_AIC_BLOCKS * (1U + CONFIG_DISPATCH_COMBINE_MOE_MIX_AIV_RATIO))
#endif

struct DispatchCombineMoeResourceConfig {
    uint32_t aicBlocks;
    uint32_t aivRatio;
    uint32_t aivNum;
    uint32_t blockDim;
};

inline bool StartsWith(const std::string &value, const char *prefix)
{
    return value.rfind(prefix, 0) == 0;
}

inline bool IsDispatchCombineMoeA5Soc(const std::string &socVersion)
{
    return socVersion.empty() || socVersion == "Ascend950" || StartsWith(socVersion, "Ascend950DT_") ||
           StartsWith(socVersion, "Ascend950PR_");
}

inline DispatchCombineMoeResourceConfig GetDispatchCombineMoeResourceConfig(const std::string &socVersion)
{
    if (!IsDispatchCombineMoeA5Soc(socVersion)) {
        throw std::runtime_error("unsupported dispatch_combine_moe resource config for soc: " + socVersion);
    }
    return DispatchCombineMoeResourceConfig{CONFIG_DISPATCH_COMBINE_MOE_MIX_AIC_BLOCKS,
                                            CONFIG_DISPATCH_COMBINE_MOE_MIX_AIV_RATIO,
                                            CONFIG_DISPATCH_COMBINE_MOE_AIV_NUM,
                                            CONFIG_DISPATCH_COMBINE_MOE_BLOCK_DIM};
}
