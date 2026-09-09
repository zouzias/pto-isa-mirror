/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "native/IFU.h"

#include <algorithm>
#include <utility>

namespace vfsim {

IFU::IFU(
    std::vector<DynamicInst> expandedInstructions, std::unordered_map<int, std::vector<int64_t>> topBlockLoopBounds,
    int64_t totalTopBlocks)
    : totalTopBlocks_(std::max<int64_t>(1, totalTopBlocks)), topBlockLoopBounds_(std::move(topBlockLoopBounds))
{
    for (auto& instruction : expandedInstructions)
        pending_.push_back(std::move(instruction));
}

std::optional<DynamicInst> IFU::nextInst()
{
    if (pending_.empty())
        return std::nullopt;
    DynamicInst instruction = std::move(pending_.front());
    pending_.pop_front();
    return instruction;
}

std::vector<DynamicInst> IFU::take(int64_t n)
{
    std::vector<DynamicInst> result;
    for (int64_t index = 0; index < std::max<int64_t>(0, n); ++index) {
        auto instruction = nextInst();
        if (!instruction)
            break;
        result.push_back(std::move(*instruction));
    }
    return result;
}

} // namespace vfsim
