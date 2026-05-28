/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_HOST_HCCL_WINDOW_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_HOST_HCCL_WINDOW_HPP_

#include "moe_dispatch_combine_a8w8_types.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace moe_dispatch_combine_a8w8 {

struct HcclWindowPlan {
    uint64_t peerWindowBytes = 0;
    uint64_t requestedWindowBytes = 0;
};

inline HcclWindowPlan MakeHcclWindowPlan(const PeerWindowLayout &layout, uint64_t hcclBuffSizeMb)
{
    constexpr uint64_t kMiB = 1024ULL * 1024ULL;
    HcclWindowPlan plan;
    plan.peerWindowBytes = layout.totalBytes;
    plan.requestedWindowBytes = hcclBuffSizeMb == 0 ? 0 : hcclBuffSizeMb * kMiB;
    return plan;
}

inline void CheckWindowSize(const HcclWindowPlan &plan)
{
    if (plan.requestedWindowBytes != 0 && plan.peerWindowBytes > plan.requestedWindowBytes) {
        throw std::runtime_error("peer window layout exceeds requested HCCL window size");
    }
}

inline void PrintHcclWindowPlan(std::ostream &os, const HcclWindowPlan &plan)
{
    os << "[HcclWindowPlan]\n";
    os << "  peer_window_bytes=" << plan.peerWindowBytes << "\n";
    os << "  requested_window_bytes=" << plan.requestedWindowBytes << "\n";
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_HOST_HCCL_WINDOW_HPP_
