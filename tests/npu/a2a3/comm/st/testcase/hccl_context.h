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
#include "pto/comm/domain/comm_device_context.hpp"

// ============================================================================
// CommDeviceContext
//
// Simplified device-side context for TPUT/TGET tests.
// On MESH topology (A2), HCCL returns HcclCombinOpParamA5 whose first fields
// match this struct directly (windowsIn[64] already contains per-rank RDMA
// addresses).
// On RING topology (A3), we build this struct manually on the host by
// extracting remote RDMA addresses from CommOpResParam's remoteRes array.
//
// Now aliased to pto::comm::domain::CommDeviceContext which has identical
// layout, allowing tests to directly use domain device APIs.
// ============================================================================

static constexpr uint32_t HCCL_MAX_RANK_NUM = pto::comm::domain::kMaxRankNum;

using CommDeviceContext = pto::comm::domain::CommDeviceContext;
