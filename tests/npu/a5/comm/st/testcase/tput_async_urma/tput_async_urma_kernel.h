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

#include <cstddef>
#include <cstdint>

template <typename T, size_t count>
bool RunPutAsyncUrmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// How the root rank maps (AIV, peer) onto a SharedPool jetty.
enum class UrmaPoolJettyPolicy : int {
    // One AIV posts to every peer on jetty 0. Covers "one jetty, many peers".
    SingleJetty = 0,
    // AIV i posts to every peer on jetty i. Covers "many AIVs, distinct jetties".
    AivOwnsJetty = 1,
    // One AIV posts peer p on jetty (p % nJetty). Covers "one AIV, many jetties".
    RoundRobinJetty = 2,
};

template <typename T, size_t count, int nAiv, int nJetty, UrmaPoolJettyPolicy policy>
bool RunPutAsyncUrmaPool(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
