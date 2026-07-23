/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

enum class RoceHns1825TestResult {
    PASSED,
    FAILED,
    SKIPPED,
};

enum class RoceHns1825CompletionMode : uint32_t {
    STATUS_WAIT_EACH,
    STATUS_WAIT_LAST,
    PUBLIC_EVENT_WAIT_TEST,
};

template <typename T, size_t count>
RoceHns1825TestResult RunPutAsyncHns1825RootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

template <typename T, size_t count>
RoceHns1825TestResult RunPutAsyncHns1825RootPutPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completion_mode);

// Shared host entry point used by the HNS1825 GET test target.
template <typename T, size_t count>
RoceHns1825TestResult RunGetAsyncHns1825RootGetPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RoceHns1825CompletionMode completion_mode);
