/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "tprefetch_compare_kernel.h"
#include "tprefetch_l2_kernel.h"

// ============================================================================
// Focused single-card tests for device-side async L2 prefetch.
//
// Functionality:
//   - GlobalTensor API correctness
//   - raw pointer API correctness
//   - TPREFETCH_L2 must reduce TLOAD latency versus cold TLOAD
//
// Performance comparison:
//   - Scenario A: host PTO_PREFETCH vs device TPREFETCH_L2 at small/large sizes
//   - Scenario G: static-address multi-stage workload
//   - Scenario H: data-dependent multi-stage workload
// ============================================================================

TEST(TPrefetchAsync, Correctness_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2Correctness<float, 4096>(0)));
}

TEST(TPrefetchAsync, Correctness_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2Correctness<int32_t, 4096>(0)));
}

TEST(TPrefetchAsync, RawPtr_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2RawPtr<float, 4096>(0)));
}

TEST(TPrefetchAsync, RawPtr_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2RawPtr<int32_t, 4096>(0)));
}

TEST(TPrefetchAsync, PrefetchReducesTloadLatency_Float_1MB)
{
    ASSERT_TRUE((RunTloadPerf<float, 262144>(0)));
}

TEST(TPrefetchAsync, A_EndToEnd_1MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 262144>(0)));
}

TEST(TPrefetchAsync, A_EndToEnd_128MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 33554432>(0)));
}

TEST(TPrefetchAsync, G_StaticAddress_M16_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioGFusedComputePrefetch<float, 262144>(0, 16, 5000)));
}

TEST(TPrefetchAsync, H_DataDependent_M16_1MB_compute50us)
{
    ASSERT_TRUE((RunScenarioHDependentPrefetch<float, 262144>(0, 16, 5000)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
