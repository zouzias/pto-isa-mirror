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

#include "tprefetch_async_kernel.h"

// ============================================================================
// Focused single-card tests for device-side async L2 prefetch.
//
// Functionality:
//   - GlobalTensor API correctness
//   - async prefetch must reduce TLOAD latency versus cold TLOAD
//
// Performance comparison:
//   - host PTO_PREFETCH vs device async prefetch at small/large sizes
//   - static-address multi-stage workload
//   - data-dependent multi-stage workload
// ============================================================================

class TPrefetchAsyncTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(TPrefetchAsyncTest, case_float_4096_globaltensor)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<float, 4096>(0)));
}

TEST_F(TPrefetchAsyncTest, case_int32_4096_globaltensor)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<int32_t, 4096>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_1mb_prefetch_reduce_tload_latency)
{
    ASSERT_TRUE((RunTloadPerf<float, 262144>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_1mb_host_device_prefetch)
{
    ASSERT_TRUE((RunHostDevicePrefetch<float, 262144>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_128mb_host_device_prefetch)
{
    ASSERT_TRUE((RunHostDevicePrefetch<float, 33554432>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_16mb_static_address_prefetch)
{
    ASSERT_TRUE((RunStaticAddressPrefetch<float, 262144>(0, 16, 5000)));
}

TEST_F(TPrefetchAsyncTest, case_float_16mb_data_dependent_prefetch)
{
    ASSERT_TRUE((RunDataDependentPrefetch<float, 262144>(0, 16, 5000)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
