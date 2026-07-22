/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <gtest/gtest.h>

#include "tprefetch_async_kernel.h"

// Correctness coverage plus a device-side latency comparison between cold and
// prefetched TLOAD access.

class TPrefetchAsyncTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TPrefetchAsyncTest, case_float_4096_globaltensor) { ASSERT_TRUE((RunPrefetchAsyncCorrectness<float, 4096>(0))); }

TEST_F(TPrefetchAsyncTest, case_int32_4096_globaltensor)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<int32_t, 4096>(0)));
}

TEST_F(TPrefetchAsyncTest, case_float_4096_multi_post_wait_last)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<float, 4096>(0, 65U, false)));
}

TEST_F(TPrefetchAsyncTest, case_int32_4096_multi_post_wait_each)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<int32_t, 4096>(0, 16U, true)));
}

TEST_F(TPrefetchAsyncTest, case_float_4096_shared_external_session)
{
    ASSERT_TRUE((RunPrefetchAsyncCorrectness<float, 4096>(0, 16U, true, true)));
}

TEST_F(TPrefetchAsyncTest, case_float_4096_l2_access_benefit)
{
    double coldAverageUs = 0.0;
    double prefetchedAverageUs = 0.0;
    ASSERT_TRUE(RunPrefetchAsyncL2Benefit(0, coldAverageUs, prefetchedAverageUs));
    EXPECT_LT(prefetchedAverageUs, coldAverageUs);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
