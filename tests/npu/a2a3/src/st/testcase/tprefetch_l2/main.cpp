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
#include <iostream>
#include <gtest/gtest.h>

#include "tprefetch_l2_kernel.h"

// ============================================================================
// Single-card tests for pto::TPREFETCH_L2 (workspace API only).
// Cross-rank cases live under comm/st/testcase/tprefetch_l2/.
// ============================================================================

TEST(TPrefetchL2, Baseline_Float_4096)
{
    ASSERT_TRUE((RunBaseline<float, 4096>(0)));
}

TEST(TPrefetchL2, Baseline_Int32_4096)
{
    ASSERT_TRUE((RunBaseline<int32_t, 4096>(0)));
}

TEST(TPrefetchL2, Correctness_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2Correctness<float, 4096>(0)));
}

TEST(TPrefetchL2, Correctness_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2Correctness<int32_t, 4096>(0)));
}

TEST(TPrefetchL2, RawPtr_Float_4096)
{
    ASSERT_TRUE((RunPrefetchL2RawPtr<float, 4096>(0)));
}

TEST(TPrefetchL2, RawPtr_Int32_4096)
{
    ASSERT_TRUE((RunPrefetchL2RawPtr<int32_t, 4096>(0)));
}

// ============================================================================
// TLOAD latency — L2-cold vs L2-prefetched (single-card)
// ============================================================================
TEST(TPrefetchL2, Perf_Tload_Float_16KB)
{
    ASSERT_TRUE((RunTloadPerf<float, 4096>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_256KB)
{
    ASSERT_TRUE((RunTloadPerf<float, 65536>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_1MB)
{
    ASSERT_TRUE((RunTloadPerf<float, 262144>(0)));
}

TEST(TPrefetchL2, Perf_Tload_Float_4MB)
{
    ASSERT_TRUE((RunTloadPerf<float, 1048576>(0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
