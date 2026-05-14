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

// ============================================================================
// Single-card prefetch comparison: Scenarios A / B / C.
// Cross-rank Scenario D lives under comm/st/testcase/tprefetch_compare/.
// ============================================================================

TEST(TPrefetchCompare, A_EndToEnd_1MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 262144>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_16MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 4194304>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_128MB)
{
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 33554432>(0)));
}

TEST(TPrefetchCompare, B_IssueOverhead_1MB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 262144>(0)));
}

TEST(TPrefetchCompare, B_IssueOverhead_16MB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0)));
}

TEST(TPrefetchCompare, B_IssueOverhead_128MB)
{
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 33554432>(0)));
}

TEST(TPrefetchCompare, C_Overlap_1MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 262144>(0)));
}

TEST(TPrefetchCompare, C_Overlap_16MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 4194304>(0)));
}

TEST(TPrefetchCompare, C_Overlap_128MB)
{
    ASSERT_TRUE((RunScenarioCOverlap<float, 33554432>(0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
