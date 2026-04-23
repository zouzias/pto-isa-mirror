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
#include "../comm_mpi.h"

// All tprefetch_compare cases are single-card: only rank 0 runs the body; any
// extra ranks (should MPI be launched with np>1) early-return so test output
// isn't duplicated. The prefetch primitive is a single-device concern.
#define CMP_SINGLE_CARD_GUARD()                                                                                        \
    do {                                                                                                               \
        if (CommMpiRank() != 0) {                                                                                      \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

// ============================================================================
// Scenario A — end-to-end wall-clock latency (baseline vs host vs device)
// ============================================================================
TEST(TPrefetchCompare, A_EndToEnd_1MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 262144>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_16MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 4194304>(0)));
}

TEST(TPrefetchCompare, A_EndToEnd_128MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioAEndToEnd<float, 33554432>(0)));
}

// ============================================================================
// Scenario B — prefetch-issue overhead (tiny payload, host-us vs device-cycles)
// ============================================================================
TEST(TPrefetchCompare, B_IssueOverhead_1MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 262144>(0)));
}

TEST(TPrefetchCompare, B_IssueOverhead_16MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 4194304>(0)));
}

TEST(TPrefetchCompare, B_IssueOverhead_128MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioBIssueOverhead<float, 33554432>(0)));
}

// ============================================================================
// Scenario C — overlap with AI-Core compute (C0 / C1 / C2)
// ============================================================================
TEST(TPrefetchCompare, C_Overlap_1MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioCOverlap<float, 262144>(0)));
}

TEST(TPrefetchCompare, C_Overlap_16MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioCOverlap<float, 4194304>(0)));
}

TEST(TPrefetchCompare, C_Overlap_128MB)
{
    CMP_SINGLE_CARD_GUARD();
    ASSERT_TRUE((RunScenarioCOverlap<float, 33554432>(0)));
}

// ============================================================================
// Scenario D — cross-rank receiver-side prefetch (TPUT_ASYNC -> prefetch -> TLOAD)
//
// Requires mpirun -n 2. All ranks participate (no CMP_SINGLE_CARD_GUARD); the
// runner itself picks sender=0 / receiver=1 based on rankId. If launched with
// np != 2 the runner returns false and the test fails — the default filter in
// run_st.py guarantees this test is only selected under nranks=2.
// ============================================================================
TEST(TPrefetchCompare, D_CrossRank_1MB)
{
    ASSERT_TRUE((RunScenarioDCrossRank<float, 262144>(2, 0)));
}

TEST(TPrefetchCompare, D_CrossRank_16MB)
{
    ASSERT_TRUE((RunScenarioDCrossRank<float, 4194304>(2, 0)));
}

TEST(TPrefetchCompare, D_CrossRank_128MB)
{
    ASSERT_TRUE((RunScenarioDCrossRank<float, 33554432>(2, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
