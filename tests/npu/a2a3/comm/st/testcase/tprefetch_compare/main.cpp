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

// ============================================================================
// Scenario D — cross-rank receiver-side prefetch (TPUT_ASYNC -> prefetch -> TLOAD)
//
// Single-card scenarios (A/B/C) live under
// tests/npu/a2a3/src/st/testcase/tprefetch_compare/.
//
// Requires mpirun -n 2. All ranks participate; the runner itself picks
// sender=0 / receiver=1 based on rankId.
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
