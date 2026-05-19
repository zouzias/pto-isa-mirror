/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "tall_gather_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TALL_GATHER Tests — Basic (per-rank data fits in a single UB Tile)
// ============================================================================
TEST(TAllGather, Int32_256_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunAllGatherInt32_256(2, 2, 0, 0)));
}
TEST(TAllGather, Int32_256_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunAllGatherInt32_256(4, 4, 0, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
