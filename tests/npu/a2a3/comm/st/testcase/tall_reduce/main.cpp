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

#include "tall_reduce_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TALL_REDUCE Tests - Basic (data fits in single UB Tile)
// ============================================================================
TEST(TAllReduce, Int32_256_Sum_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunAllReduceInt32_256_Sum(2, 2, 0, 0)));
}
TEST(TAllReduce, Int32_256_Sum_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunAllReduceInt32_256_Sum(4, 4, 0, 0)));
}
TEST(TAllReduce, Int32_256_Sum_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunAllReduceInt32_256_Sum(8, 8, 0, 0)));
}
TEST(TAllReduce, Int32_256_Max_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunAllReduceInt32_256_Max(2, 2, 0, 0)));
}

// ============================================================================
// TALL_REDUCE Tests - Large Shape Chunked (GlobalTensor > UB Tile, auto-chunked)
// ============================================================================
// int32: 128x32, tile 16 rows → 8 chunks, Sum, 2 ranks
TEST(TAllReduce, LargeShape_Int32_128x32_tile16_Sum)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunAllReduceLargeShape_Int32_128x32_tile16_Sum(2, 2, 0, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
