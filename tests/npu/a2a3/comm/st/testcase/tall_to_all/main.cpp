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

#include "tall_to_all_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TALL_TO_ALL Tests - Basic AllToAll (int32, sliceSize=256)
// ============================================================================
TEST(TAllToAll, Int32_Slice256_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunAllToAllInt32_256_2Ranks(2, 2, 0, 0)));
}
TEST(TAllToAll, Int32_Slice256_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunAllToAllInt32_256_4Ranks(4, 4, 0, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
