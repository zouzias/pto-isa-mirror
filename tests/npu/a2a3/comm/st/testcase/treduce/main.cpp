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

// Declared in treduce_kernel.cpp
bool RunReduceFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_4096_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_512_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_256_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
bool RunReduceInt32_256_Min(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// TREDUCE Tests - Reduce data from all ranks to root
// ============================================================================
TEST(TReduce, FloatSmall_Sum) { ASSERT_TRUE((RunReduceFloat256Sum(4, 4, 0, 0))); }
TEST(TReduce, FloatSmall_Sum_8Ranks) { ASSERT_TRUE((RunReduceFloat256Sum(8, 8, 0, 0))); }
TEST(TReduce, Int32Large_Sum) { ASSERT_TRUE((RunReduceInt32_4096_Sum(2, 2, 0, 0))); }
TEST(TReduce, Int32Large_Sum_8Ranks) { ASSERT_TRUE((RunReduceInt32_4096_Sum(8, 8, 0, 0))); }
TEST(TReduce, Int32Small_Sum) { ASSERT_TRUE((RunReduceInt32_512_Sum(2, 2, 0, 0))); }
TEST(TReduce, Int32Small_Sum_8Ranks) { ASSERT_TRUE((RunReduceInt32_512_Sum(8, 8, 0, 0))); }
TEST(TReduce, Int32Small_Max) { ASSERT_TRUE((RunReduceInt32_256_Max(2, 2, 0, 0))); }
TEST(TReduce, Int32Small_Max_8Ranks) { ASSERT_TRUE((RunReduceInt32_256_Max(8, 8, 0, 0))); }
TEST(TReduce, Int32Small_Min) { ASSERT_TRUE((RunReduceInt32_256_Min(2, 2, 0, 0))); }
TEST(TReduce, Int32Small_Min_8Ranks) { ASSERT_TRUE((RunReduceInt32_256_Min(8, 8, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
