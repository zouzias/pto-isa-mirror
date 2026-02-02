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

// Declared in tscatter_kernel.cpp
template <typename T, size_t count>
bool RunScatter(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// TSCATTER Tests - Root scatters data to all ranks
// ============================================================================
TEST(TScatter, FloatSmall) { ASSERT_TRUE((RunScatter<float, 256>(4, 4, 0, 0))); }
TEST(TScatter, Int32Large) { ASSERT_TRUE((RunScatter<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TScatter, Uint8Small) { ASSERT_TRUE((RunScatter<uint8_t, 512>(2, 2, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
