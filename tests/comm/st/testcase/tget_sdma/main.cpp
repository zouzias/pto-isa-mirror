/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TGET_SDMA (asynchronous remote read using SDMA) operation via PTO (Shmem backend)
// Ring communication pattern: each rank reads data from next rank using SDMA engine

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// Declaration of 1D test functions implemented in tget_sdma_kernel.cpp
template <typename T, size_t count>
bool RunGetSdmaRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Declaration of 2D test functions implemented in tget_sdma_kernel.cpp
template <typename T, size_t rows, size_t cols>
bool RunGetSdmaRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tests - Direct GM to GM transfer without UB staging
// ============================================================================
TEST(TGet_sdma, Vec_FloatSmall) { ASSERT_TRUE((RunGetSdmaRing<float, 256>(2, 2, 0, 0))); }
TEST(TGet_sdma, Vec_Int32Large) { ASSERT_TRUE((RunGetSdmaRing<int32_t, 4096>(2, 2, 0, 0))); }
TEST(TGet_sdma, Vec_Uint8Small) { ASSERT_TRUE((RunGetSdmaRing<uint8_t, 512>(2, 2, 0, 0))); }

// ============================================================================
// 2D Shape Tests (GlobalTensor with 2D shape, direct SDMA transfer)
// ============================================================================
TEST(TGet_sdma, Shape2D_Float16x16) { ASSERT_TRUE((RunGetSdmaRing2D<float, 16, 16>(2, 2, 0, 0))); }
TEST(TGet_sdma, Shape2D_Float8x32) { ASSERT_TRUE((RunGetSdmaRing2D<float, 8, 32>(2, 2, 0, 0))); }
TEST(TGet_sdma, Shape2D_Int32_4x64) { ASSERT_TRUE((RunGetSdmaRing2D<int32_t, 4, 64>(2, 2, 0, 0))); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
