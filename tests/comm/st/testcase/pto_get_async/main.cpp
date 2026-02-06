/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test cases for PTO_GET_ASYNC (D2D async memory copy via SDMA/AIV - GET operation)
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// ============================================================================
// Test function declarations (implemented in pto_get_async_kernel.cpp)
// ============================================================================

// SDMA path tests (using aclrtMemcpyAsync)
template <typename T, size_t count>
bool RunGetAsyncSdmaTest(int localDeviceId, int remoteDeviceId);

// AIV path tests (using AIV kernel calling TGET_ASYNC_SDMA_IMPL)
template <typename T, size_t count, int AivCores>
bool RunGetAsyncAivTest(int localDeviceId, int remoteDeviceId);

// ============================================================================
// SDMA Path Tests (PTO_GET_ASYNC with default template params)
// GET: Read from remote device (D1) to local device (D0)
// ============================================================================
TEST(PtoGetAsync_SDMA, Float256_D1toD0) { 
    ASSERT_TRUE((RunGetAsyncSdmaTest<float, 256>(0, 1))); 
}

TEST(PtoGetAsync_SDMA, Int32_4096_D1toD0) { 
    ASSERT_TRUE((RunGetAsyncSdmaTest<int32_t, 4096>(0, 1))); 
}

TEST(PtoGetAsync_SDMA, Uint8_512_D1toD0) { 
    ASSERT_TRUE((RunGetAsyncSdmaTest<uint8_t, 512>(0, 1))); 
}

// ============================================================================
// AIV Path Tests (PTO_GET_ASYNC<false, AivCores>)
// ============================================================================
TEST(PtoGetAsync_AIV, Float256_D1toD0) { 
    ASSERT_TRUE((RunGetAsyncAivTest<float, 256, 20>(0, 1))); 
}

TEST(PtoGetAsync_AIV, Int32_1024_D1toD0) { 
    ASSERT_TRUE((RunGetAsyncAivTest<int32_t, 1024, 10>(0, 1))); 
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
