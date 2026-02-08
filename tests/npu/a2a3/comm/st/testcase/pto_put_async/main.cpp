/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test cases for PTO_PUT_ASYNC (D2D async memory copy via SDMA/AIV)
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// ============================================================================
// Test function declarations (implemented in pto_put_async_kernel.cpp)
// ============================================================================

// SDMA path tests (using aclrtMemcpyAsync)
template <typename T, size_t count>
bool RunPutAsyncSdmaTest(int srcDeviceId, int dstDeviceId);

// AIV path tests (using AIV kernel calling TPUT_ASYNC_SDMA_IMPL)
template <typename T, size_t count, int AivCores>
bool RunPutAsyncAivTest(int srcDeviceId, int dstDeviceId);

// ============================================================================
// SDMA Path Tests (PTO_PUT_ASYNC with default template params)
// ============================================================================
TEST(PtoPutAsync_SDMA, Float256_D0toD1) { 
    ASSERT_TRUE((RunPutAsyncSdmaTest<float, 256>(0, 1))); 
}

TEST(PtoPutAsync_SDMA, Int32_4096_D0toD1) { 
    ASSERT_TRUE((RunPutAsyncSdmaTest<int32_t, 4096>(0, 1))); 
}

TEST(PtoPutAsync_SDMA, Uint8_512_D0toD1) { 
    ASSERT_TRUE((RunPutAsyncSdmaTest<uint8_t, 512>(0, 1))); 
}

// ============================================================================
// AIV Path Tests (PTO_PUT_ASYNC<false, AivCores>)
// ============================================================================
// TEST(PtoPutAsync_AIV, Float256_D0toD1) { 
//     ASSERT_TRUE((RunPutAsyncAivTest<float, 256, 20>(0, 1))); 
// }

// TEST(PtoPutAsync_AIV, Int32_1024_D0toD1) { 
//     ASSERT_TRUE((RunPutAsyncAivTest<int32_t, 1024, 10>(0, 1))); 
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
