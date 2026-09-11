/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace PtoTestCommon;

template <typename T, int kRows, int kCols>
void LaunchTROWEXPAND(T* out, T* src, void* stream);

class TROWEXPAND_Test : public testing::Test {};

static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

static void setup_stream(aclrtStream& stream)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&stream);
}

static void teardown_stream(aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <typename T, int kRows, int kCols>
void test_trowexpand()
{
    constexpr size_t elementCount = static_cast<size_t>(kRows) * static_cast<size_t>(kCols);
    const size_t size = elementCount * sizeof(T);

    aclrtStream stream;
    setup_stream(stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;
    aclrtMallocHost((void**)(&dstHost), size);
    aclrtMallocHost((void**)(&srcHost), size);
    aclrtMalloc((void**)&dstDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, size));
    aclrtMemcpy(srcDevice, size, srcHost, size, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTROWEXPAND<T, kRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, size, dstDevice, size, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, size);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    teardown_stream(stream);

    std::vector<T> golden(elementCount);
    std::vector<T> out(elementCount);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), size));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), size));
    EXPECT_TRUE(ResultCmp<T>(golden, out.data(), 0.0f));
}

TEST_F(TROWEXPAND_Test, case_expand_float_64x64) { test_trowexpand<float, 64, 64>(); }
TEST_F(TROWEXPAND_Test, case_expand_int64_64x64) { test_trowexpand<int64_t, 64, 64>(); }
TEST_F(TROWEXPAND_Test, case_expand_uint64_64x64) { test_trowexpand<uint64_t, 64, 64>(); }
