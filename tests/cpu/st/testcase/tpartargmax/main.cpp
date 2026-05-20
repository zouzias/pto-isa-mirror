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

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTARGMAX(float *outVal, float *src0Val, float *src1Val, uint32_t *outIdx, uint32_t *src0Idx, uint32_t *src1Idx, void *stream);

class TPARTARGMAX_Test : public testing::Test {
};

namespace {

constexpr int kDeviceId = 0;
constexpr float kEpsilon = 0.0f;
constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kValidRows1 = 32;
constexpr int kValidCols1 = 32;

} // namespace

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

TEST_F(TPARTARGMAX_Test, case_float_64x64_src1_32x32)
{
    const size_t fileSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);
    const size_t idxFileSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstValHost, *src0ValHost, *src1ValHost;
    float *dstValDevice, *src0ValDevice, *src1ValDevice;
    uint32_t *dstIdxHost, *src0IdxHost, *src1IdxHost;
    uint32_t *dstIdxDevice, *src0IdxDevice, *src1IdxDevice;
    aclrtMallocHost((void **)(&dstValHost), fileSize);
    aclrtMallocHost((void **)(&src0ValHost), fileSize);
    aclrtMallocHost((void **)(&src1ValHost), fileSize);
    aclrtMallocHost((void **)(&dstIdxHost), idxFileSize);
    aclrtMallocHost((void **)(&src0IdxHost), idxFileSize);
    aclrtMallocHost((void **)(&src1IdxHost), idxFileSize);
    aclrtMalloc((void **)&dstValDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0ValDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1ValDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstIdxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0IdxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1IdxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input0_val.bin", readSize, src0ValHost, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1_val.bin", readSize, src1ValHost, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input0_idx.bin", readSize, src0IdxHost, idxFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1_idx.bin", readSize, src1IdxHost, idxFileSize));
    aclrtMemcpy(src0ValDevice, fileSize, src0ValHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1ValDevice, fileSize, src1ValHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0IdxDevice, idxFileSize, src0IdxHost, idxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1IdxDevice, idxFileSize, src1IdxHost, idxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPARTARGMAX<kRows, kCols, kValidRows1, kValidCols1>(dstValDevice, src0ValDevice, src1ValDevice, dstIdxDevice, src0IdxDevice, src1IdxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstValHost, fileSize, dstValDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstIdxHost, idxFileSize, dstIdxDevice, idxFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_val.bin", dstValHost, fileSize);
    WriteFile(GetGoldenDir() + "/output_idx.bin", dstIdxHost, idxFileSize);

    aclrtFree(dstValDevice);
    aclrtFree(src0ValDevice);
    aclrtFree(src1ValDevice);
    aclrtFree(dstIdxDevice);
    aclrtFree(src0IdxDevice);
    aclrtFree(src1IdxDevice);
    aclrtFreeHost(dstValHost);
    aclrtFreeHost(src0ValHost);
    aclrtFreeHost(src1ValHost);
    aclrtFreeHost(dstIdxHost);
    aclrtFreeHost(src0IdxHost);
    aclrtFreeHost(src1IdxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    std::vector<float> golden_val(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<float> out_val(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<uint32_t> golden_idx(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<uint32_t> out_idx(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden_val.bin", readSize, golden_val.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_val.bin", readSize, out_val.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden_idx.bin", readSize, golden_idx.data(), idxFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_idx.bin", readSize, out_idx.data(), idxFileSize));
    EXPECT_TRUE(ResultCmp<float>(golden_val, out_val.data(), kEpsilon));
    EXPECT_TRUE(ResultCmp<uint32_t>(golden_idx, out_idx.data(), 0.0f));
}
