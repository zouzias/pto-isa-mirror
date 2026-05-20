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

template <typename T>
void AllocateAndLoadData(T* &host, T* &device, size_t size, const std::string& filename)
{
    aclrtMallocHost((void **)(&host), size);
    aclrtMalloc((void **)&device, size, ACL_MEM_MALLOC_HUGE_FIRST);
    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(filename, readSize, host, size));
    aclrtMemcpy(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void RunKernelAndGetResults(
    aclrtStream stream,
    float* dstValDevice, float* src0ValDevice, float* src1ValDevice,
    uint32_t* dstIdxDevice, uint32_t* src0IdxDevice, uint32_t* src1IdxDevice,
    float* dstValHost, uint32_t* dstIdxHost,
    size_t valSize, size_t idxSize)
{
    LaunchTPARTARGMAX<kRows, kCols, kValidRows1, kValidCols1>(dstValDevice, src0ValDevice, src1ValDevice, dstIdxDevice, src0IdxDevice, 
                                                              src1IdxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstValHost, valSize, dstValDevice, valSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstIdxHost, idxSize, dstIdxDevice, idxSize, ACL_MEMCPY_DEVICE_TO_HOST);
}

bool VerifyResults(const std::string& goldenDir, size_t valSize, size_t idxSize)
{
    std::vector<float> golden_val(kRows * kCols);
    std::vector<float> out_val(kRows * kCols);
    std::vector<uint32_t> golden_idx(kRows * kCols);
    std::vector<uint32_t> out_idx(kRows * kCols);
    
    size_t readSize = 0;
    
    ReadFile(goldenDir + "/golden_val.bin", readSize, golden_val.data(), valSize);
    ReadFile(goldenDir + "/output_val.bin", readSize, out_val.data(), valSize);
    ReadFile(goldenDir + "/golden_idx.bin", readSize, golden_idx.data(), idxSize);
    ReadFile(goldenDir + "/output_idx.bin", readSize, out_idx.data(), idxSize);
    
    return ResultCmp<float>(golden_val, out_val, kEpsilon) && ResultCmp<uint32_t>(golden_idx, out_idx, 0.0f);
}

TEST_F(TPARTARGMAX_Test, case_float_64x64_src1_32x32)
{
    const size_t valSize = static_cast<size_t>(kRows) * kCols * sizeof(float);
    const size_t idxSize = static_cast<size_t>(kRows) * kCols * sizeof(uint32_t);
    const std::string goldenDir = GetGoldenDir();
    
    float *dstValHost = nullptr, *src0ValHost = nullptr, *src1ValHost = nullptr;
    float *dstValDevice = nullptr, *src0ValDevice = nullptr, *src1ValDevice = nullptr;
    uint32_t *dstIdxHost = nullptr, *src0IdxHost = nullptr, *src1IdxHost = nullptr;
    uint32_t *dstIdxDevice = nullptr, *src0IdxDevice = nullptr, *src1IdxDevice = nullptr;
    
    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    AllocateAndLoadData(src0ValHost, src0ValDevice, valSize, goldenDir + "/input0_val.bin");
    AllocateAndLoadData(src1ValHost, src1ValDevice, valSize, goldenDir + "/input1_val.bin");
    AllocateAndLoadData(src0IdxHost, src0IdxDevice, idxSize, goldenDir + "/input0_idx.bin");
    AllocateAndLoadData(src1IdxHost, src1IdxDevice, idxSize, goldenDir + "/input1_idx.bin");
    aclrtMallocHost((void **)(&dstValHost), valSize);
    aclrtMalloc((void **)&dstValDevice, valSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)(&dstIdxHost), idxSize);
    aclrtMalloc((void **)&dstIdxDevice, idxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    RunKernelAndGetResults(stream, dstValDevice, src0ValDevice, src1ValDevice, dstIdxDevice, src0IdxDevice, src1IdxDevice,
                           dstValHost, dstIdxHost, valSize, idxSize);
    
    WriteFile(goldenDir + "/output_val.bin", dstValHost, valSize);
    WriteFile(goldenDir + "/output_idx.bin", dstIdxHost, idxSize);

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

    EXPECT_TRUE(VerifyResults(goldenDir, valSize, idxSize));
}
