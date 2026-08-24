/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "test_common.h"
#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

namespace {
constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;
constexpr size_t kScoresSize = kSeqLen * kSeqLen * sizeof(aclFloat16);
constexpr size_t kProbsSize = kSeqLen * kSeqLen * sizeof(aclFloat16);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}
} // namespace

void LaunchTFLASHATTNHalf(
    aclFloat16* out, aclFloat16* q, aclFloat16* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf, void* stream);
void LaunchTFLASHATTNS8(
    aclFloat16* out, int8_t* q, int8_t* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf, void* stream);
void LaunchTFLASHATTNS16(
    aclFloat16* out, int16_t* q, int16_t* kT, aclFloat16* v,
    aclFloat16* scoresBuf, aclFloat16* probsBuf,
    aclFloat16* qHalfBuf, aclFloat16* ktHalfBuf, void* stream);

class TFLASHATTNTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TFLASHATTNTest, case_half)
{
    constexpr size_t qSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t ktSize = kHeadDim * kSeqLen * sizeof(aclFloat16);
    constexpr size_t vSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t outSize = kSeqLen * kHeadDim * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *qHost, *ktHost, *vHost, *outHost;
    aclFloat16 *qDev, *ktDev, *vDev, *outDev, *scoresDev, *probsDev;

    size_t fileSize = 0;
    aclrtMallocHost((void**)&qHost, qSize);
    aclrtMallocHost((void**)&ktHost, ktSize);
    aclrtMallocHost((void**)&vHost, vSize);
    aclrtMallocHost((void**)&outHost, outSize);

    aclrtMalloc((void**)&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ktDev, ktSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&vDev, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDev, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scoresDev, kScoresSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&probsDev, kProbsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/q.bin", fileSize, qHost, qSize);
    aclrtMemcpy(qDev, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/kt.bin", fileSize, ktHost, ktSize);
    aclrtMemcpy(ktDev, ktSize, ktHost, ktSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/v.bin", fileSize, vHost, vSize);
    aclrtMemcpy(vDev, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTFLASHATTNHalf(outDev, qDev, ktDev, vDev, scoresDev, probsDev, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(outHost, outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, outSize);

    aclrtFree(qDev);
    aclrtFree(ktDev);
    aclrtFree(vDev);
    aclrtFree(outDev);
    aclrtFree(scoresDev);
    aclrtFree(probsDev);
    aclrtFreeHost(qHost);
    aclrtFreeHost(ktHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    vector<aclFloat16> golden(outSize / sizeof(aclFloat16));
    vector<aclFloat16> actual(outSize / sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, actual.data(), outSize);
    bool ret = ResultCmp(golden, actual, 0.005f);
    EXPECT_TRUE(ret);
}

TEST_F(TFLASHATTNTest, case_s8)
{
    constexpr size_t qSize = kSeqLen * kHeadDim * sizeof(int8_t);
    constexpr size_t ktSize = kHeadDim * kSeqLen * sizeof(int8_t);
    constexpr size_t vSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t outSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t scoresIntSize = kSeqLen * kSeqLen * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int8_t *qHost, *ktHost;
    aclFloat16 *vHost, *outHost;
    int8_t *qDev, *ktDev;
    aclFloat16 *vDev, *outDev, *scoresDev, *probsDev;

    size_t fileSize = 0;
    aclrtMallocHost((void**)&qHost, qSize);
    aclrtMallocHost((void**)&ktHost, ktSize);
    aclrtMallocHost((void**)&vHost, vSize);
    aclrtMallocHost((void**)&outHost, outSize);

    aclrtMalloc((void**)&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ktDev, ktSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&vDev, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDev, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scoresDev, scoresIntSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&probsDev, kProbsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/q.bin", fileSize, qHost, qSize);
    aclrtMemcpy(qDev, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/kt.bin", fileSize, ktHost, ktSize);
    aclrtMemcpy(ktDev, ktSize, ktHost, ktSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/v.bin", fileSize, vHost, vSize);
    aclrtMemcpy(vDev, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTFLASHATTNS8(outDev, qDev, ktDev, vDev, scoresDev, probsDev, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(outHost, outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, outSize);

    aclrtFree(qDev);
    aclrtFree(ktDev);
    aclrtFree(vDev);
    aclrtFree(outDev);
    aclrtFree(scoresDev);
    aclrtFree(probsDev);
    aclrtFreeHost(qHost);
    aclrtFreeHost(ktHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    vector<aclFloat16> golden(outSize / sizeof(aclFloat16));
    vector<aclFloat16> actual(outSize / sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, actual.data(), outSize);
    bool ret = ResultCmp(golden, actual, 0.01f);
    EXPECT_TRUE(ret);
}

TEST_F(TFLASHATTNTest, case_s16)
{
    constexpr size_t qSize = kSeqLen * kHeadDim * sizeof(int16_t);
    constexpr size_t ktSize = kHeadDim * kSeqLen * sizeof(int16_t);
    constexpr size_t vSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t outSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t qHalfSize = kSeqLen * kHeadDim * sizeof(aclFloat16);
    constexpr size_t ktHalfSize = kHeadDim * kSeqLen * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int16_t *qHost, *ktHost;
    aclFloat16 *vHost, *outHost;
    int16_t *qDev, *ktDev;
    aclFloat16 *vDev, *outDev, *scoresDev, *probsDev, *qHalfDev, *ktHalfDev;

    size_t fileSize = 0;
    aclrtMallocHost((void**)&qHost, qSize);
    aclrtMallocHost((void**)&ktHost, ktSize);
    aclrtMallocHost((void**)&vHost, vSize);
    aclrtMallocHost((void**)&outHost, outSize);

    aclrtMalloc((void**)&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ktDev, ktSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&vDev, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDev, outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scoresDev, kScoresSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&probsDev, kProbsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&qHalfDev, qHalfSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ktHalfDev, ktHalfSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/q.bin", fileSize, qHost, qSize);
    aclrtMemcpy(qDev, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/kt.bin", fileSize, ktHost, ktSize);
    aclrtMemcpy(ktDev, ktSize, ktHost, ktSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/v.bin", fileSize, vHost, vSize);
    aclrtMemcpy(vDev, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTFLASHATTNS16(outDev, qDev, ktDev, vDev, scoresDev, probsDev, qHalfDev, ktHalfDev, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(outHost, outSize, outDev, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, outSize);

    aclrtFree(qDev);
    aclrtFree(ktDev);
    aclrtFree(vDev);
    aclrtFree(outDev);
    aclrtFree(scoresDev);
    aclrtFree(probsDev);
    aclrtFree(qHalfDev);
    aclrtFree(ktHalfDev);
    aclrtFreeHost(qHost);
    aclrtFreeHost(ktHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    vector<aclFloat16> golden(outSize / sizeof(aclFloat16));
    vector<aclFloat16> actual(outSize / sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, actual.data(), outSize);
    bool ret = ResultCmp(golden, actual, 0.01f);
    EXPECT_TRUE(ret);
}
