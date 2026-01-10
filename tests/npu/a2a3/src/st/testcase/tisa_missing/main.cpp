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
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace PtoTestCommon;

void LaunchTisaMissingIntOps(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
void LaunchTisaMissingFloatOps(float *out, float *src0, float *src1, float *src2, void *stream);
void LaunchTisaMissingMGatherMScatter(int32_t *out, int32_t *memSrc, int32_t *idx, int32_t *scatterSrc,
    int32_t *memDstInit, void *stream);

class TISAMISSINGTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T>
static bool ExactEqual(const std::vector<T> &exp, const std::vector<T> &act, size_t &firstBad)
{
    if (exp.size() != act.size()) {
        firstBad = 0;
        return false;
    }
    for (size_t i = 0; i < exp.size(); ++i) {
        if (exp[i] != act[i]) {
            firstBad = i;
            return false;
        }
    }
    firstBad = 0;
    return true;
}

static bool AllClose(const std::vector<float> &exp, const std::vector<float> &act, float eps, size_t &firstBad)
{
    if (exp.size() != act.size()) {
        firstBad = 0;
        return false;
    }
    for (size_t i = 0; i < exp.size(); ++i) {
        const float e = exp[i];
        const float a = act[i];
        const float diff = std::fabs(e - a);
        const float rel = (std::fabs(e) > 1e-12f) ? (diff / std::fabs(e)) : diff;
        if (diff > eps && rel > eps) {
            firstBad = i;
            return false;
        }
    }
    firstBad = 0;
    return true;
}

TEST_F(TISAMISSINGTest, case_int32_vec_ops)
{
    constexpr size_t kRows = 64;
    constexpr size_t kCols = 64;
    constexpr size_t kNumOut = 11;
    const size_t tileElems = kRows * kCols;
    const size_t tileBytes = tileElems * sizeof(int32_t);
    const size_t outBytes = kNumOut * tileBytes;

    aclInit(nullptr);
    aclrtSetDevice(GetDeviceId());
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *outHost = nullptr;
    int32_t *src0Host = nullptr;
    int32_t *src1Host = nullptr;

    int32_t *outDev = nullptr;
    int32_t *src0Dev = nullptr;
    int32_t *src1Dev = nullptr;

    aclrtMallocHost((void **)(&outHost), outBytes);
    aclrtMallocHost((void **)(&src0Host), tileBytes);
    aclrtMallocHost((void **)(&src1Host), tileBytes);

    aclrtMalloc((void **)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Dev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Dev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, tileBytes);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, tileBytes);

    aclrtMemcpy(src0Dev, tileBytes, src0Host, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Dev, tileBytes, src1Host, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTisaMissingIntOps(outDev, src0Dev, src1Dev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(src0Dev);
    aclrtFree(src1Dev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(GetDeviceId());
    aclFinalize();

    std::vector<int32_t> golden(outBytes / sizeof(int32_t));
    std::vector<int32_t> devFinal(outBytes / sizeof(int32_t));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outBytes);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), outBytes);
    size_t firstBad = 0;
    const bool ok = ExactEqual(golden, devFinal, firstBad);
    if (!ok) {
        std::cout << "first mismatch idx=" << firstBad << " exp=" << golden[firstBad] << " act=" << devFinal[firstBad]
                  << std::endl;
    }
    EXPECT_TRUE(ok);
}

TEST_F(TISAMISSINGTest, case_float32_vec_ops)
{
    constexpr size_t kRows = 64;
    constexpr size_t kCols = 64;
    constexpr size_t kNumOut = 12;
    const size_t tileElems = kRows * kCols;
    const size_t tileBytes = tileElems * sizeof(float);
    const size_t outBytes = kNumOut * tileBytes;

    aclInit(nullptr);
    aclrtSetDevice(GetDeviceId());
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *outHost = nullptr;
    float *src0Host = nullptr;
    float *src1Host = nullptr;
    float *src2Host = nullptr;

    float *outDev = nullptr;
    float *src0Dev = nullptr;
    float *src1Dev = nullptr;
    float *src2Dev = nullptr;

    aclrtMallocHost((void **)(&outHost), outBytes);
    aclrtMallocHost((void **)(&src0Host), tileBytes);
    aclrtMallocHost((void **)(&src1Host), tileBytes);
    aclrtMallocHost((void **)(&src2Host), tileBytes);

    aclrtMalloc((void **)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Dev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Dev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Dev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, tileBytes);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, tileBytes);
    ReadFile(GetGoldenDir() + "/input3.bin", fileSize, src2Host, tileBytes);

    aclrtMemcpy(src0Dev, tileBytes, src0Host, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Dev, tileBytes, src1Host, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Dev, tileBytes, src2Host, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTisaMissingFloatOps(outDev, src0Dev, src1Dev, src2Dev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(src0Dev);
    aclrtFree(src1Dev);
    aclrtFree(src2Dev);
    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(GetDeviceId());
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outBytes);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), outBytes);
    size_t firstBad = 0;
    const bool ok = AllClose(golden, devFinal, 0.002f, firstBad);
    if (!ok) {
        std::cout << "first mismatch idx=" << firstBad << " exp=" << golden[firstBad] << " act=" << devFinal[firstBad]
                  << std::endl;
    }
    EXPECT_TRUE(ok);
}

TEST_F(TISAMISSINGTest, case_mgather_mscatter_int32)
{
    constexpr size_t kRows = 64;
    constexpr size_t kCols = 64;
    constexpr size_t kNumOut = 2;
    const size_t tileElems = kRows * kCols;
    const size_t tileBytes = tileElems * sizeof(int32_t);
    const size_t outBytes = kNumOut * tileBytes;

    aclInit(nullptr);
    aclrtSetDevice(GetDeviceId());
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *outHost = nullptr;
    int32_t *memSrcHost = nullptr;
    int32_t *idxHost = nullptr;
    int32_t *scatterSrcHost = nullptr;
    int32_t *memDstInitHost = nullptr;

    int32_t *outDev = nullptr;
    int32_t *memSrcDev = nullptr;
    int32_t *idxDev = nullptr;
    int32_t *scatterSrcDev = nullptr;
    int32_t *memDstInitDev = nullptr;

    aclrtMallocHost((void **)(&outHost), outBytes);
    aclrtMallocHost((void **)(&memSrcHost), tileBytes);
    aclrtMallocHost((void **)(&idxHost), tileBytes);
    aclrtMallocHost((void **)(&scatterSrcHost), tileBytes);
    aclrtMallocHost((void **)(&memDstInitHost), tileBytes);

    aclrtMalloc((void **)&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&memSrcDev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scatterSrcDev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&memDstInitDev, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, memSrcHost, tileBytes);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, idxHost, tileBytes);
    ReadFile(GetGoldenDir() + "/input3.bin", fileSize, scatterSrcHost, tileBytes);
    ReadFile(GetGoldenDir() + "/input4.bin", fileSize, memDstInitHost, tileBytes);

    aclrtMemcpy(memSrcDev, tileBytes, memSrcHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDev, tileBytes, idxHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scatterSrcDev, tileBytes, scatterSrcHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(memDstInitDev, tileBytes, memDstInitHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTisaMissingMGatherMScatter(outDev, memSrcDev, idxDev, scatterSrcDev, memDstInitDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes);

    aclrtFree(outDev);
    aclrtFree(memSrcDev);
    aclrtFree(idxDev);
    aclrtFree(scatterSrcDev);
    aclrtFree(memDstInitDev);

    aclrtFreeHost(outHost);
    aclrtFreeHost(memSrcHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(scatterSrcHost);
    aclrtFreeHost(memDstInitHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(GetDeviceId());
    aclFinalize();

    std::vector<int32_t> golden(outBytes / sizeof(int32_t));
    std::vector<int32_t> devFinal(outBytes / sizeof(int32_t));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), outBytes);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), outBytes);
    size_t firstBad = 0;
    const bool ok = ExactEqual(golden, devFinal, firstBad);
    if (!ok) {
        std::cout << "first mismatch idx=" << firstBad << " exp=" << golden[firstBad] << " act=" << devFinal[firstBad]
                  << std::endl;
    }
    EXPECT_TRUE(ok);
}
