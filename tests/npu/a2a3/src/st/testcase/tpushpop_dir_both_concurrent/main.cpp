/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopDirBoth(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    uint8_t* outCube, void* stream);

class TPushPopDirBothConcurrentTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int32_t key>
void TPushPopDirBothConcurrentTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(T);
    size_t bFileSize = M * K * sizeof(T);
    size_t dFileSize = K * N * sizeof(T);
    size_t fFileSize = M * N * sizeof(T);
    size_t outFileSize = M * N * sizeof(T);
    // A DIR_BOTH pipe is two rings in GM: C2V at offset 0 and V2C at SLOT_NUM * SLOT_SIZE.
    // SLOT_SIZE = M * N * sizeof(T) and SLOT_NUM = FIFO_DEPTH = 2 in the kernel, so the
    // buffer must be 2 * SLOT_NUM * SLOT_SIZE.
    size_t fifoFileSize = 4 * M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *outHost, *srcAHost, *srcBHost, *srcDHost, *srcFHost, *outCubeHost;
    uint8_t *outDevice, *srcADevice, *srcBDevice, *srcDDevice, *srcFDevice, *fifoMemDevice, *outCubeDevice;

    aclrtMallocHost((void**)(&outHost), outFileSize);
    aclrtMallocHost((void**)(&outCubeHost), outFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);
    aclrtMallocHost((void**)(&srcDHost), dFileSize);
    aclrtMallocHost((void**)(&srcFHost), fFileSize);

    aclrtMalloc((void**)&outDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outCubeDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDDevice, dFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcFDevice, fFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/srcA_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/srcB_gm.bin", bFileSize, srcBHost, bFileSize);
    ReadFile(GetGoldenDir() + "/srcD_gm.bin", dFileSize, srcDHost, dFileSize);
    ReadFile(GetGoldenDir() + "/srcF_gm.bin", fFileSize, srcFHost, fFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDDevice, dFileSize, srcDHost, dFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcFDevice, fFileSize, srcFHost, fFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopDirBoth<key>(
        (uint8_t*)ffts, outDevice, srcADevice, srcBDevice, srcDDevice, srcFDevice, fifoMemDevice, outCubeDevice,
        stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outFileSize, outDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(outCubeHost, outFileSize, outCubeDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, outFileSize);
    WriteFile(GetGoldenDir() + "/output_cube.bin", outCubeHost, outFileSize);

    aclrtFree(outDevice);
    aclrtFree(outCubeDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcDDevice);
    aclrtFree(srcFDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(outHost);
    aclrtFreeHost(outCubeHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcDHost);
    aclrtFreeHost(srcFHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outFileSize);
    std::vector<T> devFinal(outFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outFileSize, golden.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret) << "vector side (C2V) mismatch";

    // Check the cube's V2C read as well, not just the vector's C2V read. The two payloads
    // are different sizes -- C2V is a full slot, V2C is half of one -- so a push that lands
    // second overwrites only part of the other tile and a single-sided check can still come
    // back clean. On an affected build both comparisons fail.
    std::vector<T> goldenCube(outFileSize);
    std::vector<T> devCube(outFileSize);
    ReadFile(GetGoldenDir() + "/goldenCube.bin", outFileSize, goldenCube.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_cube.bin", outFileSize, devCube.data(), outFileSize);

    bool retCube = ResultCmp(goldenCube, devCube, 0.001f);
    EXPECT_TRUE(retCube) << "cube side (V2C) mismatch -- C2V and V2C aliased the same GM slot";
}

TEST_F(TPushPopDirBothConcurrentTest, case1_float_dir_both_concurrent)
{
    TPushPopDirBothConcurrentTestFunc<float, 1>(128, 64, 128);
}

TEST_F(TPushPopDirBothConcurrentTest, case2_float_dir_both_concurrent_left_right)
{
    TPushPopDirBothConcurrentTestFunc<float, 2>(128, 64, 128);
}
