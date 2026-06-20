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
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace PtoTestCommon;

class SYNCALLTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    std::filesystem::create_directories(fullPath);
    return fullPath;
}

void LaunchSoftSyncAll(int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t totalBlocks, void *stream);
void LaunchHardSyncAll(int32_t *out, int32_t *flags, int32_t totalBlocks, void *stream);
void LaunchSoftSyncAllMix11(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);
void LaunchSoftSyncAllMix12(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream);
void LaunchSoftSyncDomainMix12(int32_t *out, int32_t *syncWorkspace, void *stream);
void LaunchHardSyncAllAIC(int32_t *out, void *stream);
void LaunchVcSoftAtomicVecProducer(int32_t *payload, int32_t *syncState, int32_t vecParticipants, int32_t seq,
                                   void *stream);
void LaunchVcSoftAtomicCubeWaitConsumer(int32_t *out, int32_t *payload, int32_t *syncState, int32_t aicBlocks,
                                        int32_t vecParticipants, int32_t seq, void *stream);

#define EXPECT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

TEST_F(SYNCALLTest, case_soft_aiv_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAll(outDevice, flagsDevice, syncWorkspaceDevice, blockCount, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("soft out[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft flags[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_hard_aiv_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchHardSyncAll(outDevice, flagsDevice, blockCount, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("hard out[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_mix_1_2_all_blocks)
{
    constexpr int32_t blockCount = 54;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("soft_mix_1_2 out[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft_mix_1_2 flags[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_sync_domain_mix_1_2)
{
    constexpr int32_t blockCount = 54;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t outElementCount = blockCount * int32PerCacheLine;
    constexpr size_t workspaceElementCount = 2048;
    constexpr size_t outByteSize = outElementCount * sizeof(int32_t);
    constexpr size_t workspaceByteSize = workspaceElementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *workspaceHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *workspaceDevice = nullptr;
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), outByteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&workspaceHost), workspaceByteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), outByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(
        aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, outElementCount, 0);
    std::fill_n(workspaceHost, workspaceElementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, outByteSize, outHost, outByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(
        aclrtMemcpy(workspaceDevice, workspaceByteSize, workspaceHost, workspaceByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncDomainMix12(outDevice, workspaceDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize));

    bool ret = true;
    for (int32_t i = 0; i < blockCount; ++i) {
        int32_t expect = (i < 18) ? 1 : 10;
        if (i < 3) {
            expect |= 4;
        }
        const int32_t actual = outHost[i * int32PerCacheLine];
        ret &= actual == expect;
        if (actual != expect) {
            std::printf("domain status[%d]=%d expect=%d\n", i, actual, expect);
        }
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(workspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(workspaceHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    constexpr int32_t blockCount = 36;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;
    int32_t *syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAllMix11(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    if (!ret) {
        std::printf("soft_mix_1_1 out[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft_mix_1_1 flags[0..7]:");
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t byteSize = blockCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outDevice = nullptr;
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    LaunchHardSyncAllAIC(outDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_vc_soft_atomic_vec_all_done_then_cube_1_2_seven_blocks)
{
    constexpr int32_t aicBlocks = 7;
    constexpr int32_t vecParticipants = aicBlocks * 2;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t outElementCount = aicBlocks * int32PerCacheLine;
    constexpr size_t payloadElementCount = vecParticipants * int32PerCacheLine;
    constexpr size_t syncStateElementCount = 2 * int32PerCacheLine;
    constexpr size_t outByteSize = outElementCount * sizeof(int32_t);
    constexpr size_t payloadByteSize = payloadElementCount * sizeof(int32_t);
    constexpr size_t syncStateByteSize = syncStateElementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *payloadHost = nullptr;
    int32_t *syncStateHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *payloadDevice = nullptr;
    int32_t *syncStateDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), outByteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&payloadHost), payloadByteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&syncStateHost), syncStateByteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), outByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&payloadDevice), payloadByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(
        aclrtMalloc(reinterpret_cast<void **>(&syncStateDevice), syncStateByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, outElementCount, 0);
    std::fill_n(payloadHost, payloadElementCount, 0);
    std::fill_n(syncStateHost, syncStateElementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, outByteSize, outHost, outByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(payloadDevice, payloadByteSize, payloadHost, payloadByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(
        aclrtMemcpy(syncStateDevice, syncStateByteSize, syncStateHost, syncStateByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    auto runRound = [&](int32_t seq) {
        std::fill_n(outHost, outElementCount, 0);
        EXPECT_ACL_OK(aclrtMemcpy(outDevice, outByteSize, outHost, outByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

        LaunchVcSoftAtomicVecProducer(payloadDevice, syncStateDevice, vecParticipants, seq, stream);
        EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
        LaunchVcSoftAtomicCubeWaitConsumer(outDevice, payloadDevice, syncStateDevice, aicBlocks, vecParticipants, seq,
                                           stream);
        EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    };

    runRound(1);
    runRound(2);

    EXPECT_ACL_OK(aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(payloadHost, payloadByteSize, payloadDevice, payloadByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(
        aclrtMemcpy(syncStateHost, syncStateByteSize, syncStateDevice, syncStateByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize));

    std::vector<int32_t> golden(aicBlocks);
    std::vector<int32_t> devFinal(aicBlocks);
    for (int32_t i = 0; i < aicBlocks; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.0f);
    for (int32_t i = 0; i < vecParticipants; ++i) {
        ret &= payloadHost[i * int32PerCacheLine] == 200 + i;
    }
    ret &= syncStateHost[0] == 2 * vecParticipants;
    ret &= syncStateHost[int32PerCacheLine] == 2;

    if (!ret) {
        std::printf("vc_atomic out:");
        for (int32_t i = 0; i < aicBlocks; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nvc_atomic payload:");
        for (int32_t i = 0; i < vecParticipants; ++i) {
            std::printf(" %d", payloadHost[i * int32PerCacheLine]);
        }
        std::printf("\nvc_atomic counter=%d doorbell=%d\n", syncStateHost[0], syncStateHost[int32PerCacheLine]);
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(payloadDevice));
    EXPECT_ACL_OK(aclrtFree(syncStateDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(payloadHost));
    EXPECT_ACL_OK(aclrtFreeHost(syncStateHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}
