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
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    std::filesystem::create_directories(fullPath);
    return fullPath;
}

void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t totalBlocks, void* stream);
void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchHardSyncAll(int32_t* out, int32_t* flags, int32_t totalBlocks, void* stream);
void LaunchSoftSyncAllMix11(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t* marker, void* stream);
void LaunchHardSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchHardSyncAllAIC(int32_t* out, void* stream);

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

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

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

// AIC-only soft SYNCALL: all cube cores publish a flag via scalar GM store, run
// the AIC-only atomic-counter barrier, then scalar-read every flag. out[idx]==1
// proves this cube core saw all peers' round-1 and round-2 writes across barriers.
TEST_F(SYNCALLTest, case_soft_aic_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    // 16 int32 = one 64-byte A5 cache line per core slot; must match
    // kAicSoftCacheLine in syncall_aic_soft_kernel.cpp (avoids cube dcci false sharing).
    constexpr size_t int32PerCacheLine = 16;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, byteSize, 0, byteSize));

    LaunchSoftSyncAllAIC(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    std::printf("[soft_aic_only] aclrtSynchronizeStream ret=%d (507015=AICORE exception)\n", syncRet);
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "AIC-only soft barrier faulted";

    if (syncRet == ACL_SUCCESS) {
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
            std::printf("soft_aic out[0..7]:");
            for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
                std::printf(" %d", outHost[i * int32PerCacheLine]);
            }
            std::printf("\n");
        }
        EXPECT_TRUE(ret);
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
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

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

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
    // Must match pto::SYNCALL_SOFT_MIX_SLOT_INT32: the soft MIX barrier uses one
    // isolated per-core slot at this stride.
    constexpr size_t mixSlotInt32 = 32;
    constexpr size_t syncWsBytes = blockCount * mixSlotInt32 * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    constexpr int32_t aicBlocks = 18;
    const size_t markerByteSize = blockCount * int32PerCacheLine * sizeof(int32_t);

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* markerHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;
    int32_t* markerDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&markerHost), markerByteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), syncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&markerDevice), markerByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    std::fill_n(markerHost, blockCount * int32PerCacheLine, -1);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, syncWsBytes, 0, syncWsBytes));
    EXPECT_ACL_OK(aclrtMemcpy(markerDevice, markerByteSize, markerHost, markerByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, markerDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    std::printf("[soft_mix_1_2] aclrtSynchronizeStream ret=%d (507015=AICORE exception)\n", syncRet);

    if (aclrtMemcpy(markerHost, markerByteSize, markerDevice, markerByteSize, ACL_MEMCPY_DEVICE_TO_HOST) ==
        ACL_SUCCESS) {
        std::printf("[soft_mix_1_2] stage markers (-1=never started, 8=completed):\n AIC:");
        for (int32_t i = 0; i < aicBlocks; ++i) {
            std::printf(" %d", markerHost[i * int32PerCacheLine]);
        }
        std::printf("\n AIV:");
        for (int32_t i = aicBlocks; i < blockCount; ++i) {
            std::printf(" %d", markerHost[i * int32PerCacheLine]);
        }
        std::printf(
            "\n[soft_mix_1_2] stages: 0=enter 1=proxyWr1 2=barrier1 3=check1 4=barrier2 5=proxyWr2 "
            "6=barrier3 7=check2 8=outWr(done); crash is at the step AFTER the max marker\n");
    }
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(blockCount);
        std::vector<int32_t> devFinal(blockCount);
        for (size_t i = 0; i < blockCount; ++i) {
            golden[i] = 1;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }
        EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFree(markerDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtFreeHost(markerHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    constexpr int32_t blockCount = 36;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);
    // Must match pto::SYNCALL_SOFT_MIX_SLOT_INT32 (per-core isolated slot stride).
    constexpr size_t mixSlotInt32 = 32;
    constexpr size_t syncWsBytes = blockCount * mixSlotInt32 * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), syncWsBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemset(syncWorkspaceDevice, syncWsBytes, 0, syncWsBytes));

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

TEST_F(SYNCALLTest, case_hard_mix_1_2_all_blocks)
{
    constexpr int32_t blockCount = 54; // 18 cube + 36 vector (1:2), auto-split chevron.
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outHost = nullptr;
    int32_t* flagsHost = nullptr;
    int32_t* outDevice = nullptr;
    int32_t* flagsDevice = nullptr;
    int32_t* syncWorkspaceDevice = nullptr; // unused by hard barrier, kept for the shared body signature.

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&outHost), byteSize));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&flagsHost), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchHardSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    std::printf("[hard_mix_1_2] aclrtSynchronizeStream ret=%d (507015=AICORE exception)\n", syncRet);
    EXPECT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed";

    if (syncRet == ACL_SUCCESS) {
        EXPECT_ACL_OK(aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        EXPECT_ACL_OK(aclrtMemcpy(flagsHost, byteSize, flagsDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, byteSize));

        std::vector<int32_t> golden(blockCount);
        std::vector<int32_t> devFinal(blockCount);
        for (size_t i = 0; i < blockCount; ++i) {
            golden[i] = 1;
            devFinal[i] = outHost[i * int32PerCacheLine];
        }
        EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal, 0.0f));
    }

    (void)aclrtFree(outDevice);
    (void)aclrtFree(flagsDevice);
    (void)aclrtFree(syncWorkspaceDevice);
    (void)aclrtFreeHost(outHost);
    (void)aclrtFreeHost(flagsHost);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t byteSize = blockCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* outDevice = nullptr;
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    LaunchHardSyncAllAIC(outDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}