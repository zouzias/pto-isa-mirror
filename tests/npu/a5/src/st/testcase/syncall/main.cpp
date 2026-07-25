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
void LaunchHardSyncAll(int32_t* out, int32_t* flags, int32_t totalBlocks, void* stream);
void LaunchSoftSyncAllMix11(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream);
void LaunchSoftSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t* marker, void* stream);
void LaunchHardSyncAllAIC(int32_t* out, void* stream);
void LaunchAicProbeStore(int32_t* out, void* stream);
void LaunchAicProbeStAtomic(int32_t* out, void* stream);
void LaunchAicProbeLdDev(int32_t* out, void* stream);

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

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    constexpr int32_t aicBlocks = 18;
    const size_t markerByteSize = aicBlocks * int32PerCacheLine * sizeof(int32_t);

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
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&markerDevice), markerByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    std::fill_n(markerHost, aicBlocks * int32PerCacheLine, -1);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(markerDevice, markerByteSize, markerHost, markerByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncAllMix12(outDevice, flagsDevice, syncWorkspaceDevice, markerDevice, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    std::printf("[soft_mix_1_2] aclrtSynchronizeStream ret=%d (507015=AICORE exception)\n", syncRet);

    if (aclrtMemcpy(markerHost, markerByteSize, markerDevice, markerByteSize, ACL_MEMCPY_DEVICE_TO_HOST) ==
        ACL_SUCCESS) {
        std::printf("[soft_mix_1_2] AIC stage markers (per AIC block, -1=never started, 8=completed):\n");
        for (int32_t i = 0; i < aicBlocks; ++i) {
            std::printf(" aic[%d]=%d", i, markerHost[i * int32PerCacheLine]);
        }
        std::printf("\n[soft_mix_1_2] stages: 0=enter 1=proxyWr1 2=barrier1 3=check1 4=barrier2 5=proxyWr2 "
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

    int32_t* outDevice = nullptr;
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    LaunchHardSyncAllAIC(outDevice, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

// --------------------------------------------------------------------------
// AIC intrinsic probes: isolate whether A5 AIC (cube) can run a plain GM scalar
// store, st_atomic and ld_dev. Each probe MUST be run in its own process, e.g.
//   ./syncall --gtest_filter=SYNCALLTest.case_aic_probe_store
//   ./syncall --gtest_filter=SYNCALLTest.case_aic_probe_st_atomic
//   ./syncall --gtest_filter=SYNCALLTest.case_aic_probe_ld_dev
// A 507015 stream failure means the intrinsic faults on AIC; 0 + expected value
// means it works. Cleanup always runs so a crash never leaves the device dirty.
// --------------------------------------------------------------------------
namespace {
void RunAicProbe(
    const char* label, void (*launch)(int32_t*, void*), int32_t preset0, int32_t preset8, int32_t expected)
{
    constexpr int32_t elementCount = 16;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t* hostBuf = nullptr;
    int32_t* devBuf = nullptr;
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void**>(&hostBuf), byteSize));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void**>(&devBuf), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(hostBuf, elementCount, 0);
    hostBuf[0] = preset0;
    hostBuf[8] = preset8;
    EXPECT_ACL_OK(aclrtMemcpy(devBuf, byteSize, hostBuf, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    launch(devBuf, stream);
    const int32_t syncRet = static_cast<int32_t>(aclrtSynchronizeStream(stream));
    std::printf("[aic_probe:%s] aclrtSynchronizeStream ret=%d (0=ok, 507015=AICORE exception)\n", label, syncRet);

    if (syncRet == ACL_SUCCESS) {
        const int32_t copyRet =
            static_cast<int32_t>(aclrtMemcpy(hostBuf, byteSize, devBuf, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        if (copyRet == ACL_SUCCESS) {
            std::printf("[aic_probe:%s] out[0]=%d (expected %d)\n", label, hostBuf[0], expected);
            EXPECT_EQ(hostBuf[0], expected) << label << ": AIC intrinsic executed but produced wrong value";
        } else {
            std::printf("[aic_probe:%s] D2H copy failed ret=%d\n", label, copyRet);
        }
    }
    EXPECT_EQ(syncRet, ACL_SUCCESS) << label << ": AIC intrinsic faulted at runtime (likely unsupported on AIC)";

    (void)aclrtFree(devBuf);
    (void)aclrtFreeHost(hostBuf);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
}
} // namespace

TEST_F(SYNCALLTest, case_aic_probe_store)
{
    RunAicProbe("store", LaunchAicProbeStore, /*preset0=*/0, /*preset8=*/0, /*expected=*/42);
}

TEST_F(SYNCALLTest, case_aic_probe_st_atomic)
{
    RunAicProbe("st_atomic", LaunchAicProbeStAtomic, /*preset0=*/10, /*preset8=*/0, /*expected=*/11);
}

TEST_F(SYNCALLTest, case_aic_probe_ld_dev)
{
    RunAicProbe("ld_dev", LaunchAicProbeLdDev, /*preset0=*/0, /*preset8=*/1234, /*expected=*/1234);
}
