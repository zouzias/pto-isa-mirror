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
#include "syncall_core_config.hpp"
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

void LaunchSyncAll(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t totalBlocks, void *stream);
void LaunchSoftSyncAll(int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t totalBlocks, void *stream);
void LaunchHardSyncAllAIC(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t launchBlocks, void *stream);
void LaunchSoftSyncAllAIC(int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t totalBlocks, void *stream);
void LaunchSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t aicBlocks, void *stream);
void LaunchSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t aicBlocks, void *stream);
void LaunchSoftSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t aicBlocks,
                            int32_t totalParticipants, void *stream);
void LaunchSoftSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t aicBlocks,
                            int32_t totalParticipants, void *stream);
void LaunchSoftSyncDomainMix12(int32_t *out, int32_t *syncWorkspace, int32_t aicBlocks, int32_t totalParticipants,
                               void *stream);
void LaunchVcSoftVecAllDoneProducer(int32_t *payload, int32_t *syncFlags, int32_t vecParticipants, int32_t seq,
                                    void *stream);
void LaunchVcSoftCubeWaitConsumer(int32_t *out, int32_t *payload, int32_t *syncFlags, int32_t aicBlocks,
                                  int32_t vecParticipants, int32_t seq, void *stream);

#define EXPECT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

#define EXPECT_RT_OK(expr)                                    \
    do {                                                      \
        const auto ret = (expr);                              \
        ASSERT_EQ(ret, 0) << #expr << " failed, ret=" << ret; \
    } while (0)

template <bool withWorkspace, typename LaunchFn>
void RunMixCase(size_t blockCount, LaunchFn launchFn, const char *label)
{
    constexpr size_t int32PerCacheLine = 8;
    const size_t elementCount = blockCount * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(
            aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(aclrtMemcpy(syncWorkspaceDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    if constexpr (withWorkspace) {
        launchFn(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, syncWorkspaceDevice, stream);
    } else {
        launchFn(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, stream);
    }
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
        std::printf("%s out[0..7]:", label);
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\n%s flags[0..7]:", label);
        for (size_t i = 0; i < std::min<size_t>(8, blockCount); ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
        }
        std::printf("\n");
    }
    EXPECT_TRUE(ret);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(flagsDevice));
    if constexpr (withWorkspace) {
        EXPECT_ACL_OK(aclrtFree(syncWorkspaceDevice));
    }
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(flagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_aiv_only_all_blocks)
{
    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    const size_t blockCount = static_cast<size_t>(syncall_cfg::GetCoreConfig().aiv);
    constexpr size_t int32PerCacheLine = 8;
    const size_t elementCount = blockCount * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    LaunchSyncAll(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, static_cast<int32_t>(blockCount), stream);
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
        std::printf("out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nflags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
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

TEST_F(SYNCALLTest, case_soft_aiv_only_all_blocks)
{
    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    const size_t blockCount = static_cast<size_t>(syncall_cfg::GetCoreConfig().aiv);
    constexpr size_t int32PerCacheLine = 8;
    const size_t elementCount = blockCount * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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

    LaunchSoftSyncAll(outDevice, flagsDevice, syncWorkspaceDevice, static_cast<int32_t>(blockCount), stream);
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
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
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

TEST_F(SYNCALLTest, case_mix_1_1_all_blocks)
{
    const int32_t aicBlocks = syncall_cfg::GetCoreConfig().aic;
    const int32_t total = aicBlocks * 2; // 1 cube + 1 vector per cube
    RunMixCase<false>(
        static_cast<size_t>(total),
        [aicBlocks](uint8_t *ffts, int32_t *out, int32_t *flags, void *stream) {
            LaunchSyncAllMix11(ffts, out, flags, aicBlocks, stream);
        },
        "mix_1_1");
}

TEST_F(SYNCALLTest, case_mix_1_2_all_blocks)
{
    const int32_t aicBlocks = syncall_cfg::GetCoreConfig().aic;
    const int32_t total = aicBlocks * 3; // 1 cube + 2 vectors per cube
    RunMixCase<false>(
        static_cast<size_t>(total),
        [aicBlocks](uint8_t *ffts, int32_t *out, int32_t *flags, void *stream) {
            LaunchSyncAllMix12(ffts, out, flags, aicBlocks, stream);
        },
        "mix_1_2");
}

TEST_F(SYNCALLTest, case_soft_mix_1_1_all_blocks)
{
    const int32_t aicBlocks = syncall_cfg::GetCoreConfig().aic;
    const int32_t total = aicBlocks * 2;
    RunMixCase<true>(
        static_cast<size_t>(total),
        [aicBlocks, total](uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *ws, void *stream) {
            LaunchSoftSyncAllMix11(ffts, out, flags, ws, aicBlocks, total, stream);
        },
        "soft_mix_1_1");
}

TEST_F(SYNCALLTest, case_soft_mix_1_2_all_blocks)
{
    const int32_t aicBlocks = syncall_cfg::GetCoreConfig().aic;
    const int32_t total = aicBlocks * 3;
    RunMixCase<true>(
        static_cast<size_t>(total),
        [aicBlocks, total](uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *ws, void *stream) {
            LaunchSoftSyncAllMix12(ffts, out, flags, ws, aicBlocks, total, stream);
        },
        "soft_mix_1_2");
}

TEST_F(SYNCALLTest, case_soft_sync_domain_mix_1_2)
{
    const int32_t aicBlocks = syncall_cfg::GetCoreConfig().aic;
    const int32_t total = aicBlocks * 3;
    constexpr int32_t int32PerCacheLine = 8;
    const size_t outElements = static_cast<size_t>(total * int32PerCacheLine);
    constexpr size_t workspaceElements = 2048;
    const size_t outBytes = outElements * sizeof(int32_t);
    constexpr size_t workspaceBytes = workspaceElements * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *workspaceHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *workspaceDevice = nullptr;
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), outBytes));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&workspaceHost), workspaceBytes));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, outElements, 0);
    std::fill_n(workspaceHost, workspaceElements, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(
        aclrtMemcpy(workspaceDevice, workspaceBytes, workspaceHost, workspaceBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchSoftSyncDomainMix12(outDevice, workspaceDevice, aicBlocks, total, stream);
    EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    EXPECT_ACL_OK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes));

    bool ret = true;
    for (int32_t i = 0; i < total; ++i) {
        int32_t expect = (i < aicBlocks) ? 1 : 10;
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

TEST_F(SYNCALLTest, case_vc_soft_vec_all_done_then_cube_1_2_seven_blocks)
{
    const int32_t availableAicBlocks = syncall_cfg::GetCoreConfig().aic;
    ASSERT_GE(availableAicBlocks, 7);

    constexpr int32_t aicBlocks = 7;
    constexpr int32_t vecParticipants = aicBlocks * 2;
    constexpr int32_t int32PerCacheLine = 8;
    const size_t outElements = static_cast<size_t>(aicBlocks * int32PerCacheLine);
    const size_t payloadElements = static_cast<size_t>(vecParticipants * int32PerCacheLine);
    const size_t flagElements = static_cast<size_t>((vecParticipants + 1) * int32PerCacheLine);
    const size_t outBytes = outElements * sizeof(int32_t);
    const size_t payloadBytes = payloadElements * sizeof(int32_t);
    const size_t flagBytes = flagElements * sizeof(int32_t);

    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    int32_t *outHost = nullptr;
    int32_t *payloadHost = nullptr;
    int32_t *syncFlagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *payloadDevice = nullptr;
    int32_t *syncFlagsDevice = nullptr;

    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&outHost), outBytes));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&payloadHost), payloadBytes));
    EXPECT_ACL_OK(aclrtMallocHost(reinterpret_cast<void **>(&syncFlagsHost), flagBytes));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&outDevice), outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&payloadDevice), payloadBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    EXPECT_ACL_OK(aclrtMalloc(reinterpret_cast<void **>(&syncFlagsDevice), flagBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::fill_n(outHost, outElements, 0);
    std::fill_n(payloadHost, payloadElements, 0);
    std::fill_n(syncFlagsHost, flagElements, 0);
    EXPECT_ACL_OK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(payloadDevice, payloadBytes, payloadHost, payloadBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    EXPECT_ACL_OK(aclrtMemcpy(syncFlagsDevice, flagBytes, syncFlagsHost, flagBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto runRound = [&](int32_t seq) {
        std::fill_n(outHost, outElements, 0);
        EXPECT_ACL_OK(aclrtMemcpy(outDevice, outBytes, outHost, outBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        LaunchVcSoftVecAllDoneProducer(payloadDevice, syncFlagsDevice, vecParticipants, seq, stream);
        EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
        LaunchVcSoftCubeWaitConsumer(outDevice, payloadDevice, syncFlagsDevice, aicBlocks, vecParticipants, seq,
                                     stream);
        EXPECT_ACL_OK(aclrtSynchronizeStream(stream));
    };

    runRound(1);
    runRound(2);

    EXPECT_ACL_OK(aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(payloadHost, payloadBytes, payloadDevice, payloadBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    EXPECT_ACL_OK(aclrtMemcpy(syncFlagsHost, flagBytes, syncFlagsDevice, flagBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes));

    for (int32_t i = 0; i < aicBlocks; ++i) {
        EXPECT_EQ(outHost[i * int32PerCacheLine], 1) << "cube block " << i << " observed incomplete Vec payload";
    }
    for (int32_t i = 0; i < vecParticipants; ++i) {
        EXPECT_EQ(payloadHost[i * int32PerCacheLine], 2000 + i) << "payload slot " << i;
        EXPECT_EQ(syncFlagsHost[i * int32PerCacheLine], 2) << "Vec ready slot " << i;
    }
    EXPECT_EQ(syncFlagsHost[vecParticipants * int32PerCacheLine], 2);

    EXPECT_ACL_OK(aclrtFree(outDevice));
    EXPECT_ACL_OK(aclrtFree(payloadDevice));
    EXPECT_ACL_OK(aclrtFree(syncFlagsDevice));
    EXPECT_ACL_OK(aclrtFreeHost(outHost));
    EXPECT_ACL_OK(aclrtFreeHost(payloadHost));
    EXPECT_ACL_OK(aclrtFreeHost(syncFlagsHost));
    EXPECT_ACL_OK(aclrtDestroyStream(stream));
    EXPECT_ACL_OK(aclrtResetDevice(0));
    EXPECT_ACL_OK(aclFinalize());
}

TEST_F(SYNCALLTest, case_hard_aic_only_all_blocks)
{
    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    const size_t blockCount = static_cast<size_t>(syncall_cfg::GetCoreConfig().aic);
    constexpr size_t int32PerCacheLine = 8;
    const size_t elementCount = blockCount * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    EXPECT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));
    ASSERT_NE(ffts, 0UL);

    LaunchHardSyncAllAIC(reinterpret_cast<uint8_t *>(ffts), outDevice, flagsDevice, static_cast<int32_t>(blockCount),
                         stream);
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
        std::printf("hard_aic out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nhard_aic flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", flagsHost[i * int32PerCacheLine]);
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

TEST_F(SYNCALLTest, case_soft_aic_only_all_blocks)
{
    EXPECT_ACL_OK(aclInit(nullptr));
    EXPECT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    EXPECT_ACL_OK(aclrtCreateStream(&stream));

    const size_t blockCount = static_cast<size_t>(syncall_cfg::GetCoreConfig().aic);
    constexpr size_t int32PerCacheLine = 8;
    const size_t elementCount = blockCount * int32PerCacheLine;
    const size_t byteSize = elementCount * sizeof(int32_t);

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

    LaunchSoftSyncAllAIC(outDevice, flagsDevice, syncWorkspaceDevice, static_cast<int32_t>(blockCount), stream);
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
        std::printf("soft_aic out[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
            std::printf(" %d", outHost[i * int32PerCacheLine]);
        }
        std::printf("\nsoft_aic flags[0..7]:");
        for (size_t i = 0; i < 8; ++i) {
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
