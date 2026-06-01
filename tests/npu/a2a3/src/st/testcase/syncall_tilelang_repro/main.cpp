/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "syncall_tilelang_repro_common.hpp"
#include "test_common.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>

using namespace PtoTestCommon;

void LaunchSyncallTilelangRepro(uint8_t *workspaceGm, uint8_t *resultGm, void *stream, int32_t variantRaw);

namespace {
std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    const std::string fullPath = "../" + suiteName + "." + caseName;
    std::filesystem::create_directories(fullPath);
    return fullPath;
}

int32_t GetSlotStride(ReproVariant variant)
{
    switch (variant) {
        case ReproVariant::FixCacheLine:
        case ReproVariant::FixCacheLineSyncMix:
            return 8;
        default:
            return kReproNumExperts;
    }
}

int32_t CountWorkspaceMismatches(const int32_t *workspace, ReproVariant variant)
{
    const int32_t wsElems = GetReproWorkspaceElems(variant);
    const int32_t stride = GetSlotStride(variant);
    int32_t mismatches = 0;
    for (int32_t core = 0; core < kReproNumCores; ++core) {
        const int32_t expected = core + 1;
        for (int32_t e = 0; e < kReproNumExperts; ++e) {
            const int32_t actual = workspace[core * stride + e];
            if (actual != expected) {
                ++mismatches;
            }
        }
    }
    (void)wsElems;
    return mismatches;
}

void PrintWorkspace(const int32_t *workspace, ReproVariant variant)
{
    const int32_t wsElems = GetReproWorkspaceElems(variant);
    std::printf("workspace_gm (expected: [1,1,1,1, 2,2,2,2, ..., 16,16,16,16]):\nActual:");
    for (int32_t i = 0; i < wsElems; ++i) {
        if (i % 16 == 0) {
            std::printf("\n");
        }
        std::printf(" %d", workspace[i]);
    }
    std::printf("\n");
}

void PrintCoreResults(const int32_t *resultHost, ReproVariant variant)
{
    const int32_t wsElems = GetReproWorkspaceElems(variant);
    const int32_t stride = GetSlotStride(variant);
    for (int32_t core = 0; core < kReproNumCores; ++core) {
        const int32_t *coreResult = resultHost + core * wsElems;
        int32_t coreMismatches = 0;
        for (int32_t c = 0; c < kReproNumCores; ++c) {
            for (int32_t e = 0; e < kReproNumExperts; ++e) {
                if (coreResult[c * stride + e] != c + 1) {
                    ++coreMismatches;
                }
            }
        }
        if (coreMismatches > 0) {
            std::printf("Core %d: INCORRECT (%d/%d mismatches)\n", core, coreMismatches,
                        kReproNumCores * kReproNumExperts);
        }
    }
}

void ExpectWorkspaceGolden(const int32_t *workspace, ReproVariant variant)
{
    const int32_t stride = GetSlotStride(variant);
    for (int32_t core = 0; core < kReproNumCores; ++core) {
        const int32_t expected = core + 1;
        for (int32_t e = 0; e < kReproNumExperts; ++e) {
            EXPECT_EQ(workspace[core * stride + e], expected)
                << "workspace mismatch at core " << core << " expert " << e;
        }
    }
}

void RunReproCase(ReproVariant variant)
{
    const int32_t wsElems = GetReproWorkspaceElems(variant);
    const size_t workspaceBytes = static_cast<size_t>(wsElems) * sizeof(int32_t);
    const size_t resultBytes = static_cast<size_t>(kReproNumCores) * workspaceBytes;

    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);

    int32_t *workspaceHost = nullptr;
    int32_t *resultHost = nullptr;
    uint8_t *workspaceDevice = nullptr;
    uint8_t *resultDevice = nullptr;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&workspaceHost), workspaceBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&resultHost), resultBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&resultDevice), resultBytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);

    std::fill_n(workspaceHost, wsElems, 0);
    std::fill_n(resultHost, static_cast<size_t>(kReproNumCores) * wsElems, 0);
    ASSERT_EQ(aclrtMemcpy(workspaceDevice, workspaceBytes, workspaceHost, workspaceBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(resultDevice, resultBytes, resultHost, resultBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    LaunchSyncallTilelangRepro(workspaceDevice, resultDevice, stream, static_cast<int32_t>(variant));
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(workspaceHost, workspaceBytes, workspaceDevice, workspaceBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(resultHost, resultBytes, resultDevice, resultBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/workspace.bin", workspaceHost, workspaceBytes));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/result.bin", resultHost, resultBytes));

    const int32_t mismatches = CountWorkspaceMismatches(workspaceHost, variant);
    if (mismatches > 0) {
        PrintWorkspace(workspaceHost, variant);
        std::printf("Mismatches: %d / %d\n", mismatches, kReproNumCores * kReproNumExperts);
        PrintCoreResults(resultHost, variant);
    }

    ExpectWorkspaceGolden(workspaceHost, variant);

    ASSERT_EQ(aclrtFree(workspaceDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(resultDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(workspaceHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(resultHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    ASSERT_EQ(aclFinalize(), ACL_SUCCESS);
}
} // namespace

class SYNCALLTilelangReproTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

TEST_F(SYNCALLTilelangReproTest, case_baseline)
{
    RunReproCase(ReproVariant::Baseline);
}

TEST_F(SYNCALLTilelangReproTest, case_mix_aivonly_16cores)
{
    RunReproCase(ReproVariant::Baseline);
}

TEST_F(SYNCALLTilelangReproTest, case_fix_cacheline)
{
    RunReproCase(ReproVariant::FixCacheLine);
}

TEST_F(SYNCALLTilelangReproTest, case_fix_sync_mix)
{
    RunReproCase(ReproVariant::FixSyncMix);
}

TEST_F(SYNCALLTilelangReproTest, case_fix_aiv_only)
{
    RunReproCase(ReproVariant::FixAivOnly);
}

TEST_F(SYNCALLTilelangReproTest, case_fix_cacheline_sync_mix)
{
    RunReproCase(ReproVariant::FixCacheLineSyncMix);
}
