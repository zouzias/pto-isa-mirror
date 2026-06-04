/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_token_permute_common.hpp"
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include <vector>

using namespace PtoTestCommon;

void LaunchMoeTokenPermuteChevron(aclFloat16 *tokensGm, int32_t *indicesGm, aclFloat16 *permOutGm, int32_t *sioOutGm,
                                  int32_t *workspaceGm, void *stream);

namespace {
std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

void PrintMismatchSamples(const int32_t *workspace, const aclFloat16 *permOut, const int32_t *sioOut,
                          const std::vector<aclFloat16> &goldenPerm, const std::vector<int32_t> &goldenSio)
{
    std::printf("workspace[0:15]:");
    for (int32_t i = 0; i < 16; ++i) {
        std::printf(" %d", workspace[i]);
    }
    std::printf("\nsio[0:15]:");
    for (int32_t i = 0; i < 16; ++i) {
        std::printf(" %d", sioOut[i]);
    }
    std::printf("\nexpected sio[0:15]:");
    for (int32_t i = 0; i < 16; ++i) {
        std::printf(" %d", goldenSio[i]);
    }
    std::printf("\nperm mismatch count (first 8 rows): ");
    int32_t permMismatch = 0;
    for (int32_t row = 0; row < 8; ++row) {
        for (int32_t col = 0; col < kMoeHiddenSize; ++col) {
            const size_t idx = static_cast<size_t>(row) * kMoeHiddenSize + col;
            if (static_cast<float>(permOut[idx]) != static_cast<float>(goldenPerm[idx])) {
                ++permMismatch;
            }
        }
    }
    std::printf("%d\n", permMismatch);
}

void RunChevronCase()
{
    const size_t tokensBytes = static_cast<size_t>(kMoeNumTokens) * kMoeHiddenSize * sizeof(aclFloat16);
    const size_t indicesBytes = static_cast<size_t>(kMoeE) * sizeof(int32_t);
    const size_t permBytes = static_cast<size_t>(kMoeOutLen) * kMoeHiddenSize * sizeof(aclFloat16);
    const size_t sioBytes = static_cast<size_t>(kMoeE) * sizeof(int32_t);
    const size_t workspaceBytes = static_cast<size_t>(kMoeWsTotal) * sizeof(int32_t);

    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);

    aclFloat16 *tokensHost = nullptr;
    int32_t *indicesHost = nullptr;
    aclFloat16 *permHost = nullptr;
    int32_t *sioHost = nullptr;
    int32_t *workspaceHost = nullptr;
    aclFloat16 *tokensDevice = nullptr;
    int32_t *indicesDevice = nullptr;
    aclFloat16 *permDevice = nullptr;
    int32_t *sioDevice = nullptr;
    int32_t *workspaceDevice = nullptr;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&tokensHost), tokensBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&indicesHost), indicesBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&permHost), permBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&sioHost), sioBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void **>(&workspaceHost), workspaceBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&tokensDevice), tokensBytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&indicesDevice), indicesBytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&permDevice), permBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&sioDevice), sioBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_SUCCESS);

    const std::string goldenDir = GetGoldenDir();
    size_t fileSize = 0;
    ASSERT_TRUE(ReadFile(goldenDir + "/tokens.bin", fileSize, tokensHost, tokensBytes));
    ASSERT_TRUE(ReadFile(goldenDir + "/indices.bin", fileSize, indicesHost, indicesBytes));

    std::fill_n(permHost, static_cast<size_t>(kMoeOutLen) * kMoeHiddenSize, aclFloat16(0));
    std::fill_n(sioHost, static_cast<size_t>(kMoeE), 0);
    std::fill_n(workspaceHost, static_cast<size_t>(kMoeWsTotal), 0);

    ASSERT_EQ(aclrtMemcpy(tokensDevice, tokensBytes, tokensHost, tokensBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(indicesDevice, indicesBytes, indicesHost, indicesBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(permDevice, permBytes, permHost, permBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(sioDevice, sioBytes, sioHost, sioBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(workspaceDevice, workspaceBytes, workspaceHost, workspaceBytes, ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    LaunchMoeTokenPermuteChevron(tokensDevice, indicesDevice, permDevice, sioDevice, workspaceDevice, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    ASSERT_EQ(aclrtMemcpy(permHost, permBytes, permDevice, permBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(sioHost, sioBytes, sioDevice, sioBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(workspaceHost, workspaceBytes, workspaceDevice, workspaceBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);

    std::filesystem::create_directories(goldenDir);
    ASSERT_TRUE(WriteFile(goldenDir + "/perm_out.bin", permHost, permBytes));
    ASSERT_TRUE(WriteFile(goldenDir + "/sio_out.bin", sioHost, sioBytes));
    ASSERT_TRUE(WriteFile(goldenDir + "/workspace.bin", workspaceHost, workspaceBytes));

    std::vector<aclFloat16> goldenPerm(static_cast<size_t>(kMoeOutLen) * kMoeHiddenSize);
    std::vector<int32_t> goldenSio(static_cast<size_t>(kMoeE));
    ASSERT_TRUE(ReadFile(goldenDir + "/golden_perm.bin", fileSize, goldenPerm.data(), permBytes));
    ASSERT_TRUE(ReadFile(goldenDir + "/golden_sio.bin", fileSize, goldenSio.data(), sioBytes));

    const bool permOk = ResultCmp<aclFloat16>(goldenPerm, permHost, 0.0f, 0, 1000, false, true);
    const bool sioOk = ResultCmp<int32_t>(goldenSio, sioHost, 0.0f, 0, 1000, false, true);
    if (!permOk || !sioOk) {
        PrintMismatchSamples(workspaceHost, permHost, sioHost, goldenPerm, goldenSio);
    }
    EXPECT_TRUE(permOk);
    EXPECT_TRUE(sioOk);

    ASSERT_EQ(aclrtFree(tokensDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(indicesDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(permDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(sioDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFree(workspaceDevice), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(tokensHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(indicesHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(permHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(sioHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtFreeHost(workspaceHost), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    ASSERT_EQ(aclFinalize(), ACL_SUCCESS);
}
} // namespace

class MoeTokenPermuteChevronTest : public testing::Test {};

TEST_F(MoeTokenPermuteChevronTest, case_fp16_standard)
{
    RunChevronCase();
}
