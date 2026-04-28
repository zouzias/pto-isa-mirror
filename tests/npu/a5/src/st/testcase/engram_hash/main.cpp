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

using namespace std;
using namespace PtoTestCommon;

constexpr int MAX_NGRAM_SIZE = 3;
constexpr int NUM_NGRAM_LAYERS = 2;
constexpr int NUM_EMBED_TABLE_PER_NGRAM = 8;
constexpr int NUM_OUT_COLS = (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM;

class EngramHashTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kTRows_, int kTCols_, int numTokens, int numOutCols>
void LaunchEngramHash(
    T *output,
    T *ngram_token_ids,
    T *multipliers,
    T *vocab_sizes,
    T *offsets,
    void *stream);

template <typename T, int numTokens>
void test_engram_hash()
{
    constexpr int numOutCols = NUM_OUT_COLS;

    size_t tokenIdsSize = numTokens * MAX_NGRAM_SIZE * sizeof(T);
    size_t multipliersSize = NUM_NGRAM_LAYERS * MAX_NGRAM_SIZE * sizeof(T);
    size_t vocabSizesSize = NUM_NGRAM_LAYERS * (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM * sizeof(T);
    size_t offsetsSize = NUM_NGRAM_LAYERS * numOutCols * sizeof(T);
    size_t outputSize = NUM_NGRAM_LAYERS * numTokens * numOutCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    void *outputHost, *tokenIdsHost, *multipliersHost, *vocabSizesHost, *offsetsHost;
    void *outputDevice, *tokenIdsDevice, *multipliersDevice, *vocabSizesDevice, *offsetsDevice;

    aclrtMallocHost(&outputHost, outputSize);
    aclrtMallocHost(&tokenIdsHost, tokenIdsSize);
    aclrtMallocHost(&multipliersHost, multipliersSize);
    aclrtMallocHost(&vocabSizesHost, vocabSizesSize);
    aclrtMallocHost(&offsetsHost, offsetsSize);

    aclrtMalloc(&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&tokenIdsDevice, tokenIdsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&multipliersDevice, multipliersSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&vocabSizesDevice, vocabSizesSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&offsetsDevice, offsetsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/ngram_token_ids.bin", tokenIdsSize, tokenIdsHost, tokenIdsSize);
    ReadFile(GetGoldenDir() + "/multipliers.bin", multipliersSize, multipliersHost, multipliersSize);
    ReadFile(GetGoldenDir() + "/vocab_sizes.bin", vocabSizesSize, vocabSizesHost, vocabSizesSize);
    ReadFile(GetGoldenDir() + "/offsets.bin", offsetsSize, offsetsHost, offsetsSize);

    aclrtMemcpy(tokenIdsDevice, tokenIdsSize, tokenIdsHost, tokenIdsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(multipliersDevice, multipliersSize, multipliersHost, multipliersSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vocabSizesDevice, vocabSizesSize, vocabSizesHost, vocabSizesSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetsDevice, offsetsSize, offsetsHost, offsetsSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchEngramHash<T, numTokens, MAX_NGRAM_SIZE, numTokens, numOutCols>(
        static_cast<T *>(outputDevice), static_cast<T *>(tokenIdsDevice),
        static_cast<T *>(multipliersDevice), static_cast<T *>(vocabSizesDevice),
        static_cast<T *>(offsetsDevice), stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outputHost, outputSize);

    aclrtFree(outputDevice);
    aclrtFree(tokenIdsDevice);
    aclrtFree(multipliersDevice);
    aclrtFree(vocabSizesDevice);
    aclrtFree(offsetsDevice);

    aclrtFreeHost(outputHost);
    aclrtFreeHost(tokenIdsHost);
    aclrtFreeHost(multipliersHost);
    aclrtFreeHost(vocabSizesHost);
    aclrtFreeHost(offsetsHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outputSize);
    std::vector<T> devFinal(outputSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputSize, devFinal.data(), outputSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(EngramHashTest, case_int32_1)
{
    test_engram_hash<int32_t, 1>();
}

TEST_F(EngramHashTest, case_int32_16)
{
    test_engram_hash<int32_t, 16>();
}

TEST_F(EngramHashTest, case_int32_128)
{
    test_engram_hash<int32_t, 128>();
}

TEST_F(EngramHashTest, case_int64_1)
{
    test_engram_hash<int64_t, 1>();
}

TEST_F(EngramHashTest, case_int64_16)
{
    test_engram_hash<int64_t, 16>();
}

TEST_F(EngramHashTest, case_int64_128)
{
    test_engram_hash<int64_t, 128>();
}