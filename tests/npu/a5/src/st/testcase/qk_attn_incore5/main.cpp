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

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchQKAttnIncore5(uint8_t *all_raw_scores, uint8_t *k_cache, uint8_t *q_padded, int32_t sb_idx,
                         int32_t ctx_blocks, int32_t b_idx, int32_t kvh, void *stream);

class QKAttnIncore5Test : public testing::Test {
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

template <int32_t key>
void QKAttnIncore5TestFunc(uint32_t M, uint32_t K, uint32_t N, int32_t ctx_blocks)
{
    // Input sizes
    size_t kFileSize = N * K * sizeof(uint16_t);       // k_gm.bin: [64, 128] bf16
    size_t qFileSize = M * K * sizeof(uint16_t);       // q_gm.bin: [16, 128] bf16
    // Output: [1024, 64] float (only first M*N floats are valid for ctx_blocks=1)
    size_t outTotalSize = 1024 * N * sizeof(float);
    size_t goldenSize = M * N * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *kHost, *qHost, *outHost;
    uint8_t *kDevice, *qDevice, *outDevice;

    aclrtMallocHost((void **)(&kHost), kFileSize);
    aclrtMallocHost((void **)(&qHost), qFileSize);
    aclrtMallocHost((void **)(&outHost), outTotalSize);

    aclrtMalloc((void **)&kDevice, kFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qDevice, qFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, outTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(outDevice, outTotalSize, 0, outTotalSize);

    ReadFile(GetGoldenDir() + "/k_gm.bin", kFileSize, kHost, kFileSize);
    ReadFile(GetGoldenDir() + "/q_gm.bin", qFileSize, qHost, qFileSize);

    aclrtMemcpy(kDevice, kFileSize, kHost, kFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(qDevice, qFileSize, qHost, qFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // sb_idx=0, ctx_blocks=1, b_idx=0, kvh=0
    LaunchQKAttnIncore5<key>(outDevice, kDevice, qDevice, 0, ctx_blocks, 0, 0, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outTotalSize, outDevice, outTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, outTotalSize);

    aclrtFree(kDevice);
    aclrtFree(qDevice);
    aclrtFree(outDevice);
    aclrtFreeHost(kHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare first M*N*4 bytes against golden
    std::vector<float> golden(M * N);
    std::vector<float> devFinal(M * N);
    ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), goldenSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", goldenSize, devFinal.data(), goldenSize);

    bool ret = ResultCmp(golden, devFinal, 0.01f);
    EXPECT_TRUE(ret);
}

TEST_F(QKAttnIncore5Test, case_bf16_ctx1)
{
    // M=16 (Q rows), K=128 (head_dim), N=64 (K rows = seq), ctx_blocks=1
    QKAttnIncore5TestFunc<1>(16, 128, 64, 1);
}
