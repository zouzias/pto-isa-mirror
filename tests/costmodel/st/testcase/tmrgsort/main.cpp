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
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

class TMRGSORTTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kTCols_, uint32_t blockLen, float profiling, float accuracy>
void LaunchTMrgSort(void *stream);

template <typename T, int kTCols_, uint32_t blockLen, float profiling, float accuracy>
void test_tmrgsort()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTMrgSort<T, kTCols_, blockLen, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// srcCol=256, blockLen=64, repeatTimes=1: 14 + 1*2 = 16
TEST_F(TMRGSORTTest, case_half_256_blocklen64)
{
    test_tmrgsort<aclFloat16, 256, 64, 16.0f, 1.0f>();
}

// srcCol=512, blockLen=64, repeatTimes=2: 14 + 2*2 = 18
TEST_F(TMRGSORTTest, case_half_512_blocklen64)
{
    test_tmrgsort<aclFloat16, 512, 64, 18.0f, 1.0f>();
}
