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

class TSORT32Test : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTSort32(void *stream);

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void test_tsort32()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTSort32<T, kTRows_, kTCols_, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// 16 + 3*20 = 76
TEST_F(TSORT32Test, case_half_4x32)
{
    test_tsort32<aclFloat16, 4, 32, 76.0f, 1.0f>();
}

// 16
TEST_F(TSORT32Test, case_half_1x32)
{
    test_tsort32<aclFloat16, 1, 32, 16.0f, 1.0f>();
}

TEST_F(TSORT32Test, case_float_4x32)
{
    test_tsort32<float, 4, 32, 76.0f, 1.0f>();
}
