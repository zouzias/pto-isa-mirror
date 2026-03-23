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

class TTRANSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kSrcRows_, int kSrcCols_, float profiling, float accuracy>
void LaunchTTrans(void *stream);

// half, 16x16: numSubTileX=1, ceil_Y=1 → startup(14) + 1*1*2 = 16
TEST_F(TTRANSTest, case_half_16x16)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTTrans<aclFloat16, 16, 16, 16.0f, 1.0f>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// half, 16x32: numSubTileX=2, ceil_Y=1 → startup(14) + 2*1*2 = 18
TEST_F(TTRANSTest, case_half_16x32)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTTrans<aclFloat16, 16, 32, 18.0f, 1.0f>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// float, 16x16: blockSizeElem=8, numSubTileX=2, ceil_Y=1 → startup(14) + 2*1*2 = 18
TEST_F(TTRANSTest, case_float_16x16)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTTrans<float, 16, 16, 18.0f, 1.0f>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}
