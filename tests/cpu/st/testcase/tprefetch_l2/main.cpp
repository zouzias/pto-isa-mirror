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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;

class TPREFETCH_L2_Test : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kGRows_, int kGCols_>
void LaunchTPrefetchL2(T *src, void *stream);

template <typename T, int kGRows_, int kGCols_>
void test_tprefetch_l2()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *srcHost;
    T *srcDevice;

    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    for (size_t i = 0; i < static_cast<size_t>(kGRows_) * kGCols_; ++i) {
        srcHost[i] = static_cast<T>(i);
    }

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPrefetchL2<T, kGRows_, kGCols_>(srcDevice, stream);
    aclrtSynchronizeStream(stream);

    aclrtFree(srcDevice);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    SUCCEED();
}

TEST_F(TPREFETCH_L2_Test, case_float_64x64)
{
    test_tprefetch_l2<float, 64, 64>();
}

TEST_F(TPREFETCH_L2_Test, case_int32_64x64)
{
    test_tprefetch_l2<int32_t, 64, 64>();
}

TEST_F(TPREFETCH_L2_Test, case_half_16x256)
{
    test_tprefetch_l2<aclFloat16, 16, 256>();
}
