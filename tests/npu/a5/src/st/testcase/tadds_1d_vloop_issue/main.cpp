/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <gtest/gtest.h>
#include <cstdio>

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream);

class TADDS1DVloopTest : public testing::Test {
protected:
    void SetUp() override { aclInit(nullptr); aclrtSetDevice(0); }
    void TearDown() override { aclrtResetDevice(0); aclFinalize(); }
};

// TADDS 1D - shows RV_VLDS + RVECSU issue (no RV_VLOOP)
TEST_F(TADDS1DVloopTest, TADDS_float_1x4096) {
    void *devOut, *devSrc;
    aclrtMalloc(&devOut, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADDS<float, 1, 4096, 1, 4096>(devOut, devSrc, 1.5f, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc);
}

// TADD 1D - baseline with RV_VLDI + RV_VLOOP
TEST_F(TADDS1DVloopTest, TADD_float_1x4096_baseline) {
    void *devOut, *devSrc0, *devSrc1;
    aclrtMalloc(&devOut, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc0, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc1, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    launchTADD<float, 1, 4096, 1, 4096>(devOut, devSrc0, devSrc1, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
}

int main(int argc, char **argv) {
    printf("TADDS 1D RV_VLOOP Issue Test\n");
    printf("After running, check: grep -c RV_VLOOP dump; grep -c RVECSU dump\n\n");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
