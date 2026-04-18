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

// Forward declarations
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream);

class TADDSVloopTest : public testing::Test {
protected:
    void SetUp() override {
        aclInit(nullptr);
        aclrtSetDevice(0);
    }
    void TearDown() override {
        aclrtResetDevice(0);
        aclFinalize();
    }
};

// TADD baseline - should use RV_VLDI + RV_VLOOP
TEST_F(TADDSVloopTest, TADD_float_1x4096_baseline) {
    constexpr size_t elements = 4096;
    constexpr size_t bytes = elements * sizeof(float);
    
    void *devOut, *devSrc0, *devSrc1;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    printf("\n[TADD] Running TADD 1x4096 baseline...\n");
    printf("[TADD] Expected: RV_VLDI + RV_VLOOP (efficient hardware loop)\n");
    
    launchTADD<float, 1, 4096, 1, 4096>(devOut, devSrc0, devSrc1, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
}

// TADDS - currently uses RV_VLDS + RVECSU (inefficient)
TEST_F(TADDSVloopTest, TADDS_float_1x4096_issue) {
    constexpr size_t elements = 4096;
    constexpr size_t bytes = elements * sizeof(float);
    constexpr float scalar = 1.5f;
    
    void *devOut, *devSrc;
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    printf("\n[TADDS] Running TADDS 1x4096 with scalar=1.5f...\n");
    printf("[TADDS] Current: RV_VLDS + RVECSU (inefficient scalar loop)\n");
    printf("[TADDS] Expected: RV_VLDI + RV_VLOOP (efficient hardware loop)\n");
    
    launchTADDS<float, 1, 4096, 1, 4096>(devOut, devSrc, scalar, stream);
    aclrtSynchronizeStream(stream);
    
    aclrtDestroyStream(stream);
    aclrtFree(devOut);
    aclrtFree(devSrc);
}

int main(int argc, char **argv) {
    printf("=======================================================\n");
    printf("TADDS RV_VLOOP Hardware Loop Issue Test\n");
    printf("=======================================================\n");
    printf("This test demonstrates the performance gap between\n");
    printf("TADD and TADDS due to different instruction generation.\n");
    printf("\n");
    printf("After running, analyze dumps:\n");
    printf("  grep -c RV_VLOOP core0.veccore0.instr_log.dump\n");
    printf("  grep -c RVECSU core0.veccore0.instr_log.dump\n");
    printf("=======================================================\n\n");
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
