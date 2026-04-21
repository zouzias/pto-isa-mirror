/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/**
 * AscendC Eltwise Test - GTest harness
 *
 * Measures EPC for AscendC Add kernel variants on A5 simulator.
 * EPC extracted from vf_real_execute_time in core0.veccore0.instr_log.dump
 *
 * Comparison target: PTO TADD results from tile_perf/REPORT.md
 *   fp32 1x8192:  PTO EPC = 27.68
 *   fp32 16x512:  PTO EPC = 27.31
 *   fp32 32x256:  PTO EPC = 24.24
 *   fp16 1x16384: PTO EPC = 55.35
 *   fp16 32x512:  PTO EPC = 48.47
 *   fp16 64x256:  PTO EPC = 32.90
 */
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

// Kernel declarations
extern "C" {
void asc_add_flat_fp32_1x8192(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_flat_fp32_16x512(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_flat_fp32_32x256(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_flat_fp16_1x16384(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_flat_fp16_32x512(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_flat_fp16_64x256(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_rowloop_fp32_16x512(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_rowloop_fp32_32x256(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_rowloop_fp16_32x512(uint8_t* x, uint8_t* y, uint8_t* z);
void asc_add_rowloop_fp16_64x256(uint8_t* x, uint8_t* y, uint8_t* z);
}

template <typename T>
void RunAddTest(const char* test_name,
                void (*kernel)(uint8_t*, uint8_t*, uint8_t*),
                size_t elements) {
    size_t byte_size = elements * sizeof(T);

    uint8_t *x = nullptr, *y = nullptr, *z = nullptr;
    aclrtMalloc((void**)&x, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&y, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&z, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // Init host data and copy to device
    std::vector<T> hx(elements), hy(elements), hz(elements);
    for (size_t i = 0; i < elements; i++) {
        hx[i] = static_cast<T>(1.0f);
        hy[i] = static_cast<T>(2.0f);
    }
    aclrtMemcpy(x, byte_size, hx.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(y, byte_size, hy.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Run kernel
    kernel(x, y, z);

    // Copy result back and verify
    aclrtMemcpy(hz.data(), byte_size, z, byte_size, ACL_MEMCPY_DEVICE_TO_HOST);

    bool pass = true;
    for (size_t i = 0; i < elements; i++) {
        float expected = 3.0f;
        float actual = static_cast<float>(hz[i]);
        if (std::fabs(actual - expected) > 1e-3f) {
            pass = false;
            printf("[%s] MISMATCH at [%zu]: expected=%.3f got=%.3f\n",
                   test_name, i, expected, actual);
            break;
        }
    }
    EXPECT_TRUE(pass) << test_name << " precision check failed";

    aclrtFree(x);
    aclrtFree(y);
    aclrtFree(z);
}

class AscEltwiseTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        aclInit(nullptr);
        aclrtSetDevice(0);
    }
    static void TearDownTestSuite() {
        aclrtResetDevice(0);
        aclFinalize();
    }
};

// FLAT variants (whole-tile Add — potential VF row fusion)
TEST_F(AscEltwiseTest, flat_fp32_1x8192) {
    RunAddTest<float>("flat_fp32_1x8192", asc_add_flat_fp32_1x8192, 1 * 8192);
}
TEST_F(AscEltwiseTest, flat_fp32_16x512) {
    RunAddTest<float>("flat_fp32_16x512", asc_add_flat_fp32_16x512, 16 * 512);
}
TEST_F(AscEltwiseTest, flat_fp32_32x256) {
    RunAddTest<float>("flat_fp32_32x256", asc_add_flat_fp32_32x256, 32 * 256);
}
TEST_F(AscEltwiseTest, flat_fp16_1x16384) {
    RunAddTest<half>("flat_fp16_1x16384", asc_add_flat_fp16_1x16384, 1 * 16384);
}
TEST_F(AscEltwiseTest, flat_fp16_32x512) {
    RunAddTest<half>("flat_fp16_32x512", asc_add_flat_fp16_32x512, 32 * 512);
}
TEST_F(AscEltwiseTest, flat_fp16_64x256) {
    RunAddTest<half>("flat_fp16_64x256", asc_add_flat_fp16_64x256, 64 * 256);
}

// ROWLOOP variants (per-row Add — no VF row fusion)
TEST_F(AscEltwiseTest, rowloop_fp32_16x512) {
    RunAddTest<float>("rowloop_fp32_16x512", asc_add_rowloop_fp32_16x512, 16 * 512);
}
TEST_F(AscEltwiseTest, rowloop_fp32_32x256) {
    RunAddTest<float>("rowloop_fp32_32x256", asc_add_rowloop_fp32_32x256, 32 * 256);
}
TEST_F(AscEltwiseTest, rowloop_fp16_32x512) {
    RunAddTest<half>("rowloop_fp16_32x512", asc_add_rowloop_fp16_32x512, 32 * 512);
}
TEST_F(AscEltwiseTest, rowloop_fp16_64x256) {
    RunAddTest<half>("rowloop_fp16_64x256", asc_add_rowloop_fp16_64x256, 64 * 256);
}
