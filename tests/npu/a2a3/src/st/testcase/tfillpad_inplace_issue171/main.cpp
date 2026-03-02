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
 * Issue #171 Reproducer: TFILLPAD_INPLACE bug for N=16, valid_len in [1,8]
 * 
 * The bug is in TFillPad (include/pto/npu/a2a3/TFillPad.hpp):
 * - Path B (PadRightSingleRow + PadRightRemainingRows) uses vcopy with srcRepeatStride=0
 * - This produces incorrect results on hardware when N=16 and valid_len <= 8
 * - On simulator, ALL N values fail
 * 
 * Expected behavior after TFILLPAD_INPLACE:
 * - Columns [0, validLen): original data preserved
 * - Columns [validLen, N): filled with -inf (PadValue::Min)
 */

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace PtoTestCommon;

// Forward declarations
template <int32_t testKey>
void launchTFILLPAD_ISSUE171(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

class TFILLPADIssue171Test : public testing::Test {
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

#define LOGSIZE 128
#define MAXBLOCK 64

template <int32_t testKey, int M, int N, int validLen>
void tfillpad_issue171_test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int byteSize = M * N * sizeof(float);

    void *dstHost, *srcHost, *goldHost;
    void *dstDevice, *srcDevice;
    void *logDevice;

    aclrtMallocHost((void **)(&srcHost), byteSize);
    aclrtMallocHost((void **)(&dstHost), byteSize);
    aclrtMallocHost((void **)(&goldHost), byteSize);

    aclrtMalloc((void **)&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Read pre-generated input and golden data
    size_t fileSize = byteSize;
    ReadFile(GetGoldenDir() + "/input.bin", fileSize, srcHost, fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, goldHost, fileSize);
    cout << "Loaded data from: " << GetGoldenDir() << endl;
    cout << "M=" << M << ", N=" << N << ", validLen=" << validLen << endl;
    cout << "Pad columns: [" << validLen << ", " << N << ") should be -inf" << endl;
    
    std::fill((uint8_t *)dstHost, ((uint8_t *)(dstHost)) + byteSize, 0);

    aclrtMemcpy(srcDevice, byteSize, srcHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, byteSize, dstHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t logHost[MAXBLOCK][LOGSIZE];
    std::fill((uint8_t *)logHost, ((uint8_t *)(logHost)) + sizeof(logHost), 0);
    aclrtMalloc((void **)&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);

    launchTFILLPAD_ISSUE171<testKey>((uint8_t *)dstDevice, (uint8_t *)srcDevice, (uint64_t *)logDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Save output for debugging
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, byteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(logDevice);

    // Compare results - focus on padding columns
    float *goldF = reinterpret_cast<float*>(goldHost);
    float *outF = reinterpret_cast<float*>(dstHost);
    
    int validColErrors = 0;
    int padColErrors = 0;
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    
    cout << "\n=== Checking Padding Columns (Issue #171 Focus) ===" << endl;
    
    // Check each row
    for (int row = 0; row < M; row++) {
        // Check valid columns [0, validLen)
        for (int col = 0; col < validLen; col++) {
            int idx = row * N + col;
            if (outF[idx] != goldF[idx]) {
                if (validColErrors < 5) {
                    cout << "Valid col error [" << row << "," << col << "]: "
                         << "expected=" << goldF[idx] << " actual=" << outF[idx] << endl;
                }
                validColErrors++;
            }
        }
        
        // Check pad columns [validLen, N) - these should all be -inf
        for (int col = validLen; col < N; col++) {
            int idx = row * N + col;
            bool isNegInf = std::isinf(outF[idx]) && outF[idx] < 0;
            if (!isNegInf) {
                if (padColErrors < 10) {
                    cout << "PAD ERROR row " << row << " col " << col << ": "
                         << "expected=-inf actual=" << outF[idx] << endl;
                }
                padColErrors++;
            }
        }
    }
    
    cout << "\n=== Summary ===" << endl;
    cout << "Valid columns [0," << validLen << ") errors: " << validColErrors << endl;
    cout << "Pad columns [" << validLen << "," << N << ") errors: " << padColErrors 
         << " (should be -inf)" << endl;
    
    // Print first few rows for visualization
    cout << "\n=== First 4 rows of output ===" << endl;
    for (int row = 0; row < min(4, M); row++) {
        cout << "Row " << setw(2) << row << ": ";
        for (int col = 0; col < N; col++) {
            float val = outF[row * N + col];
            if (std::isinf(val) && val < 0) {
                cout << setw(8) << "-inf";
            } else {
                cout << setw(8) << fixed << setprecision(0) << val;
            }
            if (col == validLen - 1) cout << " |";  // Mark boundary
        }
        cout << endl;
    }

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Test passes if no errors in padding columns
    bool padOk = (padColErrors == 0);
    bool validOk = (validColErrors == 0);
    
    if (!padOk) {
        cout << "\n*** ISSUE #171 REPRODUCED: Padding columns not filled with -inf! ***" << endl;
    }
    
    EXPECT_EQ(padColErrors, 0) << "Padding columns should all be -inf";
    EXPECT_EQ(validColErrors, 0) << "Valid columns should preserve original data";
}

// Test cases based on Issue #171 analysis:
// valid_len in [1,8] triggers Path B (buggy on HW N=16, buggy on SIM all N)
// valid_len in [9,15] triggers Path B NO-OP (should work)

TEST_F(TFILLPADIssue171Test, case_float_16x16_validlen_1_SHOULD_FAIL)
{
    tfillpad_issue171_test<1, 16, 16, 1>();
}

TEST_F(TFILLPADIssue171Test, case_float_16x16_validlen_8_SHOULD_FAIL)
{
    tfillpad_issue171_test<2, 16, 16, 8>();
}

TEST_F(TFILLPADIssue171Test, case_float_16x16_validlen_9_SHOULD_PASS)
{
    tfillpad_issue171_test<3, 16, 16, 9>();
}

TEST_F(TFILLPADIssue171Test, case_float_16x16_validlen_15_SHOULD_PASS)
{
    tfillpad_issue171_test<4, 16, 16, 15>();
}

// Issue #171 exact cases: N=32/64/128

TEST_F(TFILLPADIssue171Test, case_float_16x32_validlen_1)
{
    tfillpad_issue171_test<5, 16, 32, 1>();
}

TEST_F(TFILLPADIssue171Test, case_float_16x64_validlen_1)
{
    tfillpad_issue171_test<6, 16, 64, 1>();
}

TEST_F(TFILLPADIssue171Test, case_float_16x128_validlen_1)
{
    tfillpad_issue171_test<7, 16, 128, 1>();
}
