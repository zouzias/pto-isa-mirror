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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TARGREDUCEOPTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, int rows, int cols>
inline void InitDstDevice(T *dst)
{
    constexpr int size = rows * cols;
    for (int k = 0; k < size; k++) {
        dst[k] = T{0};
    }
}

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMAX(TIdx *out, T *src, void *stream);

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMIN(TIdx *out, T *src, void *stream);

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMAX(TIdx *out, T *src, void *stream);

template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMIN(TIdx *out, T *src, void *stream);


template <typename T, typename TIdx, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols, typename LaunchFn>
void run_vec_op(LaunchFn fn)
{
    const size_t iMatSize = iRow * iCol;
    const size_t oMatSize = oRow * oCol;
    size_t iMatFileSize = iMatSize * sizeof(T);
    size_t oMatFileSize = oMatSize * sizeof(TIdx);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    TIdx *dstHost;
    TIdx *dstDevice;

    T *srcHost;
    T *srcDevice;

    aclrtMallocHost((void **)(&dstHost), oMatFileSize);
    aclrtMallocHost((void **)(&srcHost), iMatFileSize);

    aclrtMalloc((void **)&dstDevice, oMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, iMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    InitDstDevice<TIdx, oRow, oCol>(dstDevice);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", iMatFileSize, srcHost, iMatFileSize));
    aclrtMemcpy(srcDevice, iMatFileSize, srcHost, iMatFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, oMatFileSize, dstDevice, oMatFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, oMatFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(oMatSize);
    std::vector<T> devFinal(oMatSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", oMatFileSize, golden.data(), oMatFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", oMatFileSize, devFinal.data(), oMatFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TARGREDUCEOPTest, case_row_max_float_uint32_64x64_64x64_64x64)
{
    run_vec_op<float, uint32_t, 64, 64>(
        [](uint32_t *out, float *src, void *stream) { LaunchTROWARGMAX<float, uint32_t, 64, 64>(out, src, stream); });
}

TEST_F(TARGREDUCEOPTest, case_row_max_half_int32_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMAX<aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_max_float_uint32_16x16_32x32_64x64)
{
    run_vec_op<float, uint32_t, 16, 16, 32, 32, 64, 64>([](uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMAX<float, uint32_t, 16, 16, 32, 32, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_min_float_uint32_64x64_64x64_64x64)
{
    run_vec_op<float, uint32_t, 64, 64>(
        [](uint32_t *out, float *src, void *stream) { LaunchTROWARGMIN<float, uint32_t, 64, 64>(out, src, stream); });
}

TEST_F(TARGREDUCEOPTest, case_row_min_half_int32_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMIN<aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_min_float_uint32_16x16_32x32_64x64)
{
    run_vec_op<float, uint32_t, 16, 16, 32, 32, 64, 64>([](uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMIN<float, uint32_t, 16, 16, 32, 32, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_max_float_uint32_64x64_64x64_64x64)
{
    run_vec_op<float, uint32_t, 64, 64>(
        [](uint32_t *out, float *src, void *stream) { LaunchTCOLARGMAX<float, uint32_t, 64, 64>(out, src, stream); });
}

TEST_F(TARGREDUCEOPTest, case_col_max_half_int32_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src, void *stream) {
        LaunchTCOLARGMAX<aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_max_float_uint32_16x16_32x32_64x64)
{
    run_vec_op<float, uint32_t, 16, 16, 32, 32, 64, 64>([](uint32_t *out, float *src, void *stream) {
        LaunchTCOLARGMAX<float, uint32_t, 16, 16, 32, 32, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_min_float_uint32_64x64_64x64_64x64)
{
    run_vec_op<float, uint32_t, 64, 64>(
        [](uint32_t *out, float *src, void *stream) { LaunchTCOLARGMIN<float, uint32_t, 64, 64>(out, src, stream); });
}

TEST_F(TARGREDUCEOPTest, case_col_min_half_int32_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src, void *stream) {
        LaunchTCOLARGMIN<aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_min_float_uint32_16x16_32x32_64x64)
{
    run_vec_op<float, uint32_t, 16, 16, 32, 32, 64, 64>([](uint32_t *out, float *src, void *stream) {
        LaunchTCOLARGMIN<float, uint32_t, 16, 16, 32, 32, 64, 64>(out, src, stream);
    });
}
