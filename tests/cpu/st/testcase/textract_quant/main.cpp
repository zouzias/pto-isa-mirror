/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;
using namespace pto;

class TEXTRACTTest : public testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols,
          size_t dstValidRows, size_t dstValidCols, size_t idxRow, size_t idxCol, bool applyRelu>
void runTEXTRACT_Scalar()
{
    size_t srcFileSize = validRows * validCols * sizeof(ST);
    size_t dstFileSize = dstValidRows * dstValidCols * sizeof(DT);
    size_t quantFileSize = sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost, *quantHost;
    uint8_t *dstDevice, *srcDevice, *quantDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&quantHost), quantFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    using GlobalDataSrc = GlobalTensor<ST, pto::Shape<1, 1, 1, validRows, validCols>,
        pto::Stride<1 * validRows * validCols, 1 * validRows * validCols, validRows * validCols, validCols, 1>>;
    using GlobalDataDst = GlobalTensor<DT, pto::Shape<1, 1, 1, dstValidRows, dstValidCols>,
        pto::Stride<1 * dstValidRows * dstValidCols, 1 * dstValidRows * dstValidCols, dstValidRows * dstValidCols, dstValidCols, 1>>;

    GlobalDataSrc srcGlobal((ST *)srcDevice);
    GlobalDataDst dstGlobal((DT *)dstDevice);

    Tile<TileType::Mat, ST, rows, cols, BLayout::RowMajor, validRows, validCols, SLayout::NoneBox, 512> srcTile;
    Tile<TileType::Mat, DT, rows, cols, BLayout::RowMajor, dstValidRows, dstValidCols, SLayout::NoneBox, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    std::fill(dstTile.data(), dstTile.data() + rows * cols, 0);

    size_t inputSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, srcFileSize));
    size_t quantSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/quant.bin", quantSize, quantHost, quantFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantDevice, quantFileSize, quantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    uint64_t scalarQuant = ((uint64_t *)quantDevice)[0];
    constexpr ReluPreMode reluMode = applyRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT<DT, ST, reluMode>(dstTile, srcTile, scalarQuant, static_cast<uint16_t>(idxRow), static_cast<uint16_t>(idxCol));

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    std::vector<DT> golden(dstFileSize / sizeof(DT));
    size_t goldenSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), dstFileSize));

    bool ret = ResultCmp(golden, (DT *)dstHost, 0);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(quantDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(quantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols,
          size_t dstValidRows, size_t dstValidCols, size_t idxRow, size_t idxCol, bool applyRelu>
void runTEXTRACT_Vector()
{
    size_t srcFileSize = validRows * validCols * sizeof(ST);
    size_t dstFileSize = dstValidRows * dstValidCols * sizeof(DT);
    size_t quantFileSize = dstValidCols * sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost, *quantHost;
    uint8_t *dstDevice, *srcDevice, *quantDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&quantHost), quantFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    using GlobalDataSrc = GlobalTensor<ST, pto::Shape<1, 1, 1, validRows, validCols>,
        pto::Stride<1 * validRows * validCols, 1 * validRows * validCols, validRows * validCols, validCols, 1>>;
    using GlobalDataDst = GlobalTensor<DT, pto::Shape<1, 1, 1, dstValidRows, dstValidCols>,
        pto::Stride<1 * dstValidRows * dstValidCols, 1 * dstValidRows * dstValidCols, dstValidRows * dstValidCols, dstValidCols, 1>>;

    GlobalDataSrc srcGlobal((ST *)srcDevice);
    GlobalDataDst dstGlobal((DT *)dstDevice);

    Tile<TileType::Mat, ST, rows, cols, BLayout::RowMajor, validRows, validCols, SLayout::NoneBox, 512> srcTile;
    Tile<TileType::Mat, DT, rows, cols, BLayout::RowMajor, dstValidRows, dstValidCols, SLayout::NoneBox, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    std::fill(dstTile.data(), dstTile.data() + rows * cols, 0);

    size_t inputSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, srcFileSize));
    size_t quantSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/quant.bin", quantSize, quantHost, quantFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantDevice, quantFileSize, quantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    using GlobalDataFp = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, dstValidCols>,
        pto::Stride<1 * dstValidCols, dstValidCols, dstValidCols, dstValidCols, 1>>;
    GlobalDataFp fpGlobal((uint64_t *)quantDevice);

    Tile<TileType::Mat, uint64_t, 1, dstValidCols, BLayout::RowMajor, 1, dstValidCols, SLayout::NoneBox, 512> fpTileLocal;
    TASSIGN(fpTileLocal, 0x30000);

    TLOAD(srcTile, srcGlobal);
    TLOAD(fpTileLocal, fpGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    constexpr ReluPreMode reluMode = applyRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT_FP<DT, ST, uint64_t, reluMode>(dstTile, srcTile, fpTileLocal, static_cast<uint16_t>(idxRow), static_cast<uint16_t>(idxCol));

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    std::vector<DT> golden(dstFileSize / sizeof(DT));
    size_t goldenSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), dstFileSize));

    bool ret = ResultCmp(golden, (DT *)dstHost, 0);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(quantDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(quantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case_1_int32_t_int8_t) { runTEXTRACT_Scalar<int32_t, int8_t, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_2_int32_t_int8_t) { runTEXTRACT_Scalar<int32_t, int8_t, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_3_int32_t_int8_t) { runTEXTRACT_Scalar<int32_t, int8_t, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_4_int32_t_int8_t) { runTEXTRACT_Scalar<int32_t, int8_t, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_5_int32_t_int8_t) { runTEXTRACT_Vector<int32_t, int8_t, 128, 64, 128, 64, 64, 32, 8, 0, false>(); }
TEST_F(TEXTRACTTest, case_6_int32_t_int8_t) { runTEXTRACT_Vector<int32_t, int8_t, 96, 96, 96, 96, 64, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_7_int32_t_int8_t) { runTEXTRACT_Vector<int32_t, int8_t, 128, 128, 128, 128, 96, 96, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_8_int32_t_int8_t) { runTEXTRACT_Vector<int32_t, int8_t, 256, 64, 256, 64, 128, 32, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_9_int32_t_uint8_t) { runTEXTRACT_Scalar<int32_t, uint8_t, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_10_int32_t_uint8_t) { runTEXTRACT_Scalar<int32_t, uint8_t, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_11_int32_t_uint8_t) { runTEXTRACT_Scalar<int32_t, uint8_t, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_12_int32_t_uint8_t) { runTEXTRACT_Scalar<int32_t, uint8_t, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_13_int32_t_uint8_t) { runTEXTRACT_Vector<int32_t, uint8_t, 128, 64, 128, 64, 64, 32, 8, 0, false>(); }
TEST_F(TEXTRACTTest, case_14_int32_t_uint8_t) { runTEXTRACT_Vector<int32_t, uint8_t, 96, 96, 96, 96, 64, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_15_int32_t_uint8_t) { runTEXTRACT_Vector<int32_t, uint8_t, 128, 128, 128, 128, 96, 96, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_16_int32_t_uint8_t) { runTEXTRACT_Vector<int32_t, uint8_t, 256, 64, 256, 64, 128, 32, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_17_int32_t_half) { runTEXTRACT_Scalar<int32_t, half, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_18_int32_t_half) { runTEXTRACT_Scalar<int32_t, half, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_19_int32_t_half) { runTEXTRACT_Scalar<int32_t, half, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_20_int32_t_half) { runTEXTRACT_Scalar<int32_t, half, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_21_int32_t_half) { runTEXTRACT_Vector<int32_t, half, 128, 64, 128, 64, 64, 32, 8, 0, false>(); }
TEST_F(TEXTRACTTest, case_22_int32_t_half) { runTEXTRACT_Vector<int32_t, half, 96, 96, 96, 96, 64, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_23_int32_t_half) { runTEXTRACT_Vector<int32_t, half, 128, 128, 128, 128, 96, 96, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_24_int32_t_half) { runTEXTRACT_Vector<int32_t, half, 256, 64, 256, 64, 128, 32, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_25_float_int8_t) { runTEXTRACT_Scalar<float, int8_t, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_26_float_int8_t) { runTEXTRACT_Scalar<float, int8_t, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_27_float_int8_t) { runTEXTRACT_Scalar<float, int8_t, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_28_float_int8_t) { runTEXTRACT_Scalar<float, int8_t, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_29_float_int8_t) { runTEXTRACT_Vector<float, int8_t, 128, 64, 128, 64, 64, 32, 8, 0, false>(); }
TEST_F(TEXTRACTTest, case_30_float_int8_t) { runTEXTRACT_Vector<float, int8_t, 96, 96, 96, 96, 64, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_31_float_int8_t) { runTEXTRACT_Vector<float, int8_t, 128, 128, 128, 128, 96, 96, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_32_float_int8_t) { runTEXTRACT_Vector<float, int8_t, 256, 64, 256, 64, 128, 32, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_33_float_uint8_t) { runTEXTRACT_Scalar<float, uint8_t, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_34_float_uint8_t) { runTEXTRACT_Scalar<float, uint8_t, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_35_float_uint8_t) { runTEXTRACT_Scalar<float, uint8_t, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_36_float_uint8_t) { runTEXTRACT_Scalar<float, uint8_t, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_37_float_uint8_t) { runTEXTRACT_Vector<float, uint8_t, 128, 64, 128, 64, 64, 32, 8, 0, false>(); }
TEST_F(TEXTRACTTest, case_38_float_uint8_t) { runTEXTRACT_Vector<float, uint8_t, 96, 96, 96, 96, 64, 64, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_39_float_uint8_t) { runTEXTRACT_Vector<float, uint8_t, 128, 128, 128, 128, 96, 96, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_40_float_uint8_t) { runTEXTRACT_Vector<float, uint8_t, 256, 64, 256, 64, 128, 32, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_41_float_half) { runTEXTRACT_Scalar<float, half, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_42_float_half) { runTEXTRACT_Scalar<float, half, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_43_float_half) { runTEXTRACT_Scalar<float, half, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_44_float_half) { runTEXTRACT_Scalar<float, half, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }

TEST_F(TEXTRACTTest, case_45_float_bfloat16_t) { runTEXTRACT_Scalar<float, bfloat16_t, 128, 64, 128, 64, 128, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_46_float_bfloat16_t) { runTEXTRACT_Scalar<float, bfloat16_t, 128, 64, 128, 64, 96, 32, 0, 0, true>(); }
TEST_F(TEXTRACTTest, case_47_float_bfloat16_t) { runTEXTRACT_Scalar<float, bfloat16_t, 128, 128, 128, 128, 64, 64, 0, 0, false>(); }
TEST_F(TEXTRACTTest, case_48_float_bfloat16_t) { runTEXTRACT_Scalar<float, bfloat16_t, 256, 128, 256, 128, 128, 64, 0, 0, true>(); }