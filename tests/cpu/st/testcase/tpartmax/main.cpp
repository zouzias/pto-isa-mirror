// --------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "common/common.h"
#include "kernel.h"

template <typename T, int T_GROWS1, int T_GCOLS1, int T_ROWS1, int T_COLS1,
          int T_GROWS2, int T_GCOLS2, int T_ROWS2, int T_COLS2,
          int T_ROWS3, int T_COLS3>
TPartMaxTest() : TestCase("tpartmax")
{
    int row1 = T_ROWS1;
    int col1 = T_COLS1;
    int global_row1 = T_GROWS1;
    int global_col1 = T_GCOLS1;
    int tile_row1 = T_ROWS1;
    int tile_col1 = T_COLS1;
    int valid_row1 = T_ROWS1;
    int valid_col1 = T_COLS1;

    int row2 = T_ROWS2;
    int col2 = T_COLS2;
    int global_row2 = T_GROWS2;
    int global_col2 = T_GCOLS2;
    int tile_row2 = T_ROWS2;
    int tile_col2 = T_COLS2;
    int valid_row2 = T_ROWS2;
    int valid_col2 = T_COLS2;

    int row3 = T_ROWS3;
    int col3 = T_COLS3;

    std::string data_dir = GetTestCaseDir();
    std::string input1_path = data_dir + "/input1.bin";
    std::string golden_path = data_dir + "/golden.bin";

    T *input1 = nullptr;
    T *input2 = nullptr;
    T *golden = nullptr;
    T *dst = nullptr;
    T *dst_golden = nullptr;
    T *dst_out = nullptr;

    size_t input_size1 = global_row1 * global_col1 * sizeof(T);
    size_t input_size2 = global_row2 * global_col2 * sizeof(T);
    size_t golden_size = row3 * col3 * sizeof(T);

    if (ReadBinFile(input1_path.c_str(), input_size1, (void **)&input1) != 0) {
        std::cerr << "ERROR: ReadBinFile failed" << std::endl;
        throw std::runtime_error("ReadBinFile failed");
    }

    if (ReadBinFile(golden_path.c_str(), golden_size, (void **)&golden) != 0) {
        std::cerr << "ERROR: ReadBinFile failed" << std::endl;
        throw std::runtime_error("ReadBinFile failed");
    }

    dst = reinterpret_cast<T *>(malloc(input_size1 + input_size2));
    dst_out = reinterpret_cast<T *>(malloc(golden_size));
    dst_golden = reinterpret_cast<T *>(malloc(golden_size));

    if (dst == nullptr || dst_out == nullptr || dst_golden == nullptr) {
        std::cerr << "ERROR: Malloc failed" << std::endl;
        throw std::runtime_error("Malloc failed");
    }

    memset(dst, 0, input_size1 + input_size2);
    memset(dst_out, 0, golden_size);
    memset(dst_golden, 0, golden_size);

    input2 = reinterpret_cast<T *>(dst) + global_row1 * global_col1;
    for (int i = 0; i < global_row1 * global_col1; i++) {
        dst[i] = input1[i];
    }

    for (int i = 0; i < global_row2 * global_col2; i++) {
        input2[i] = static_cast<T>(rand() % 100);
    }

    LaunchTPARTMAX<T>(dst, dst_out, global_row1, global_col1, tile_row1, tile_col1, valid_row1, valid_col1,
                      global_row2, global_col2, tile_row2, tile_col2, valid_row2, valid_col2,
                      row3, col3);

    for (int i = 0; i < row3 * col3; i++) {
        dst_golden[i] = golden[i];
    }

    if (std::is_same<T, float>::value) {
        for (int i = 0; i < row3 * col3; i++) {
            if (fabsf(dst_out[i] - dst_golden[i]) > 1e-3f) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    } else if (std::is_same<T, aclFloat16>::value) {
        for (int i = 0; i < row3 * col3; i++) {
            if (fabsf(Float2Float(dst_out[i]) - Float2Float(dst_golden[i])) > 1e-3f) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    } else {
        for (int i = 0; i < row3 * col3; i++) {
            if (Float2Float(dst_out[i]) - Float2Float(dst_golden[i]) > FLT_EPSILON) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    }

    std::cout << "Test case pass" << std::endl;

    if (input1) {
        free(input1);
        input1 = nullptr;
    }
    if (golden) {
        free(golden);
        golden = nullptr;
    }
    if (dst) {
        free(dst);
        dst = nullptr;
    }
    if (dst_out) {
        free(dst_out);
        dst_out = nullptr;
    }
    if (dst_golden) {
        free(dst_golden);
        dst_golden = nullptr;
    }
}

// FP32 tile shapes
TEST_F(TPartMaxTest, case_float_32x32_32x16_64x64_32x32_32x16)
{
    TPartMaxTest<float, 32, 32, 32, 16, 64, 64, 32, 32, 32, 16>();
}

TEST_F(TPartMaxTest, case_float_128x128_32x32_64x64_32x32_32x32)
{
    TPartMaxTest<float, 128, 128, 32, 32, 64, 64, 32, 32, 32, 32>();
}

TEST_F(TPartMaxTest, case_float_64x32_64x16_96x96_48x48_48x16)
{
    TPartMaxTest<float, 64, 32, 64, 16, 96, 96, 48, 48, 48, 16>();
}

// FP16 tile shapes
TEST_F(TPartMaxTest, case_float16_32x32_32x16_64x64_32x32_32x16)
{
    TPartMaxTest<aclFloat16, 32, 32, 32, 16, 64, 64, 32, 32, 32, 16>();
}

TEST_F(TPartMaxTest, case_float16_128x128_32x32_64x64_32x32_32x32)
{
    TPartMaxTest<aclFloat16, 128, 128, 32, 32, 64, 64, 32, 32, 32, 32>();
}

// BF16 tile shapes
#if defined(CPU_SIM_BFLOAT_ENABLED)
TEST_F(TPartMaxTest, case_bfloat16_32x32_32x16_64x64_32x32_32x16)
{
    TPartMaxTest<bfloat16_t, 32, 32, 32, 16, 64, 64, 32, 32, 32, 16>();
}
#endif

// INT8 tile shapes
TEST_F(TPartMaxTest, case_int8_32x32_32x16_64x64_32x32_32x16)
{
    TPartMaxTest<int8_t, 32, 32, 32, 16, 64, 64, 32, 32, 32, 16>();
}

TEST_F(TPartMaxTest, case_int8_128x128_32x32_64x64_32x32_32x32)
{
    TPartMaxTest<int8_t, 128, 128, 32, 32, 64, 64, 32, 32, 32, 32>();
}

TEST_F(TPartMaxTest, case_int8_64x32_64x16_96x96_48x48_48x16)
{
    TPartMaxTest<int8_t, 64, 32, 64, 16, 96, 96, 48, 48, 48, 16>();
}

TEST_F(TPartMaxTest, case_int8_48x48_24x24_48x48_24x24_24x24)
{
    TPartMaxTest<int8_t, 48, 48, 24, 24, 48, 48, 24, 24, 24, 24>();
}

// UINT8 tile shapes
TEST_F(TPartMaxTest, case_uint8_32x32_32x16_64x64_32x32_32x16)
{
    TPartMaxTest<uint8_t, 32, 32, 32, 16, 64, 64, 32, 32, 32, 16>();
}

TEST_F(TPartMaxTest, case_uint8_128x128_32x32_64x64_32x32_32x32)
{
    TPartMaxTest<uint8_t, 128, 128, 32, 32, 64, 64, 32, 32, 32, 32>();
}

TEST_F(TPartMaxTest, case_uint8_64x32_64x16_96x96_48x48_48x16)
{
    TPartMaxTest<uint8_t, 64, 32, 64, 16, 96, 96, 48, 48, 48, 16>();
}

TEST_F(TPartMaxTest, case_uint8_48x48_24x24_48x48_24x24_24x24)
{
    TPartMaxTest<uint8_t, 48, 48, 24, 24, 48, 48, 24, 24, 24, 24>();
}
