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

template <typename T, int T_GROWS, int T_GCOLS, int T_ROWS, int T_COLS>
TROWMINTest() : TestCase("trowmin")
{
    int row = T_ROWS;
    int col = T_COLS;
    int global_row = T_GROWS;
    int global_col = T_GCOLS;
    int tile_row = T_ROWS;
    int tile_col = T_COLS;
    int valid_row = T_ROWS;
    int valid_col = T_COLS;

    std::string data_dir = GetTestCaseDir();
    std::string input1_path = data_dir + "/input1.bin";
    std::string golden_path = data_dir + "/golden.bin";

    T *input = nullptr;
    T *golden = nullptr;
    T *dst = nullptr;
    T *dst_golden = nullptr;
    T *dst_out = nullptr;

    size_t input_size = global_row * global_col * sizeof(T);
    size_t golden_size = global_row * sizeof(T);

    if (ReadBinFile(input1_path.c_str(), input_size, (void **)&input) != 0) {
        std::cerr << "ERROR: ReadBinFile failed" << std::endl;
        throw std::runtime_error("ReadBinFile failed");
    }

    if (ReadBinFile(golden_path.c_str(), golden_size, (void **)&golden) != 0) {
        std::cerr << "ERROR: ReadBinFile failed" << std::endl;
        throw std::runtime_error("ReadBinFile failed");
    }

    dst = reinterpret_cast<T *>(malloc(input_size));
    dst_out = reinterpret_cast<T *>(malloc(golden_size));
    dst_golden = reinterpret_cast<T *>(malloc(golden_size));

    if (dst == nullptr || dst_out == nullptr || dst_golden == nullptr) {
        std::cerr << "ERROR: Malloc failed" << std::endl;
        throw std::runtime_error("Malloc failed");
    }

    memset(dst, 0, input_size);
    memset(dst_out, 0, golden_size);
    memset(dst_golden, 0, golden_size);

    for (int i = 0; i < global_row * global_col; i++) {
        dst[i] = input[i];
    }

    LaunchTROWMIN<T>(dst, dst_out, global_row, global_col, tile_row, tile_col, valid_row, valid_col);

    for (int i = 0; i < global_row; i++) {
        dst_golden[i] = golden[i];
    }

    if (std::is_same<T, float>::value) {
        for (int i = 0; i < global_row; i++) {
            if (fabsf(dst_out[i] - dst_golden[i]) > 1e-3f) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    } else if (std::is_same<T, aclFloat16>::value) {
        for (int i = 0; i < global_row; i++) {
            if (fabsf(Float2Float(dst_out[i]) - Float2Float(dst_golden[i])) > 1e-3f) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    } else {
        for (int i = 0; i < global_row; i++) {
            if (Float2Float(dst_out[i]) - Float2Float(dst_golden[i]) > FLT_EPSILON) {
                std::cerr << "ERROR: Compare failed: dst_out[" << i << "]=" << dst_out[i]
                          << " dst_golden[" << i << "]=" << dst_golden[i] << std::endl;
                throw std::runtime_error("Compare failed");
            }
        }
    }

    std::cout << "Test case pass" << std::endl;

    if (input) {
        free(input);
        input = nullptr;
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
TEST_F(TROWMINTest, case_float_64x64_64x64)
{
    TROWMINTest<float, 64, 64, 64, 64>();
}

TEST_F(TROWMINTest, case_float_77x81_32x16)
{
    TROWMINTest<float, 77, 81, 32, 16>();
}

TEST_F(TROWMINTest, case_float_32x32_32x16)
{
    TROWMINTest<float, 32, 32, 32, 16>();
}

// FP16 tile shapes
TEST_F(TROWMINTest, case_float16_64x64_64x64)
{
    TROWMINTest<aclFloat16, 64, 64, 64, 64>();
}

TEST_F(TROWMINTest, case_float16_161x161_32x32)
{
    TROWMINTest<aclFloat16, 161, 161, 32, 32>();
}

// BF16 tile shapes
#if defined(CPU_SIM_BFLOAT_ENABLED)
TEST_F(TROWMINTest, case_bfloat16_64x64_64x64)
{
    TROWMINTest<bfloat16_t, 64, 64, 64, 64>();
}
#endif

// INT8 tile shapes
TEST_F(TROWMINTest, case_int8_64x64_64x64)
{
    TROWMINTest<int8_t, 64, 64, 64, 64>();
}

TEST_F(TROWMINTest, case_int8_128x128_32x32)
{
    TROWMINTest<int8_t, 128, 128, 32, 32>();
}

TEST_F(TROWMINTest, case_int8_64x32_32x16)
{
    TROWMINTest<int8_t, 64, 32, 32, 16>();
}

TEST_F(TROWMINTest, case_int8_96x96_48x48)
{
    TROWMINTest<int8_t, 96, 96, 48, 48>();
}

// UINT8 tile shapes
TEST_F(TROWMINTest, case_uint8_64x64_64x64)
{
    TROWMINTest<uint8_t, 64, 64, 64, 64>();
}

TEST_F(TROWMINTest, case_uint8_128x128_32x32)
{
    TROWMINTest<uint8_t, 128, 128, 32, 32>();
}

TEST_F(TROWMINTest, case_uint8_64x32_32x16)
{
    TROWMINTest<uint8_t, 64, 32, 32, 16>();
}

TEST_F(TROWMINTest, case_uint8_96x96_48x48)
{
    TROWMINTest<uint8_t, 96, 96, 48, 48>();
}
