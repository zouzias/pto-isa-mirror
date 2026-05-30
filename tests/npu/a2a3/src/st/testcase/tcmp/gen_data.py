#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np


def gen_golden_data_tcmp(param):
    dtype = param.dtype

    src0_row, src0_col, src1_row, src1_col = param.src0_row, param.src0_col, param.src1_row, param.src1_col
    dst_row, dst_col = param.dst_row, param.dst_col
    valid_row, valid_col = param.valid_row, param.valid_col

    # Generate random input arrays
    if dtype == np.int16 or dtype == np.int32:
        input1 = np.random.randint(-10, 10, size=[src0_row, src0_col]).astype(dtype)
        input2 = np.random.randint(-10, 10, size=[src1_row, src1_col]).astype(dtype)
        input1 = np.full((src0_row, src0_col), 8).astype(dtype)
        input2 = np.full((src1_row, src1_col), 8).astype(dtype)
    else:
        input1 = np.random.uniform(-10, 10, size=[src0_row, src0_col]).astype(dtype)
        input2 = np.random.uniform(-10, 10, size=[src1_row, src1_col]).astype(dtype)
    
    inputV1 = input1[:valid_row, :valid_col]
    inputV2 = input2[:valid_row, :valid_col]

    if param.mode == "EQ":
        bool_result = np.isclose(inputV1, inputV2, rtol=0, atol=1e-9)
    elif param.mode == "NE":
        bool_result = ~np.isclose(inputV1, inputV2, rtol=0, atol=1e-9)
    elif param.mode == "LT":
        bool_result = (inputV1 < inputV2)
    elif param.mode == "GT":
        bool_result = (inputV1 > inputV2)
    elif param.mode == "GE":
        bool_result = (inputV1 >= inputV2)
    elif param.mode == "LE":
        bool_result = (inputV1 <= inputV2)
    bool_result = np.packbits(bool_result, axis=1, bitorder='little')

    # Apply valid region constraints
    golden = np.zeros((dst_row, dst_col), dtype=np.uint8)
    golden[:bool_result.shape[0], :bool_result.shape[1]] = bool_result

    # Save the input and bool_result data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TcmpParams:
    DTYPE_STR = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16'
    }
    def __init__(self, dtype, dst_row, dst_col, src0_row, src0_col, src1_row, src1_col, valid_row, valid_col, cmp_mode):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.mode = cmp_mode
        dtype_str = self.__class__.DTYPE_STR[dtype]
        self.name = f"TCMPTest.case_{dtype_str}_{dst_row}x{dst_col}_{src0_row}x{src0_col}_{src1_row}x{src1_col}_"\
            f"{valid_row}x{valid_col}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [ # Comment out test cases that do not handle size corectly
        TcmpParams(np.float16, 32, 32, 32, 32, 32, 32, 32, 32, "EQ"),
        TcmpParams(np.float32, 8, 64, 8, 64, 8, 64, 8, 64, "GT"),
        TcmpParams(np.int32, 4, 64, 4, 64, 4, 64, 4, 64, "EQ"),
        TcmpParams(np.int32, 128, 128, 128, 128, 128, 128, 64, 64, "EQ"),
        TcmpParams(np.int32, 64, 64, 64, 64, 64, 64, 32, 32, "EQ"),
        TcmpParams(np.int32, 16, 32, 16, 32, 16, 32, 16, 32, "EQ"),
        TcmpParams(np.float32, 128, 128, 128, 128, 128, 128, 64, 64, "LE"),
        TcmpParams(np.int32, 77, 32, 77, 80, 77, 80, 32, 32, "EQ"),
        TcmpParams(np.int32, 32, 32, 32, 32, 32, 32, 32, 32, "EQ"),
        TcmpParams(np.int32, 2, 32, 2, 88, 2, 80, 2, 64, "EQ"),
        TcmpParams(np.int32, 66, 32, 66, 88, 66, 80, 66, 64, "EQ"),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data_tcmp(case)
        os.chdir(original_dir)
