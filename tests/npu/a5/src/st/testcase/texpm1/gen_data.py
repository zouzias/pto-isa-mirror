#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import math
import numpy as np
np.random.seed(19)


def gen_golden_data(param):
    dst_dtype = param.dst_dtype
    src_dtype = param.src_dtype
    dst_row, dst_col = [param.dst_row, param.dst_col]
    src_row, src_col = [param.src_row, param.src_col]
    valid_row, valid_col = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input_arr = np.random.uniform(-30, 100, size=(src_row, src_col)).astype(src_dtype)
    golden = np.zeros((dst_row, dst_col), dtype=dst_dtype)
    # Perform the operation
    for i in range(valid_row):
        for j in range(valid_col):
            golden[i][j] = math.expm1(input_arr[i][j])

    # Save the input and golden data to binary files
    input_arr.tofile("input.bin")
    golden.tofile("golden.bin")

class tunaryParams:
    def __init__(self, name, dst_dtype, src_dtype, dst_row, dst_col, src_row, src_col, valid_row, valid_col):
        self.name = name
        self.dst_dtype = dst_dtype
        self.src_dtype = src_dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src_row = src_row
        self.src_col = src_col
        self.valid_row = valid_row
        self.valid_col = valid_col


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tunaryParams("TEXPM1Test.case1", np.float32, np.float32, 16, 64, 16, 64, 16, 64),
        tunaryParams("TEXPM1Test.case2", np.float32, np.float32, 1, 1024, 1, 1024, 1, 1024),
        tunaryParams("TEXPM1Test.case3", np.float16, np.float16, 32, 32, 32, 32, 32, 32),
        tunaryParams("TEXPM1Test.case4", np.float32, np.float16, 32, 32, 32, 32, 32, 32),
        tunaryParams("TEXPM1Test.case5", np.float32, np.int32, 32, 32, 32, 32, 32, 32),
        tunaryParams("TEXPM1Test.case6", np.float32, np.int16, 32, 32, 32, 32, 32, 32),
        tunaryParams("TEXPM1Test.case7", np.float16, np.int16, 32, 32, 32, 32, 32, 32),
    ]

    for _, param in enumerate(case_params_list):
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data(param)
        os.chdir(original_dir)