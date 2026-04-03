#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)


def gen_golden_data_taddsubmuldiv(case_name, param):
    dtype = param.dtype

    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input3 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input4 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input5 = np.random.randint(1, 10, size=[h, w]).astype(dtype)

    # Perform fused operation: ((src0 + src1) - src2) * src3 / src4
    golden = ((input1 + input2) - input3) * input4 / input5

    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    input3.tofile("input3.bin")
    input4.tofile("input4.bin")
    input5.tofile("input5.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, input3, input4, input5, golden


class TAddSubMulDivParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.gm_row = gm_row
        self.gm_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TAddSubMulDivTest.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get absolute path of script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TAddSubMulDivParams(np.float32, 64, 64, 64, 64, 64, 64),
        TAddSubMulDivParams(np.int32, 64, 64, 64, 64, 64, 64),
        TAddSubMulDivParams(np.int16, 64, 64, 64, 64, 64, 64),
        TAddSubMulDivParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_taddsubmuldiv(case_name, param)
        os.chdir(original_dir)