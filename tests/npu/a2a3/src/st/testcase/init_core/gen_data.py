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
import numpy as np

np.random.seed(19)


def gen_golden_data_init_core(case_name, param):
    dtype = param.dtype

    height, width = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    input1 = np.random.randint(1, 10, size=[height, width]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[height, width]).astype(dtype)

    golden = input1 + input2
    output = np.zeros([height, width]).astype(dtype)
    for row in range(height):
        for col in range(width):
            if row >= h_valid or col >= w_valid:
                golden[row][col] = output[row][col]

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class InitCoreParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.gm_row = gm_row
        self.gm_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    return f"INITCORETest.case_float_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [InitCoreParams(np.float32, 64, 64, 64, 64, 64, 64)]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_init_core(case_name, param)
        os.chdir(original_dir)
