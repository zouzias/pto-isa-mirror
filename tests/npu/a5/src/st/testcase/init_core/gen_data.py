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


def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    input1 = np.random.randint(1, 10, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[src1_tile_row, src1_tile_col]).astype(dtype)

    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:h_valid, 0:w_valid] = input1[0:h_valid, 0:w_valid] + input2[0:h_valid, 0:w_valid]

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class InitCoreParams:
    def __init__(self, dtype, dst_h, dst_w, src0_h, src0_w, src1_h, src1_w, v_row, v_col):
        self.dtype = dtype
        self.dst_tile_row = dst_h
        self.dst_tile_col = dst_w
        self.src0_tile_row = src0_h
        self.src0_tile_col = src0_w
        self.src1_tile_row = src1_h
        self.src1_tile_col = src1_w
        self.valid_row = v_row
        self.valid_col = v_col


def generate_case_name(param):
    return (
        f"INITCORETest.case_float_{param.dst_tile_row}x{param.dst_tile_col}_"
        f"{param.src0_tile_row}x{param.src0_tile_col}_{param.src1_tile_row}x{param.src1_tile_col}_"
        f"{param.valid_row}x{param.valid_col}"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [InitCoreParams(np.float32, 64, 64, 64, 64, 64, 64, 64, 64)]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
