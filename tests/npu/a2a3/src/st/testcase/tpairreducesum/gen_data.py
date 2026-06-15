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


def _input_value(input1, row, col, h_valid, w_valid):
    if row < h_valid and col < w_valid and row < input1.shape[0] and col < input1.shape[1]:
        return input1[row, col]
    return 0


def _compute_golden(input1, h_valid, w_valid):
    golden = np.zeros((h_valid, w_valid), dtype=input1.dtype)
    lower_half_cols = w_valid // 2
    for row in range(h_valid):
        for out_col in range(lower_half_cols):
            left = _input_value(input1, row, out_col * 2, h_valid, w_valid)
            right = _input_value(input1, row, out_col * 2 + 1, h_valid, w_valid)
            golden[row, out_col] = left + right
    return golden


def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    input1 = np.random.randint(1, 10, size=[src0_tile_row, src0_tile_col]).astype(dtype)

    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:h_valid, 0:w_valid] = _compute_golden(input1, h_valid, w_valid)

    input1.tofile("input1.bin")
    golden.tofile("golden.bin")


class TPairReduceSumParams:
    def __init__(self, dtype, dstH, dstW, src0H, src0W, vRow, vCol):
        self.dtype = dtype
        self.dst_tile_row = dstH
        self.dst_tile_col = dstW
        self.src0_tile_row = src0H
        self.src0_tile_col = src0W
        self.valid_row = vRow
        self.valid_col = vCol


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
    }[param.dtype]
    return f"TPAIRREDUCESUMTest.case_{dtype_str}_{param.dst_tile_row}x{param.dst_tile_col}_\
{param.src0_tile_row}x{param.src0_tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPairReduceSumParams(np.float32, 64, 128, 64, 128, 64, 128),
        TPairReduceSumParams(np.float32, 64, 64, 64, 64, 64, 64),
        TPairReduceSumParams(np.float16, 16, 256, 16, 256, 16, 256),
        TPairReduceSumParams(np.float16, 16, 64, 16, 128, 16, 64),
        TPairReduceSumParams(np.float32, 16, 32, 16, 64, 16, 32),
        TPairReduceSumParams(np.float16, 16, 64, 16, 128, 16, 63),
        TPairReduceSumParams(np.float32, 16, 32, 16, 64, 16, 31),
        TPairReduceSumParams(np.float16, 2, 128, 2, 128, 1, 106),
        TPairReduceSumParams(np.float32, 8, 128, 8, 128, 8, 127),
        TPairReduceSumParams(np.float16, 8, 256, 8, 256, 8, 255),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
