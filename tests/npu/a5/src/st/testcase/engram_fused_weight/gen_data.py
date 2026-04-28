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
np.random.seed(19)
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16


def gen_golden_data(param):
    input_dtype = param.input_dtype
    output_dtype = param.output_dtype

    tile_row, tile_col = param.tile_row, param.tile_col

    input1 = np.random.uniform(0.1, 1.0, size=[tile_row, tile_col]).astype(input_dtype)
    input2 = np.random.uniform(0.1, 1.0, size=[tile_row, tile_col]).astype(input_dtype)
    
    if input_dtype == output_dtype:
        golden = input1 * input2
    else:
        input1_fp32 = input1.astype(output_dtype)
        input2_fp32 = input2.astype(output_dtype)
        golden = input1_fp32 * input2_fp32

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class EngramFusedWeightParams:
    def __init__(self, input_dtype, output_dtype, tile_row, tile_col):
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.tile_row = tile_row
        self.tile_col = tile_col


def generate_case_name(param):
    input_dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
    }.get(param.input_dtype, 'bf16')
    
    output_dtype_str = {
        np.float32: 'float',
    }.get(param.output_dtype, 'float')
    
    return f"EngramFusedWeightTest.case_{input_dtype_str}_{output_dtype_str}_{param.tile_row}x{param.tile_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        EngramFusedWeightParams(bfloat16, np.float32, 4, 4096),
        EngramFusedWeightParams(bfloat16, np.float32, 4, 7168),
        EngramFusedWeightParams(np.float32, np.float32, 4, 7168),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)