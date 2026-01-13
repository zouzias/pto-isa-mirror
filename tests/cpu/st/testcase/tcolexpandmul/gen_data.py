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

np.random.seed(2026)


class TColExpandParams:
    def __init__(self, dtype, dst_row, dst_col, src1_row, src1_col):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src1_row = src1_row
        self.src1_col = src1_col


def gen_case_dir_name(param: TColExpandParams) -> str:
    dtype_str = {
        np.float32: "fp32",
        np.float16: "fp16",
    }[param.dtype]
    return f"TCOLEXPANDMULTest.case_{dtype_str}_{param.dst_row}_{param.dst_col}_{param.src1_row}_{param.src1_col}"


def gen_golden_data(param: TColExpandParams):
    dst_row, dst_col = param.dst_row, param.dst_col
    src1_row, src1_col = param.src1_row, param.src1_col
    dtype = param.dtype

    src0 = np.random.uniform(low=-32, high=32, size=(dst_row, dst_col)).astype(dtype)
    src1 = np.random.uniform(low=-8, high=8, size=(src1_row, src1_col)).astype(dtype)

    reps = (dst_col + src1_col - 1) // src1_col
    src1_expand = np.tile(src1, (1, reps))[:, :dst_col]
    golden = src0 * src1_expand

    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    case_params_list = [
        TColExpandParams(np.float32, 32, 64, 1, 64),
        TColExpandParams(np.float32, 8, 32, 1, 32),
        TColExpandParams(np.float16, 16, 64, 1, 64),
        TColExpandParams(np.float16, 4, 128, 1, 128),
    ]

    for param in case_params_list:
        case_name = gen_case_dir_name(param)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)

