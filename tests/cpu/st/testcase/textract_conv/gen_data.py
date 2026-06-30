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

PRINT_C_CASE = True
SHIFT_BLOCK_LEN = 16
SHIFT_BLOCK_BYTE = 32

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def type2str(t):
    if t is np.float16:
        return "half"
    if t is np.float32:
        return "float"
    return np.dtype(t).name + "_t" 

class TextractParams:
    def __init__(
            self, 
            dtype,
            shape_0,
            shape_1,
            shape_2,
            shape_3,
            dst_row, 
            dst_col, 
            idxR,
            idxC):
        self.dtype = dtype
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.shape_3 = shape_3
        self.dst_row = dst_row 
        self.dst_col = dst_col  
        self.idxR = idxR
        self.idxC = idxC


def gen_golden_data(param : TextractParams):
    dtype = param.dtype
    c1hw = param.shape_0
    n1 = param.shape_1
    n0 = param.shape_2
    c0 = param.shape_3
    dst_row = param.dst_row
    dst_col =  param.dst_col
    idxR = param.idxR
    idxC = param.idxC

    input1 = np.random.randint(1, 5, size=(c1hw, n1, c0, n0)).astype(dtype)
    # input1.tofile("input.bin")

    dtype_size = np.dtype(dtype).itemsize
    dst_row_aligned = (dst_row * dtype_size) // SHIFT_BLOCK_BYTE
    dst_col_aligned = dst_col // SHIFT_BLOCK_LEN
    idx_row_aligned = (idxR * dtype_size) // SHIFT_BLOCK_BYTE
    idx_col_aligned = idxC // SHIFT_BLOCK_LEN

    print(f"Input shape : {input1.shape}")
    print(f"{dst_row_aligned}-{dst_col_aligned}|{idx_row_aligned}-{idx_col_aligned}")

    output = input1[idx_row_aligned:(idx_row_aligned + dst_row_aligned):][idx_col_aligned:(idx_col_aligned + dst_col_aligned):]
    print(f"Output shape : {output.shape}")
    print()
    print()


def gen_case_name(i):
    return f"case_{i}"


if __name__ == "__main__":
    case_params_list = [
        TextractParams(np.float16, 4, 3, 16, 16,   3*16, 2*16,   16, 16),
        TextractParams(np.uint16, 4, 3, 16, 16,   3*16, 2*16,   16, 16),
        TextractParams(np.float32, 4, 3, 16, 8,   3*8, 2*16,   8, 16),
        TextractParams(np.int32, 4, 3, 16, 8,   3*8, 2*16,   8, 16)
    ]

    for i in range(len(case_params_list)):
        param = case_params_list[i]
        case_name = gen_case_name(i)
        full_name = "TEXTRACTTest." + case_name
        if not os.path.exists(full_name):
            os.makedirs(full_name)
        original_dir = os.getcwd()
        os.chdir(full_name)

        gen_golden_data(param)

        os.chdir(original_dir)

