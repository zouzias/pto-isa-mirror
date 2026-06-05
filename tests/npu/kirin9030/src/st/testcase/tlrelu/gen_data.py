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
import struct
import ctypes
import numpy as np

np.random.seed(23)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col

    input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=-8, high=8, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            if input_arr[i, j] > 0:
                output_arr[i, j] = input_arr[i, j]
            else:
                output_arr[i, j] = input_arr[i, j] * divider[0, 0]

    input_arr.tofile("input.bin")
    with open("divider.bin", "wb") as f:
        f.write(struct.pack("f", np.float32(divider[0, 0])))
    output_arr.tofile("golden.bin")


class TLReluParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TLReluParams("TLRELUTest.case1", np.float32, 32, 128, 32, 64),
        TLReluParams("TLRELUTest.case2", np.float16, 63, 128, 63, 64),
        TLReluParams("TLRELUTest.case3", np.float32, 7, 512, 7, 64 * 7),
        TLReluParams("TLRELUTest.case4", np.float32, 256, 32, 256, 16),
        # Group A: FP32 (block_size=64, align_unit=8) - 8 systematic cases, col 32B aligned (multiple of 8)
        TLReluParams("TLRELUTest.case5", np.float32, 8, 64, 8, 64),
        TLReluParams("TLRELUTest.case6", np.float32, 8, 64, 8, 48),
        TLReluParams("TLRELUTest.case7", np.float32, 8, 64, 8, 56),
        TLReluParams("TLRELUTest.case8", np.float32, 12, 64, 8, 48),
        TLReluParams("TLRELUTest.case9", np.float32, 4, 96, 4, 96),
        TLReluParams("TLRELUTest.case10", np.float32, 4, 96, 4, 72),
        TLReluParams("TLRELUTest.case11", np.float32, 4, 96, 4, 80),
        TLReluParams("TLRELUTest.case12", np.float32, 8, 96, 4, 80),
        # Group B: FP16 (block_size=128, align_unit=16) - 8 systematic cases, col 32B aligned (multiple of 16)
        TLReluParams("TLRELUTest.case13", np.float16, 8, 64, 8, 64),
        TLReluParams("TLRELUTest.case14", np.float16, 8, 64, 8, 48),
        TLReluParams("TLRELUTest.case15", np.float16, 8, 64, 8, 32),
        TLReluParams("TLRELUTest.case16", np.float16, 12, 64, 8, 48),
        TLReluParams("TLRELUTest.case17", np.float16, 2, 144, 2, 144),
        TLReluParams("TLRELUTest.case18", np.float16, 2, 144, 2, 128),
        TLReluParams("TLRELUTest.case19", np.float16, 2, 144, 2, 112),
        TLReluParams("TLRELUTest.case20", np.float16, 4, 144, 2, 112),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
