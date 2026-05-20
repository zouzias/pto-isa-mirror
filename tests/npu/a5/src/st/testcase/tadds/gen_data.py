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
    if np.issubdtype(data_type, np.integer):
        value_max = np.iinfo(data_type).max
        value_min = np.iinfo(data_type).min
    else:
        value_max = np.finfo(data_type).max
        value_min = np.finfo(data_type).min

    input_arr = np.random.uniform(low=value_min, high=value_max, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=value_min, high=value_max, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            output_arr[i, j] = input_arr[i, j] + divider[0, 0]

    input_arr.tofile('input.bin')
    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class taddsParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col

if __name__ == "__main__":
    case_params_list = [
        taddsParams("TADDSTest.case1", np.float32, 32, 768, 32, 768),
        taddsParams("TADDSTest.case2", np.float32, 768, 32, 768, 32),
        taddsParams("TADDSTest.case3", np.float32, 128, 192, 128, 192),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)