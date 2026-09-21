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


def gen_golden_data(param):
    data_type = param.data_type
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col
    src_tile_row = param.src_tile_row
    src_tile_col = param.src_tile_col
    valid_row = param.valid_row
    valid_col = param.valid_col

    if np.issubdtype(data_type, np.integer):
        value_max = np.iinfo(data_type).max
        value_min = np.iinfo(data_type).min
    else:
        value_max = np.finfo(data_type).max / 100
        value_min = np.finfo(data_type).min / 100

    if data_type == np.int64:
        input_arr = np.random.randint(-1000000, 1000000, size=(src_tile_row, src_tile_col)).astype(data_type)
        if param.use_edge_cases:
            values = np.array(
                [
                    -7,
                    7,
                    -6,
                    6,
                    -1,
                    1,
                    0,
                    np.iinfo(data_type).min,
                    np.iinfo(data_type).max,
                    -(2**53 + 1),
                    2**53 + 1,
                    -(2**32 + 1),
                    2**32 + 1,
                ],
                dtype=data_type,
            )
            input_arr = np.resize(values, (src_tile_row, src_tile_col))
        divider_value = param.scalar if param.scalar is not None else 97
        divider = np.array([divider_value], dtype=data_type)
    elif data_type == np.uint64:
        input_arr = np.random.randint(0, 2000000, size=(src_tile_row, src_tile_col)).astype(data_type)
        values = np.array([0, 1, 7, 2**32 + 1, 2**53 + 1, 2**63 - 1, 2**63, 2**63 + 1, value_max], dtype=data_type)
        if param.use_edge_cases:
            input_arr = np.resize(values, (src_tile_row, src_tile_col))
        else:
            input_arr.flat[: len(values)] = values
        divider_value = param.scalar if param.scalar is not None else 97
        divider = np.array([divider_value], dtype=data_type)
    else:
        input_arr = np.random.uniform(low=value_min, high=value_max, size=(src_tile_row, src_tile_col)).astype(
            data_type
        )
        divider = np.random.uniform(low=value_min, high=value_max, size=1).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    if param.inplace:
        output_arr[:src_tile_row, :src_tile_col] = input_arr[:src_tile_row, :src_tile_col]
    if data_type == np.int64:
        rhs = int(divider[0])
        for i in range(valid_row):
            for j in range(valid_col):
                output_arr[i, j] = -1 if rhs == 0 else int(input_arr[i, j]) % rhs
    elif data_type == np.uint64 and divider[0] == 0:
        output_arr[:valid_row, :valid_col] = np.iinfo(data_type).max
    elif divider[0] == 0:
        output_arr[:valid_row, :valid_col] = 0
    else:
        output_arr[:valid_row, :valid_col] = input_arr[:valid_row, :valid_col] % divider[0]

    input_arr.tofile("input.bin")
    divider.tofile("divider.bin")
    output_arr.tofile("golden.bin")


class TremsParams:
    def __init__(
        self,
        name,
        data_type,
        dst_tile_row,
        dst_tile_col,
        src_tile_row,
        src_tile_col,
        row,
        col,
        scalar=None,
        valid_row=None,
        valid_col=None,
        inplace=False,
        use_edge_cases=False,
    ):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src_tile_row = src_tile_row
        self.src_tile_col = src_tile_col
        self.valid_row = row if valid_row is None else valid_row
        self.valid_col = col if valid_col is None else valid_col
        self.scalar = scalar
        self.inplace = inplace
        self.use_edge_cases = use_edge_cases


if __name__ == "__main__":
    case_params_list = [
        TremsParams("TREMSTest.case1", np.float32, 32, 128, 32, 128, 32, 64),
        TremsParams("TREMSTest.case2", np.float16, 63, 128, 63, 128, 63, 64),
        TremsParams("TREMSTest.case3", np.int32, 31, 256, 31, 256, 31, 128),
        TremsParams("TREMSTest.case4", np.int16, 15, 192, 15, 192, 15, 192),
        TremsParams("TREMSTest.case5", np.float32, 7, 512, 7, 512, 7, 448),
        TremsParams("TREMSTest.case6", np.float32, 256, 32, 256, 32, 256, 31),
        TremsParams("TREMSTest.caseHP1", np.float32, 64, 64, 64, 64, 64, 64),
        TremsParams("TREMSTest.caseHP2", np.float32, 64, 64, 64, 64, 64, 61),
        TremsParams("TREMSTest.case_int64_4x16", np.int64, 4, 16, 4, 16, 4, 16),
        TremsParams("TREMSTest.case_uint64_4x16", np.uint64, 4, 16, 4, 16, 4, 16),
        TremsParams("TREMSTest.case_uint64_zero_divisor_4x16", np.uint64, 4, 16, 4, 16, 4, 16, scalar=0),
        TremsParams("TREMSTest.case_int64_4x64", np.int64, 4, 64, 4, 64, 4, 64),
        TremsParams("TREMSTest.case_uint64_4x64", np.uint64, 4, 64, 4, 64, 4, 64),
        TremsParams("TREMSTest.case_int64_1x10912", np.int64, 1, 10912, 1, 10912, 1, 10912),
        TremsParams("TREMSTest.case_uint64_1x10912", np.uint64, 1, 10912, 1, 10912, 1, 10912),
        TremsParams("TREMSTest.case_int64_4x32_inplace", np.int64, 4, 32, 4, 32, 4, 32, scalar=17, inplace=True),
        TremsParams("TREMSTest.case_uint64_4x32_inplace", np.uint64, 4, 32, 4, 32, 4, 32, scalar=17, inplace=True),
        TremsParams(
            "TREMSTest.case_int64_1x1024_inplace", np.int64, 1, 1024, 1, 1024, 1, 1024, scalar=17, inplace=True
        ),
        TremsParams(
            "TREMSTest.case_int64_4x64_40_inplace", np.int64, 4, 64, 4, 64, 4, 64, scalar=17, valid_col=40, inplace=True
        ),
        TremsParams(
            "TREMSTest.case_int64_1x2048_2045_inplace",
            np.int64,
            1,
            2048,
            1,
            2048,
            1,
            2048,
            scalar=17,
            valid_col=2045,
            inplace=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_positive_divisor_4x64",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=3,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_negative_divisor_4x64",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=-3,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_minus_one_divisor_4x64",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=-1,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_min_divisor_4x64",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=-(2**63),
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_large_divisor_4x64",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=-(2**32 + 1),
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_int64_zero_divisor_4x64", np.int64, 4, 64, 4, 64, 4, 64, scalar=0, use_edge_cases=True
        ),
        TremsParams(
            "TREMSTest.case_int64_floor_negative_divisor_4x64_40_inplace",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=-3,
            valid_col=40,
            inplace=True,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_uint64_one_divisor_4x64", np.uint64, 4, 64, 4, 64, 4, 64, scalar=1, use_edge_cases=True
        ),
        TremsParams(
            "TREMSTest.case_uint64_high_bit_divisor_4x64",
            np.uint64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=2**63,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_uint64_max_divisor_4x64",
            np.uint64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=2**64 - 1,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_uint64_large_divisor_4x64",
            np.uint64,
            4,
            64,
            4,
            64,
            4,
            64,
            scalar=2**32 + 1,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_uint64_zero_divisor_4x64", np.uint64, 4, 64, 4, 64, 4, 64, scalar=0, use_edge_cases=True
        ),
        TremsParams(
            "TREMSTest.case_int64_zero_divisor_4x64_40_inplace",
            np.int64,
            4,
            64,
            4,
            64,
            4,
            40,
            scalar=0,
            inplace=True,
            use_edge_cases=True,
        ),
        TremsParams(
            "TREMSTest.case_uint64_zero_divisor_4x64_40_inplace",
            np.uint64,
            4,
            64,
            4,
            64,
            4,
            40,
            scalar=0,
            inplace=True,
            use_edge_cases=True,
        ),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
