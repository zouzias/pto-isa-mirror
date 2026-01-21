
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
    test_type = param.test_type
    rows = param.rows
    cols = param.cols

    input_arr = np.random.uniform(low=-10, high=10, size=(rows, cols)).astype(test_type)
    input_arr.tofile("input_arr.bin")

    nz_block_row = 16
    if test_type == np.int8:
        c0_size = 32
    elif test_type == np.float32 or test_type == np.int32:
        c0_size = 8
    else:
        c0_size = 16

    output_arr = input_arr.reshape(
        int(rows / nz_block_row), nz_block_row,
        int(cols / c0_size), c0_size
    ).transpose(2, 0, 1, 3).astype(test_type)
    output_arr.tofile("golden_output.bin")

    print(f"Generated: input{input_arr.shape} -> output{output_arr.shape} ({test_type})")


class TInsertCustomParams:
    def __init__(self, test_type, rows, cols, mode="NZ_PLUS_1"):
        self.test_type = test_type
        self.rows = rows
        self.cols = cols
        self.mode = mode

if __name__ == "__main__":
    np.random.seed(42)
    case_name_list = [
        "TInsertCustomTest.case1",
        "TInsertCustomTest.case2",
        "TInsertCustomTest.case3",
        "TInsertCustomTest.case4",
        "TInsertCustomTest.case5",
        "TInsertCustomTest.case6",
        "TInsertCustomTest.case7",
    ]

    case_params_list = [
        TInsertCustomParams(np.float32, 16, 32, "NZ"),
        TInsertCustomParams(np.float32, 16, 32, "NZ_PLUS_1"),
        TInsertCustomParams(np.float32, 32, 64, "NZ_PLUS_1"),
        TInsertCustomParams(np.int32, 32, 32, "NZ_PLUS_1"),
        TInsertCustomParams(np.float32, 32, 32, "SPLIT2_NZ_PLUS_1"),
        TInsertCustomParams(np.float32, 32, 32, "SPLIT4_NZ_PLUS_1"),
        TInsertCustomParams(np.float32, 64, 64, "SPLIT4_NZ_PLUS_1"),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)

    print("All test data generated successfully!")