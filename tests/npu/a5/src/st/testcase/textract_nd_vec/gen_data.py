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

if __name__ == "__main__":
    nd_vec_case_names = [
        "TExtractNDVecTest.case_nd_vec_1",
        "TExtractNDVecTest.case_nd_vec_2",
        "TExtractNDVecTest.case_nd_vec_3",
    ]

    nd_vec_params = [
        (np.float32, 32, 512, 32, 512, 0, 0),
        (np.float32, 512, 32, 512, 32, 0, 0),
        (np.float32, 128, 128, 128, 128, 0, 0),
    ]

    for i, case_name in enumerate(nd_vec_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row, idx_col = nd_vec_params[i]

        if dtype == np.int8:
            src_data = np.random.randint(-128, 127, size=(src_rows, src_cols)).astype(dtype)
            dst_init = np.random.randint(-128, 127, size=(dst_rows, dst_cols)).astype(dtype)
        else:
            src_data = np.random.uniform(-10, 10, size=(src_rows, src_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)

        src_data.tofile("src_input.bin")
        dst_init.tofile("dst_init.bin")

        golden = dst_init.copy()
        golden[0:dst_rows, 0:dst_cols] = src_data[idx_row : idx_row + dst_rows, idx_col : idx_col + dst_cols]
        golden.tofile("golden_output.bin")

        os.chdir(original_dir)

