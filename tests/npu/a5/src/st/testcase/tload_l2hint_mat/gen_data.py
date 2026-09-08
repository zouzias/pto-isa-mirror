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


def gen_golden_data(M, K):
    data_type = np.float32
    input_arr = np.random.randint(-5, 5, size=(M, K)).astype(data_type)
    output_arr = input_arr.copy()
    input_arr.tofile("./x1_gm.bin")
    # unused second src for shape2d-compatible host path
    np.zeros((M, K), dtype=data_type).tofile("./x2_gm.bin")
    output_arr.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = [
        "TLOADL2HintMatTest.case_mat_float_ND_1_1_1_128_256_alloc_x5",
        "TLOADL2HintMatTest.case_mat_float_ND_1_1_1_128_256_notalloc_x5",
    ]
    for case_name in case_name_list:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(128, 256)
        os.chdir(original_dir)
