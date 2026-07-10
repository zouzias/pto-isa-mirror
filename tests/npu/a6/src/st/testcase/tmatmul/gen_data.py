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


np.random.seed(19)


class TMatmulParams:
    def __init__(self, a_type, b_type, out_type, m, k, n):
        self.a_type = a_type
        self.b_type = b_type
        self.out_type = out_type
        self.m = m
        self.k = k
        self.n = n


def gen_golden_data(param):
    x1_gm = np.random.uniform(-5, 5, [param.m, param.k]).astype(param.a_type)
    x2_gm = np.random.uniform(-5, 5, [param.k, param.n]).astype(param.b_type)

    # Use integer-friendly generation for int8 cases to keep deterministic results stable.
    if param.a_type == np.int8 and param.b_type == np.int8:
        x1_gm = np.random.randint(-8, 8, [param.m, param.k], dtype=np.int8)
        x2_gm = np.random.randint(-8, 8, [param.k, param.n], dtype=np.int8)

    golden = np.matmul(x1_gm.astype(param.out_type), x2_gm.astype(param.out_type)).astype(param.out_type)

    x1_gm.tofile("x1_gm.bin")
    x2_gm.tofile("x2_gm.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    case_name_list = [
        "TMATMULTest.case_fp16_fp16_to_fp32_31x96x47",
        "TMATMULTest.case_int8_int8_to_int32_65x90x89",
        "TMATMULTest.case_fp32_fp32_to_fp32_16x32x64",
        "TMATMULTest.case_fp16_fp16_to_fp32_1x256x64",
    ]

    case_params_list = [
        TMatmulParams(np.float16, np.float16, np.float32, 31, 96, 47),
        TMatmulParams(np.int8, np.int8, np.int32, 65, 90, 89),
        TMatmulParams(np.float32, np.float32, np.float32, 16, 32, 64),
        TMatmulParams(np.float16, np.float16, np.float32, 1, 256, 64),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)
