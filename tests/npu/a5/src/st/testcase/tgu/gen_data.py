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


def generate_case_name(name):
    return f"TGUTest.{name}"


def write_bin(path, array):
    with open(path, "wb") as f:
        f.write(array.tobytes())


def gen_golden_data(root_dir, rows, cols, last_tile=False):
    prev = np.random.uniform(low=0.1, high=1.0, size=(rows, cols)).astype(np.float32)
    est = np.random.uniform(low=-0.5, high=0.5, size=(rows, cols)).astype(np.float32)
    # exp_max = np.random.uniform(low=0.1, high=2.0, size=(rows, 1)).astype(np.float32)
    exp_max = np.ones((rows, 1)).astype(np.float32)
    pv_pend = np.random.uniform(low=-0.2, high=0.2, size=(rows, cols)).astype(np.float32)

    write_bin(os.path.join(root_dir, "input0.bin"), prev)
    write_bin(os.path.join(root_dir, "input1.bin"), est)
    write_bin(os.path.join(root_dir, "input2.bin"), exp_max)
    write_bin(os.path.join(root_dir, "input3.bin"), pv_pend)

    if last_tile:
        global_sum = np.random.uniform(low=1.0, high=3.0, size=(rows, 1)).astype(np.float32)
        write_bin(os.path.join(root_dir, "global_sum.bin"), global_sum)
        golden = (est + exp_max * prev) / global_sum
    else:
        golden = est + exp_max * prev

    write_bin(os.path.join(root_dir, "golden.bin"), golden)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cases = [
        ("case1_32_128", False),
        ("case2_32_128_last_tile", True),
    ]

    for case_name, last_tile in cases:
        case_dir = os.path.join(script_dir, generate_case_name(case_name))
        os.makedirs(case_dir, exist_ok=True)
        gen_golden_data(case_dir, 32, 128, last_tile)
