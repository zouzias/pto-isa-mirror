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


SUITE_NAME = "TTRANSConvValidShapeTest"


def golden_NCHW2NC1HWC0(n, c, h, w, dtype):
    c0 = 32 // np.dtype(dtype).itemsize
    c1 = (c + c0 - 1) // c0

    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(dtype)
    input_arr.tofile("./input.bin")

    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')

    output_arr = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)
    output_arr.tofile("./golden.bin")
    print(f"  -> Input shape: ({n},{c},{h},{w}) | Output shape: {output_arr.shape}")


test_cases = [
    ("case_default_half", np.float16, 4, 4, 4, 4),
    ("case_valid_half", np.float16, 4, 4, 4, 4),
    ("case_default_float", np.float32, 4, 4, 4, 4),
    ("case_valid_float", np.float32, 4, 4, 4, 4),
]

if __name__ == "__main__":
    print(f"Generating test data for {len(test_cases)} cases...")

    for name, dtype, n, c, h, w in test_cases:
        dirname = f"{SUITE_NAME}.{name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        original_dir = os.getcwd()
        os.chdir(dirname)
        print(f"Case: {name} | dtype: {dtype.__name__}")
        golden_NCHW2NC1HWC0(n, c, h, w, dtype)
        os.chdir(original_dir)

    print("\nDone.")
