#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import NumExt


class TPartMulParams:
    def __init__(self, dtype, src0_rows, src0_cols, src1_rows, src1_cols, out_rows, out_cols):
        self.dtype = dtype
        self.src0_rows = src0_rows
        self.src0_cols = src0_cols
        self.src1_rows = src1_rows
        self.src1_cols = src1_cols
        self.out_rows = out_rows
        self.out_cols = out_cols


def random_array(rng, rows, cols, dtype):
    if NumExt.is_bf16(dtype):
        return rng.uniform(-2.0, 2.0, size=(rows, cols)).astype(np.float32)
    if NumExt.is_unsigned_integer(dtype):
        return rng.integers(0, 256, size=(rows, cols), dtype=dtype)
    if NumExt.is_signed_integer(dtype):
        return rng.integers(-16, 16, size=(rows, cols), dtype=dtype)
    return rng.uniform(-2.0, 2.0, size=(rows, cols)).astype(dtype)


def generate_case(param):
    rng = np.random.default_rng(19)
    src0 = random_array(rng, param.src0_rows, param.src0_cols, param.dtype)
    src1 = random_array(rng, param.src1_rows, param.src1_cols, param.dtype)
    golden = NumExt.zeros((param.out_rows, param.out_cols), param.dtype)

    if param.src0_rows > 0 and param.src0_cols > 0:
        golden[: param.src0_rows, : param.src0_cols] = src0
    if param.src1_rows > 0 and param.src1_cols > 0:
        golden[: param.src1_rows, : param.src1_cols] = src1

    overlap_rows = min(param.src0_rows, param.src1_rows)
    overlap_cols = min(param.src0_cols, param.src1_cols)
    if overlap_rows > 0 and overlap_cols > 0:
        golden[:overlap_rows, :overlap_cols] = np.multiply(
            src0[:overlap_rows, :overlap_cols],
            src1[:overlap_rows, :overlap_cols],
            dtype=golden.dtype,
        )

    NumExt.write_array("input1.bin", src0, param.dtype)
    NumExt.write_array("input2.bin", src1, param.dtype)
    NumExt.write_array("golden.bin", golden, param.dtype)


def generate_case_name(param):
    dtype_name = NumExt.get_short_type_name(param.dtype)
    return (
        f"TPARTMUL_Test.case_{dtype_name}_"
        f"{param.out_rows}x{param.out_cols}_"
        f"{param.src0_rows}x{param.src0_cols}_"
        f"{param.src1_rows}x{param.src1_cols}"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
    os.makedirs(testcases_dir, exist_ok=True)

    shape_params = [
        (64, 64, 64, 64, 64, 64),
        (64, 64, 32, 32, 64, 64),
        (32, 32, 64, 64, 64, 64),
        (64, 64, 32, 64, 64, 64),
        (64, 64, 64, 32, 64, 64),
    ]
    case_params = []
    for dtype in [np.float32, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
        for shape in shape_params[:2]:
            case_params.append(TPartMulParams(dtype, *shape))

    case_params.extend(
        [
            TPartMulParams(np.float32, *shape) for shape in shape_params[2:]
        ]
    )
    case_params.append(TPartMulParams(np.float16, 16, 256, 16, 256, 16, 256))
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params.append(TPartMulParams(NumExt.bf16, 16, 256, 16, 256, 16, 256))

    for param in case_params:
        case_name = generate_case_name(param)
        case_dir = os.path.join(testcases_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        current_dir = os.getcwd()
        os.chdir(case_dir)
        generate_case(param)
        os.chdir(current_dir)
