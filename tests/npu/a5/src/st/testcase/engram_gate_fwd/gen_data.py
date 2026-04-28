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
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16


def gen_golden_data_engram_gate_fwd(param):
    dtype = param.dtype
    dtype_float = param.dtype_float
    hidden_size = param.hidden_size
    tile_row, tile_col = [param.tile_row, param.tile_col]
    valid_row, valid_col = [param.valid_row, param.valid_col]
    eps = param.eps
    clamp_value = param.clamp_value

    scalar = 1.0 / np.sqrt(hidden_size)

    x = np.random.randn(valid_row, valid_col).astype(dtype)
    k = np.random.randn(valid_row, valid_col).astype(dtype)
    v = np.random.randn(valid_row, valid_col).astype(dtype)
    weight_fused = np.random.randn(valid_row, valid_col).astype(dtype_float)

    x_float = x.astype(np.float32)
    k_float = k.astype(np.float32)
    v_float = v.astype(np.float32)

    x2 = x_float * x_float
    k2 = k_float * k_float

    sum_x2 = np.sum(x2, axis=-1, keepdims=True)
    sum_k2 = np.sum(k2, axis=-1, keepdims=True)

    rstd_x = np.sqrt(1.0 / (sum_x2 / hidden_size + eps))
    rstd_k = np.sqrt(1.0 / (sum_k2 / hidden_size + eps))

    dot = np.sum(x_float * weight_fused * k_float, axis=-1, keepdims=True)

    dot_scaled = dot * rstd_x * rstd_k * scalar

    abs_dot_scaled = np.abs(dot_scaled)
    abs_dot_scaled = np.maximum(abs_dot_scaled, clamp_value)
    signed_sqrt = np.sqrt(abs_dot_scaled) * np.sign(dot_scaled)

    gate_score = 1.0 / (1.0 + np.exp(-signed_sqrt))

    output = x_float + gate_score * v_float
    output = output.astype(dtype)

    x.tofile("input_x.bin")
    k.tofile("input_k.bin")
    v.tofile("input_v.bin")
    weight_fused.tofile("input_weight.bin")
    output.tofile("golden.bin")


class EngramGateFwdParams:
    def __init__(self, name, dtype, dtype_float, hidden_size, tile_row, tile_col, valid_row, valid_col, eps=1e-20,
                 clamp_value=1e-6):
        self.name = name
        self.dtype = dtype
        self.dtype_float = dtype_float
        self.hidden_size = hidden_size
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.eps = eps
        self.clamp_value = clamp_value


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        EngramGateFwdParams("EngramGateFwdTest.case_bf16_64x64_64x64", np.float32, np.float32, 64, 64, 64, 64, 64),
        EngramGateFwdParams("EngramGateFwdTest.case_bf16_4096_64x64_64x64", np.float32, np.float32, 4096, 64, 64, 64, 64),
    ]

    for param in case_params_list:
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data_engram_gate_fwd(param)
        os.chdir(original_dir)