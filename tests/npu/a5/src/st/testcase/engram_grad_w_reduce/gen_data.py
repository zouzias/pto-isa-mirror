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

CASENAME = "EngramGradWReduce"

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    print("Warning: PyTorch not available, using NumPy for golden data generation")


def float32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    arr_f32 = np.asarray(arr, dtype=np.float32)
    u32 = arr_f32.view(np.uint32)
    round_bias = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
    bf16 = ((u32 + round_bias) >> 16).astype(np.uint16)
    return bf16


def bf16_bits_to_float32(arr_bits: np.ndarray) -> np.ndarray:
    u32 = np.asarray(arr_bits, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


class EngramGradWReduceParams:
    def __init__(self, num_blocks, hidden_size):
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size


def gen_golden_data(case_name, param):
    num_blocks = param.num_blocks
    hidden_size = param.hidden_size

    grad_w_partial = np.random.randn(num_blocks, hidden_size).astype(np.float32)
    weight_hidden = np.random.randn(hidden_size).astype(np.float32)
    weight_embed = np.random.randn(hidden_size).astype(np.float32)
    grad_weight_hidden_init = np.random.randn(hidden_size).astype(np.float32)
    grad_weight_embed_init = np.random.randn(hidden_size).astype(np.float32)

    weight_hidden_bf16_bits = float32_to_bf16_bits(weight_hidden)
    weight_embed_bf16_bits = float32_to_bf16_bits(weight_embed)
    weight_hidden_bf16_vals = bf16_bits_to_float32(weight_hidden_bf16_bits)
    weight_embed_bf16_vals = bf16_bits_to_float32(weight_embed_bf16_bits)

    grad_w_sum = grad_w_partial.sum(axis=0)
    golden_grad_wh = grad_weight_hidden_init + grad_w_sum * weight_embed_bf16_vals
    golden_grad_we = grad_weight_embed_init + grad_w_sum * weight_hidden_bf16_vals

    grad_w_partial.tofile("grad_w_partial.bin")
    weight_hidden_bf16_bits.tofile("weight_hidden.bin")
    weight_embed_bf16_bits.tofile("weight_embed.bin")
    grad_weight_hidden_init.tofile("grad_wh_init.bin")
    grad_weight_embed_init.tofile("grad_we_init.bin")
    golden_grad_wh.tofile("golden_grad_wh.bin")
    golden_grad_we.tofile("golden_grad_we.bin")


def generate_case_name(param):
    return f"{CASENAME}Test.case_{param.num_blocks}blocks_{param.hidden_size}cols"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        EngramGradWReduceParams(num_blocks=4, hidden_size=128),
        EngramGradWReduceParams(num_blocks=8, hidden_size=256),
        EngramGradWReduceParams(num_blocks=16, hidden_size=512),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)