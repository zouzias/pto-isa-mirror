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

"""
Generate deterministic golden data for the tpushpop_both_dir testcase.

This script mirrors the formulas used in `main.cpp`:
  - attn_out[row, col]      = row * 10 + col
  - hidden_states[row, col] = (row % 4) * 3 - (col % 5), stored as bf16 bits
  - wo is a bf16-packed [128, 64] identity-like matrix
  - golden[row, col]        = bf16_round(attn_out[row, col]) + bf16_to_fp32(hidden_states[row, col])

Files written under `TPushPopBothDirTest.case1/`:
  - attn_out.bin       : float32, shape [16, 128]
  - hidden_states.bin  : uint16 bf16 payload, shape [16, 64]
  - resid_out_init.bin : float32, shape [16, 64], filled with -777
  - wo.bin             : uint16 bf16 payload, shape [128, 64]
  - golden.bin         : float32, shape [16, 64]
"""

import os

import numpy as np


class CaseParams:
    def __init__(self, name, rows, k_cols, out_cols, init_value):
        self.name = name
        self.rows = rows
        self.k_cols = k_cols
        self.out_cols = out_cols
        self.init_value = init_value


def float_to_bf16_bits(values):
    """Round float32 values to bf16 using the same rule as main.cpp."""
    values_f32 = np.asarray(values, dtype=np.float32)
    bits = values_f32.view(np.uint32).copy()
    lsb = (bits >> 16) & np.uint32(1)
    bits += np.uint32(0x7FFF) + lsb
    return (bits >> 16).astype(np.uint16)


def bf16_bits_to_float(bits):
    bf16 = np.asarray(bits, dtype=np.uint16)
    return (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def gen_case_data(params):
    rows = params.rows
    k_cols = params.k_cols
    out_cols = params.out_cols

    row_ids = np.arange(rows, dtype=np.int32)[:, np.newaxis]
    k_col_ids = np.arange(k_cols, dtype=np.int32)[np.newaxis, :]
    out_col_ids = np.arange(out_cols, dtype=np.int32)[np.newaxis, :]

    attn_out = (row_ids * 10 + k_col_ids).astype(np.float32)
    attn_out_bf16_rounded = bf16_bits_to_float(float_to_bf16_bits(attn_out))

    hidden_fp32 = ((row_ids % 4) * 3 - (out_col_ids % 5)).astype(np.float32)
    hidden_states = float_to_bf16_bits(hidden_fp32)
    hidden_fp32_from_bf16 = bf16_bits_to_float(hidden_states)

    resid_out_init = np.full((rows, out_cols), params.init_value, dtype=np.float32)

    wo = np.zeros((k_cols, out_cols), dtype=np.float32)
    diag_len = min(k_cols, out_cols)
    wo[np.arange(diag_len), np.arange(diag_len)] = 1.0
    wo_bf16 = float_to_bf16_bits(wo)

    golden = (attn_out_bf16_rounded[:, :out_cols] + hidden_fp32_from_bf16).astype(np.float32)

    attn_out.tofile("attn_out.bin")
    hidden_states.tofile("hidden_states.bin")
    resid_out_init.tofile("resid_out_init.bin")
    wo_bf16.tofile("wo.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    case_params_list = [
        CaseParams("TPushPopBothDirTest.case1", rows=16, k_cols=128, out_cols=64, init_value=-777.0),
    ]

    for params in case_params_list:
        if not os.path.exists(params.name):
            os.makedirs(params.name)
        original_dir = os.getcwd()
        os.chdir(params.name)
        gen_case_data(params)
        os.chdir(original_dir)
        print(f"Generated: {params.name}")
