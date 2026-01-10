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

ROWS = 64
COLS = 64
ELEMS = ROWS * COLS


def _mkdir(case_dir: str):
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)


def gen_case_int32_vec_ops(case_dir: str):
    a = np.random.randint(-1000, 1000, size=(ROWS, COLS), dtype=np.int32)
    # Use strictly positive small values to avoid div-by-zero and keep shifts safe.
    b = np.random.randint(1, 32, size=(ROWS, COLS), dtype=np.int32)
    sh = (b & 31).astype(np.int32)

    rem_scalar = np.int32(7)
    logic_scalar = np.int32(0x0F0F0F0F)

    def cxx_rem(x, y):
        xf = x.astype(np.float64)
        yf = y.astype(np.float64)
        q = np.trunc(xf / yf).astype(np.int64)
        return (x.astype(np.int64) - q * y.astype(np.int64)).astype(np.int32)

    outs = [
        np.bitwise_and(a, b),
        np.bitwise_or(a, b),
        np.bitwise_xor(a, b),
        np.bitwise_not(a),
        np.left_shift(a, sh),
        np.right_shift(a, sh),
        cxx_rem(a, b),
        cxx_rem(a, rem_scalar),
        np.bitwise_and(a, logic_scalar),
        np.bitwise_or(a, logic_scalar),
        np.bitwise_xor(a, logic_scalar),
    ]
    golden = np.stack(outs, axis=0).astype(np.int32).reshape(-1)

    a.tofile(os.path.join(case_dir, "input1.bin"))
    b.tofile(os.path.join(case_dir, "input2.bin"))
    golden.tofile(os.path.join(case_dir, "golden.bin"))


def gen_case_float32_vec_ops(case_dir: str):
    x = (np.random.rand(ROWS, COLS).astype(np.float32) * 10.0) - 5.0
    y = (np.random.rand(ROWS, COLS).astype(np.float32) * 4.9) + 0.1  # avoid div-by-zero
    z = (np.random.rand(ROWS, COLS).astype(np.float32) * 4.0) - 2.0

    lrelu = np.float32(0.1)
    bias = np.float32(1.25)
    rem_scalar = np.float32(1.3)

    outs = [
        -x,
        np.maximum(x, np.float32(0.0)),
        np.where(x > 0, x, x * lrelu),
        np.where(x > 0, x, x * y),
        x + y + z,
        x - y + z,
        x + bias + y,
        x - bias + y,
        x - bias,
        np.maximum(x, bias),
        np.fmod(x, y),
        np.fmod(x, rem_scalar),
    ]
    golden = np.stack(outs, axis=0).astype(np.float32).reshape(-1)

    x.tofile(os.path.join(case_dir, "input1.bin"))
    y.tofile(os.path.join(case_dir, "input2.bin"))
    z.tofile(os.path.join(case_dir, "input3.bin"))
    golden.tofile(os.path.join(case_dir, "golden.bin"))


def gen_case_mgather_mscatter_int32(case_dir: str):
    mem_src = np.random.randint(-1000, 1000, size=(ELEMS,), dtype=np.int32)
    mem_dst_init = np.random.randint(-1000, 1000, size=(ELEMS,), dtype=np.int32)

    # Use a permutation to avoid duplicate indices (MSCATTER is implementation-defined on duplicates).
    idx = np.arange(ELEMS, dtype=np.int32)
    np.random.shuffle(idx)
    idx = idx.reshape(ROWS, COLS)
    scatter_src = np.random.randint(-1000, 1000, size=(ROWS, COLS), dtype=np.int32)

    mgather = mem_src[idx.reshape(-1)].reshape(ROWS, COLS).astype(np.int32)

    mem_dst = mem_dst_init.copy()
    mem_dst[idx.reshape(-1)] = scatter_src.reshape(-1)

    golden = np.concatenate([mgather.reshape(-1), mem_dst.reshape(-1)], axis=0).astype(np.int32)

    mem_src.tofile(os.path.join(case_dir, "input1.bin"))
    idx.tofile(os.path.join(case_dir, "input2.bin"))
    scatter_src.tofile(os.path.join(case_dir, "input3.bin"))
    mem_dst_init.tofile(os.path.join(case_dir, "input4.bin"))
    golden.tofile(os.path.join(case_dir, "golden.bin"))


if __name__ == "__main__":
    base = os.getcwd()

    cases = [
        ("TISAMISSINGTest.case_int32_vec_ops", gen_case_int32_vec_ops),
        ("TISAMISSINGTest.case_float32_vec_ops", gen_case_float32_vec_ops),
        ("TISAMISSINGTest.case_mgather_mscatter_int32", gen_case_mgather_mscatter_int32),
    ]

    for name, fn in cases:
        _mkdir(name)
        fn(name)
