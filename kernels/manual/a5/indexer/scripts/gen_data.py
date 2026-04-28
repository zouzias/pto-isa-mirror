#!/usr/bin/env python3
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
"""Golden for indexer MXFP8 kernel: A [m,k], B [k,n] dequant matmul, output 0.5 * result (float32)."""

import os

import numpy as np
import ml_dtypes

fp8_e5m2 = ml_dtypes.float8_e5m2
np.random.seed(29)

M = 512
K = 1024
N = 512
SCALE_GROUP = 32


def main():
    x1_gm = np.random.uniform(-5, 5, [M, K]).astype(fp8_e5m2)
    x2_gm = np.random.uniform(-5, 5, [K, N]).astype(fp8_e5m2)

    k_mx = K // SCALE_GROUP
    x1_scale_gm = np.random.randint(127, 130, [M, k_mx]).astype(np.uint8)
    x2_scale_gm = np.random.randint(127, 130, [k_mx, N]).astype(np.uint8)

    x1_scale = 2 ** (x1_scale_gm.astype(np.float32) - 127)
    x2_scale = 2 ** (x2_scale_gm.astype(np.float32) - 127)

    x1 = np.zeros([M, K], dtype=np.float32)
    x2 = np.zeros([K, N], dtype=np.float32)
    for i in range(K):
        x1[:, i] = x1_gm[:, i] * x1_scale[:, i // SCALE_GROUP]
        x2[i, :] = x2_gm[i, :] * x2_scale[i // SCALE_GROUP, :]

    # Match matmul_mxfp8_performance: transpose for GM layout (is_btrans True in original gen).
    x2_gm_store = x2_gm.transpose().copy()
    x2_scale_store = x2_scale_gm.transpose().copy()

    c_deq = x1.astype(np.float32) @ x2.astype(np.float32)
    golden_f32 = (0.5 * c_deq).astype(np.float32)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm_store.tofile("./input/x2_gm.bin")
    x1_scale_gm.tofile("./input/x1_scale_gm.bin")
    x2_scale_store.tofile("./input/x2_scale_gm.bin")
    golden_f32.tofile("./output/golden.bin")


if __name__ == "__main__":
    main()
