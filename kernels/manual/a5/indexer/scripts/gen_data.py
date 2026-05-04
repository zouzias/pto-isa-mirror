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
"""Data + golden for indexer pipeline.

Inputs:
- q: [2, 64, 128] (stored as merged A: [128, 128] for MX matmul path)
- k: [131072, 128] (stored as B in DN-compatible layout)
- post_scale: [2, 64]

Golden:
- matmul C: [128, 131072] float32
- score: [2, 131072] = reduce_sum_64(relu(C_reshaped) * post_scale[..., None])
- score ordered-key: [2, 131072] uint32
- topk multiset: [2, 2048] float32 values (order-free reference)
- topk idx (deterministic): [2, 2048] uint32 by ordered-key desc, idx asc
"""

import os

import numpy as np
import ml_dtypes

fp8_e5m2 = ml_dtypes.float8_e5m2
np.random.seed(29)

BS = 2
M = 128
H = M // BS
K = 1024
N = 131072
TOPK = 2048
SCALE_GROUP = 32


def main():
    global N
    N = int(os.getenv("INDEXER_TEST_N", str(N)))
    topk_env = int(os.getenv("INDEXER_TOPK", str(TOPK)))
    topk = min(topk_env, N)
    q = np.random.uniform(-5, 5, [BS, H, K]).astype(fp8_e5m2)
    k = np.random.uniform(-5, 5, [N, K]).astype(fp8_e5m2)

    x1_gm = q.reshape(M, K).copy()
    x2_math = k.transpose().copy()  # [K, N]

    k_mx = K // SCALE_GROUP
    x1_scale_gm = np.random.randint(127, 130, [M, k_mx]).astype(np.uint8)
    x2_scale_gm = np.random.randint(127, 130, [k_mx, N]).astype(np.uint8)
    post_scale = np.random.uniform(-2.0, 2.0, [BS, H]).astype(np.float32)

    x1_scale = 2 ** (x1_scale_gm.astype(np.float32) - 127)
    x2_scale = 2 ** (x2_scale_gm.astype(np.float32) - 127)

    x1 = np.zeros([M, K], dtype=np.float32)
    x2 = np.zeros([K, N], dtype=np.float32)
    for i in range(K):
        x1[:, i] = x1_gm[:, i] * x1_scale[:, i // SCALE_GROUP]
        x2[i, :] = x2_math[i, :] * x2_scale[i // SCALE_GROUP, :]

    # Match matmul_mxfp8_performance: transpose for GM layout (is_btrans True in original gen).
    x2_gm_store = x2_math.transpose().copy()
    x2_scale_store = x2_scale_gm.transpose().copy()

    c_deq = x1.astype(np.float32) @ x2.astype(np.float32)
    c_3d = c_deq.reshape(BS, H, N)
    golden_score = (np.maximum(c_3d, 0.0) * post_scale[:, :, None]).sum(axis=1).astype(np.float32)
    golden_score_key = np.empty([BS, N], dtype=np.uint32)
    golden_topk_multiset = np.empty([BS, topk], dtype=np.float32)
    golden_topk_idx = np.empty([BS, topk], dtype=np.uint32)

    def ordered_key(x: np.ndarray) -> np.ndarray:
        bits = x.view(np.uint32)
        sign = bits >> 31
        # Float order preserving mapping to unsigned integer domain.
        # negative: bitwise not, non-negative: flip sign bit.
        return np.where(sign == 1, ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)

    for b in range(BS):
        golden_score_key[b] = ordered_key(golden_score[b])
        top_vals = np.partition(golden_score[b], -topk)[-topk:]
        golden_topk_multiset[b] = np.sort(top_vals)
        keys = golden_score_key[b]
        order = np.lexsort((np.arange(N, dtype=np.uint32), -keys.astype(np.int64)))
        golden_topk_idx[b] = order[:topk]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm_store.tofile("./input/x2_gm.bin")
    x1_scale_gm.tofile("./input/x1_scale_gm.bin")
    x2_scale_store.tofile("./input/x2_scale_gm.bin")
    post_scale.tofile("./input/post_scale.bin")
    c_deq.astype(np.float32).tofile("./output/golden_matmul.bin")
    golden_score.tofile("./output/golden_score.bin")
    golden_score_key.tofile("./output/golden_score_key.bin")
    golden_topk_multiset.tofile("./output/golden_topk_multiset.bin")
    golden_topk_idx.tofile("./output/golden_topk_idx.bin")


if __name__ == "__main__":
    main()
