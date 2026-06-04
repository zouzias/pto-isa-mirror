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

NUM_TOKENS = 16
TOPK = 4
NUM_EXPERTS = 4
ACTUAL_CORES = 16
CHUNK_SIZE = 4
HIDDEN_SIZE = 8
OUT_LEN = 64
E = NUM_TOKENS * TOPK
PADDED_E = ACTUAL_CORES * CHUNK_SIZE
CASE_NAME = "MoeTokenPermuteRegisterTest.case_fp16_standard"


def compute_golden(tokens, indices):
    workspace = np.zeros(PADDED_E, dtype=np.int32)
    golden_sio = np.zeros(PADDED_E, dtype=np.int32)
    golden_perm = np.zeros((OUT_LEN, HIDDEN_SIZE), dtype=np.float16)

    for cid in range(ACTUAL_CORES):
        my_start = cid * CHUNK_SIZE
        idx_chunk = indices[my_start:my_start + CHUNK_SIZE]
        hist = np.zeros(NUM_EXPERTS, dtype=np.int32)
        for i in range(CHUNK_SIZE):
            if my_start + i < E:
                hist[idx_chunk[i]] += 1
        workspace[cid * NUM_EXPERTS:(cid + 1) * NUM_EXPERTS] = hist

    for cid in range(ACTUAL_CORES):
        my_start = cid * CHUNK_SIZE
        idx_chunk = indices[my_start:my_start + CHUNK_SIZE]
        running = 0
        offsets = np.zeros(NUM_EXPERTS, dtype=np.int32)
        for e in range(NUM_EXPERTS):
            acc = 0
            cpre = 0
            for c in range(ACTUAL_CORES):
                acc += workspace[c * NUM_EXPERTS + e]
                if c < cid:
                    cpre += workspace[c * NUM_EXPERTS + e]
            offsets[e] = running + cpre
            running += acc

        counters = np.zeros(NUM_EXPERTS, dtype=np.int32)
        sio_chunk = np.zeros(CHUNK_SIZE, dtype=np.int32)
        for i in range(CHUNK_SIZE):
            if my_start + i < E:
                expert = idx_chunk[i]
                wp = offsets[expert] + counters[expert]
                counters[expert] += 1
                sio_chunk[i] = wp
        golden_sio[my_start:my_start + CHUNK_SIZE] = sio_chunk

        if cid < NUM_TOKENS:
            token = tokens[cid]
            for k in range(TOPK):
                wp = sio_chunk[k]
                if wp < OUT_LEN:
                    golden_perm[wp] = token

    return workspace, golden_perm, golden_sio


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    tokens = rng.standard_normal((NUM_TOKENS, HIDDEN_SIZE)).astype(np.float16)
    indices = rng.integers(0, NUM_EXPERTS, size=PADDED_E, dtype=np.int32)
    _, golden_perm, golden_sio = compute_golden(tokens, indices)

    if not os.path.exists(CASE_NAME):
        os.makedirs(CASE_NAME)
    tokens.tofile(f"{CASE_NAME}/tokens.bin")
    indices.tofile(f"{CASE_NAME}/indices.bin")
    golden_perm.tofile(f"{CASE_NAME}/golden_perm.bin")
    golden_sio.tofile(f"{CASE_NAME}/golden_sio.bin")
