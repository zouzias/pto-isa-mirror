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

np.random.seed(20251220)

SEQ_LEN = 64
HEAD_DIM = 32
SCALE = 1.0 / np.sqrt(HEAD_DIM)


def flash_attention_ref(q, k, v, scale):
    scores = q @ k.T * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    out = probs @ v
    return out


def gen_case_half():
    q = (np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02).astype(np.float16)
    k = (np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02).astype(np.float16)
    v = (np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02).astype(np.float16)

    out = flash_attention_ref(q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), SCALE)
    golden = out.astype(np.float16)

    q.tofile("q.bin")
    k.T.copy().tofile("kt.bin")
    v.tofile("v.bin")
    golden.tofile("golden.bin")


def gen_case_s8():
    q_f = np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02
    k_f = np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02
    v = (np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02).astype(np.float16)

    q_s8 = np.clip(q_f * 127.0, -128, 127).astype(np.int8)
    k_s8 = np.clip(k_f * 127.0, -128, 127).astype(np.int8)

    out = flash_attention_ref(q_s8.astype(np.float32), k_s8.astype(np.float32), v.astype(np.float32), SCALE)
    golden = out.astype(np.float16)

    q_s8.tofile("q.bin")
    k_s8.T.copy().tofile("kt.bin")
    v.tofile("v.bin")
    golden.tofile("golden.bin")


def gen_case_s16():
    q_f = np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02
    k_f = np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02
    v = (np.random.randn(SEQ_LEN, HEAD_DIM) * 0.02).astype(np.float16)

    q_s16 = np.clip(q_f * 32767.0, -32768, 32767).astype(np.int16)
    k_s16 = np.clip(k_f * 32767.0, -32768, 32767).astype(np.int16)

    out = flash_attention_ref(q_s16.astype(np.float32), k_s16.astype(np.float32), v.astype(np.float32), SCALE)
    golden = out.astype(np.float16)

    q_s16.tofile("q.bin")
    k_s16.T.copy().tofile("kt.bin")
    v.tofile("v.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    cases = [
        ("TFLASHATTNTest.case_half", gen_case_half),
        ("TFLASHATTNTest.case_s8", gen_case_s8),
        ("TFLASHATTNTest.case_s16", gen_case_s16),
    ]
    for name, gen_fn in cases:
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        gen_fn()
        os.chdir("..")
