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
import ml_dtypes
np.random.seed(19)

MAX_NGRAM_SIZE = 3
NUM_NGRAM_LAYERS = 2
NUM_EMBED_TABLE_PER_NGRAM = 8
NUM_OUT_COLS = (MAX_NGRAM_SIZE - 1) * NUM_EMBED_TABLE_PER_NGRAM


def make_offsets(vocab_sizes):
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].flatten()
        prefix = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(flat[:-1], dtype=np.int32)])
        offsets_list.append(prefix)
    return np.stack(offsets_list, axis=0)


def engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets):
    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.astype(np.int64)[np.newaxis, :, :] * multipliers[:, np.newaxis, :]

    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0].copy()
    for i in range(1, max_ngram_size):
        hashes = hashes ^ prod[:, :, i]
        for layer_idx in range(num_ngram_layers):
            vocab_row = vocab_sizes[layer_idx, i - 1].astype(np.int64)[np.newaxis, :]
            result = (hashes[layer_idx][:, np.newaxis] % vocab_row).astype(np.int32)
            ans[layer_idx].append(result)

    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = np.concatenate(ans[layer_idx], axis=-1)

    output = np.stack(ans, axis=0)
    return output + offsets[:, np.newaxis, :]


def gen_golden_data(case_name, param):
    dtype = param.dtype
    num_tokens = param.num_tokens

    ngram_token_ids = np.random.randint(0, 100000, size=(num_tokens, MAX_NGRAM_SIZE)).astype(np.int32)
    multipliers = np.random.randint(0, 100000, size=(NUM_NGRAM_LAYERS, MAX_NGRAM_SIZE)).astype(np.int64)
    vocab_sizes = np.random.randint(100000, 1000000, size=(NUM_NGRAM_LAYERS, MAX_NGRAM_SIZE - 1, NUM_EMBED_TABLE_PER_NGRAM)).astype(np.int32)
    offsets = make_offsets(vocab_sizes)

    golden = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)

    if dtype == np.int32:
        ngram_token_ids = ngram_token_ids.astype(np.int32)
        multipliers = multipliers.astype(np.int32)
        vocab_sizes = vocab_sizes.astype(np.int32)
        offsets = offsets.astype(np.int32)
        golden = golden.astype(np.int32)
    elif dtype == np.int64:
        ngram_token_ids = ngram_token_ids.astype(np.int64)
        multipliers = multipliers.astype(np.int64)
        vocab_sizes = vocab_sizes.astype(np.int64)
        offsets = offsets.astype(np.int64)
        golden = golden.astype(np.int64)

    ngram_token_ids.tofile("ngram_token_ids.bin")
    multipliers.tofile("multipliers.bin")
    vocab_sizes.tofile("vocab_sizes.bin")
    offsets.tofile("offsets.bin")
    golden.tofile("golden.bin")


class EngramHashParams:
    def __init__(self, dtype, num_tokens):
        self.dtype = dtype
        self.num_tokens = num_tokens


def generate_case_name(param):
    dtype_str = {
        np.int32: 'int32',
        np.int64: 'int64',
    }[param.dtype]
    return f"EngramHashTest.case_{dtype_str}_{param.num_tokens}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        EngramHashParams(np.int32, 1),
        EngramHashParams(np.int32, 16),
        EngramHashParams(np.int32, 128),
        EngramHashParams(np.int64, 1),
        EngramHashParams(np.int64, 16),
        EngramHashParams(np.int64, 128),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        case_dir = os.path.join(script_dir, case_name)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        original_dir = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)