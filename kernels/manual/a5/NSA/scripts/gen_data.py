#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Generate NSA (Native Sparse Attention) golden data for A5 DN flash-attention branches.

Reference: Projects/attention/pytorch/NSA.py

Each branch runs standard flash attention (same golden path as flash_atten) over branch-specific K/V.
Block compression, top-n selection, and gating are computed in Python; the A5 kernel reuses the
MHA DN flash-attention pipeline three times and fuses branch outputs with learned gates on the host.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

np.random.seed(7)

HEAD_SIZE = 128
TILE_S1_DEFAULT = 128


def block_compress(k_blocks, v_blocks, head_size, block_size, compress_dim):
    """MLP compression matching NSA.py BlockCompress (ReLU)."""
    num_blocks = k_blocks.shape[0]
    flat_dim = block_size * head_size
    w1 = (np.random.randn(flat_dim, compress_dim).astype(np.float16) * 0.02).astype(np.float32)
    b1 = np.zeros(compress_dim, dtype=np.float32)
    w2 = (np.random.randn(compress_dim, head_size).astype(np.float16) * 0.02).astype(np.float32)
    b2 = np.zeros(head_size, dtype=np.float32)

    def mlp(x):
        h = np.maximum(x @ w1 + b1, 0.0)
        return h @ w2 + b2

    k_flat = k_blocks.reshape(num_blocks, flat_dim).astype(np.float32)
    v_flat = v_blocks.reshape(num_blocks, flat_dim).astype(np.float32)
    return mlp(k_flat), mlp(v_flat), (w1, b1, w2, b2)


def flash_attention_golden(q_fp32, k_fp32, v_fp32, head_size, tile_s1=128, is_causal=False):
    """Streaming flash-attention golden for one branch; returns final O [S0, H]."""
    s0, s1 = q_fp32.shape[0], k_fp32.shape[0]
    scores = q_fp32 @ k_fp32.T
    if is_causal:
        mask = np.triu((np.ones(scores.shape) * float(-3.40282e+38)).astype(np.float32), 1)
        scores += mask
    scale = 1.0 / np.sqrt(head_size)
    num_tiles = s1 // tile_s1
    assert s1 % tile_s1 == 0

    global_max = None
    global_sum = None
    o_running = np.zeros((s0, head_size), dtype=np.float32)
    exp_max_parts = []
    global_sums = []

    for ti in range(num_tiles):
        c0 = ti * tile_s1
        tile = scores[:, c0:c0 + tile_s1]
        local_max = np.max(tile, axis=1, keepdims=True).astype(np.float32)
        if global_max is not None:
            local_max = np.maximum(local_max, global_max).astype(np.float32)
        if ti == 0:
            new_global_max = local_max
            tmp_float_exp = np.exp((tile - new_global_max) * scale).astype(np.float32)
            new_global_sum = np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32)
            exp_max_tile = np.ones_like(new_global_max, dtype=np.float32)
        else:
            exp_max = np.exp((global_max - local_max) * scale).astype(np.float32)
            new_global_max = local_max
            tmp_float_exp = np.exp((tile - new_global_max) * scale).astype(np.float32)
            new_global_sum = exp_max * global_sum + np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32)
            exp_max_tile = exp_max

        soft_tile = tmp_float_exp
        v_tile = v_fp32[c0:c0 + tile_s1, :]
        pv_tile = soft_tile @ v_tile
        if ti == 0:
            o_running = pv_tile.copy()
        else:
            o_running = exp_max_tile * o_running + pv_tile
            if ti == num_tiles - 1:
                o_running = o_running / new_global_sum

        exp_max_parts.append(exp_max_tile.reshape(-1))
        global_sums.append(new_global_sum.reshape(-1))
        global_max = new_global_max
        global_sum = new_global_sum

    return o_running.astype(np.float32)


def write_branch_kv(path, prefix, k_fp32, v_fp32):
    """Write branch K/V tensors. k_fp32 is [S1, HEAD]; kt.bin follows flash_atten layout."""
    k = k_fp32.astype(np.float16)
    v = v_fp32.astype(np.float16)
    k.tofile(os.path.join(path, f'{prefix}_k.bin'))
    v.tofile(os.path.join(path, f'{prefix}_v.bin'))
    # flash_atten stores kt.bin as [S1, HEAD] (k.T where k.bin is [HEAD, S1]).
    # Branch k_fp32 is already [S1, HEAD], so write it directly to kt.bin.
    k.tofile(os.path.join(path, f'{prefix}_kt.bin'))


def gen_case(path, s0, s1_full, head_size, block_size, num_selected, select_block_size,
             window_size, compress_dim, tile_s1=TILE_S1_DEFAULT, is_causal=False):
    assert s1_full % block_size == 0
    num_blocks = s1_full // block_size
    s1_cmp = num_blocks
    s1_slc = num_selected * select_block_size
    s1_win = window_size

    q_fp32 = (np.random.randn(s0, head_size).astype(np.float16) * 1.2).astype(np.float32)
    q = q_fp32.astype(np.float16)

    k_cmp_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)
    v_cmp_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)
    k_slc_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)
    v_slc_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)
    k_win_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)
    v_win_raw = (np.random.randn(s1_full, head_size).astype(np.float16) * 1.2).astype(np.float32)

    usable_len = num_blocks * block_size
    k_blocks = k_cmp_raw[:usable_len, :].reshape(num_blocks, block_size, head_size)
    v_blocks = v_cmp_raw[:usable_len, :].reshape(num_blocks, block_size, head_size)
    k_cmp_c, v_cmp_c, _ = block_compress(k_blocks, v_blocks, head_size, block_size, compress_dim)

    k_cmp_rep = k_cmp_c
    v_cmp_rep = v_cmp_c
    cmp_scores = q_fp32 @ k_cmp_rep.T / np.sqrt(head_size)
    cmp_probs = np.exp(cmp_scores - np.max(cmp_scores, axis=-1, keepdims=True))
    cmp_probs = cmp_probs / np.sum(cmp_probs, axis=-1, keepdims=True)
    block_importance = cmp_probs.sum(axis=0)

    if select_block_size >= block_size and select_block_size % block_size == 0:
        ratio = select_block_size // block_size
        num_select_blocks = num_blocks // ratio
        block_importance = block_importance[:num_select_blocks * ratio].reshape(num_select_blocks, ratio).sum(axis=1)
    else:
        num_select_blocks = s1_full // select_block_size

    topn = min(num_selected, num_select_blocks)
    top_indices = np.argsort(-block_importance)[:topn]

    usable_slc_len = num_select_blocks * select_block_size
    k_blocks_slc = k_slc_raw[:usable_slc_len, :].reshape(num_select_blocks, select_block_size, head_size)
    v_blocks_slc = v_slc_raw[:usable_slc_len, :].reshape(num_select_blocks, select_block_size, head_size)
    k_slc_list = []
    v_slc_list = []
    for idx in top_indices:
        k_slc_list.append(k_blocks_slc[idx].reshape(select_block_size, head_size))
        v_slc_list.append(v_blocks_slc[idx].reshape(select_block_size, head_size))
    k_slc = np.concatenate(k_slc_list, axis=0)
    v_slc = np.concatenate(v_slc_list, axis=0)

    win_start = max(0, s1_full - window_size)
    k_win = k_win_raw[win_start:, :]
    v_win = v_win_raw[win_start:, :]

    gates = 1.0 / (1.0 + np.exp(-(np.random.randn(s0, 3).astype(np.float32) * 0.5)))
    g_cmp = gates[:, 0:1]
    g_slc = gates[:, 1:2]
    g_win = gates[:, 2:3]

    o_cmp = flash_attention_golden(q_fp32, k_cmp_rep, v_cmp_rep, head_size, tile_s1, is_causal)
    o_slc = flash_attention_golden(q_fp32, k_slc, v_slc, head_size, tile_s1, False)
    o_win = flash_attention_golden(q_fp32, k_win, v_win, head_size, tile_s1, False)
    o_final = g_cmp * o_cmp + g_slc * o_slc + g_win * o_win

    q.tofile(os.path.join(path, 'q.bin'))
    write_branch_kv(path, 'cmp', k_cmp_rep, v_cmp_rep)
    write_branch_kv(path, 'slc', k_slc, v_slc)
    write_branch_kv(path, 'win', k_win, v_win)
    gates.astype(np.float32).tofile(os.path.join(path, 'gates.bin'))
    top_indices.astype(np.int32).tofile(os.path.join(path, 'selected_blocks.bin'))
    o_cmp.tofile(os.path.join(path, 'o_cmp.bin'))
    o_slc.tofile(os.path.join(path, 'o_slc.bin'))
    o_win.tofile(os.path.join(path, 'o_win.bin'))
    o_final.astype(np.float32).tofile(os.path.join(path, 'o.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate NSA golden data")
    parser.add_argument("--case", dest="case_name")
    parser.add_argument("--cases", action="append")
    parser.add_argument("--cases-json", dest="cases_json", default=None)
    parser.add_argument("--causal-mask", type=int, default=0)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    default_json = script_root / "generated_cases.json"

    def load_cases():
        if args.cases_json or default_json.exists():
            json_path = Path(args.cases_json) if args.cases_json else default_json
            payload = json.loads(json_path.read_text())
            out = []
            for entry in payload:
                out.append((
                    entry["name"],
                    entry["s0"],
                    entry["head_size"],
                    entry["s1"],
                    entry["block_size"],
                    entry["num_selected"],
                    entry["select_block_size"],
                    entry["window_size"],
                    entry["compress_dim"],
                    entry.get("tile_s1", TILE_S1_DEFAULT),
                ))
            return out
        return [
            ('case_float_H_128_S0_128_S1_512_B4_N32_W128', 128, 128, 512, 4, 32, 4, 128, 256, TILE_S1_DEFAULT),
        ]

    cases = load_cases()
    if args.case_name:
        cases = [c for c in cases if c[0] == args.case_name] or cases

    build_dir = script_root / "build"
    for name, s0, head, s1, block_size, num_sel, sel_block, window, compress_dim, tile_s1 in cases:
        case_dir = build_dir / name
        os.makedirs(case_dir, exist_ok=True)
        gen_case(
            str(case_dir), s0, s1, head, block_size, num_sel, sel_block, window, compress_dim,
            tile_s1, bool(args.causal_mask),
        )
