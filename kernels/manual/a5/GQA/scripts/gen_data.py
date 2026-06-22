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
Generate GQA (Grouped Query Attention) inputs and golden outputs.

For each query head h with kv_group g = h / num_queries_per_kv:
  - Q[h] [S0, HEAD_SIZE]
  - K[g] [HEAD_SIZE, S1] (transposed for kernel input as kt)
  - V[g] [S1, HEAD_SIZE]
  - O[h] = softmax(Q[h] @ K[g]^T / sqrt(head_size)) @ V[g]

Per-head computation follows the SAME tiled softmax recurrence as flash_atten's gen_data.py.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

np.random.seed(7)

HEAD_SIZE = 128
TILE_S1_DEFAULT = 128


def gen_per_head_data(path, q_h, k_g, v_g, s0, s1, head_size, cube_s1=128, tile_s1=TILE_S1_DEFAULT,
                      is_causal=False):
    assert s1 % tile_s1 == 0, "S1 must be divisible by TILE_S1"
    assert tile_s1 % cube_s1 == 0, "TILE_S1 must be divisible by CUBE_S1"

    q_fp32 = q_h.astype(np.float32)
    k_fp32 = k_g.astype(np.float32)
    golden = q_fp32.dot(k_fp32).astype(np.float32)

    q_h_fp16 = q_fp32.astype(np.float16)
    k_g_fp16 = k_fp32.astype(np.float16)
    kt_g_fp16 = k_fp32.T.astype(np.float16)

    q_h_fp16.tofile(os.path.join(path, 'q.bin'))
    k_g_fp16.tofile(os.path.join(path, 'k.bin'))
    kt_g_fp16.tofile(os.path.join(path, 'kt.bin'))
    golden.tofile(os.path.join(path, 'qk.bin'))

    arr_f32 = golden.astype(np.float32)
    if is_causal:
        mask = np.triu((np.ones(arr_f32.shape) * float(-3.40282e+38)).astype(np.float32), 1)
        arr_f32 += mask
    scale = 1 / np.sqrt(head_size)
    num_tiles = s1 // tile_s1

    full_exp = np.zeros((s0, s1), dtype=np.float32)
    global_sums = []
    exp_max_parts = []
    global_max = None
    global_sum = None

    for ti in range(num_tiles):
        c0 = ti * tile_s1
        c1 = c0 + tile_s1
        tile = arr_f32[:, c0:c1]
        local_max = np.max(tile, axis=1, keepdims=True).astype(np.float32)
        if global_max is not None:
            local_max = np.maximum(local_max, global_max).astype(np.float32)
        if ti == 0:
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32)
            exp_max_tile = np.ones_like(new_global_max).astype(np.float32)
        else:
            exp_max = (global_max - local_max).astype(np.float32)
            exp_max = np.exp(exp_max * scale).astype(np.float32)
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = exp_max * global_sum + np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32)
            exp_max_tile = exp_max

        full_exp[:, c0:c1] = tmp_float_exp
        global_sums.append(new_global_sum.reshape(-1))
        exp_max_parts.append(exp_max_tile.reshape(-1))
        global_max = new_global_max
        global_sum = new_global_sum

    soft = full_exp.astype(np.float16)
    soft.tofile(os.path.join(path, 'p.bin'))
    full_exp.tofile(os.path.join(path, 'p_fp32.bin'))

    v_g_fp16 = v_g.astype(np.float16)
    vt_g_fp16 = v_g.T.astype(np.float16)

    soft_f32 = soft.astype(np.float32)
    v_fp32 = v_g.astype(np.float32)
    pv = np.zeros((s0, head_size), dtype=np.float32)
    pv_tile_fifo_parts = []
    for ti in range(num_tiles):
        c0 = ti * tile_s1
        soft_tile = soft_f32[:, c0:c0 + tile_s1]
        v_tile = v_fp32[c0:c0 + tile_s1, :]
        pv_tile_fifo = soft_tile.dot(v_tile).astype(np.float32)
        pv_tile_fifo_parts.append(pv_tile_fifo)
        pv += pv_tile_fifo

    v_g_fp16.tofile(os.path.join(path, 'v.bin'))
    vt_g_fp16.tofile(os.path.join(path, 'vt.bin'))
    pv.tofile(os.path.join(path, 'pv.bin'))
    for idx, part in enumerate(pv_tile_fifo_parts):
        part.tofile(os.path.join(path, f'pv_tile_fifo{idx}.bin'))
    for idx, g in enumerate(global_sums):
        g.astype(np.float32).tofile(os.path.join(path, f'global_sum_part{idx}.bin'))
    for idx, e in enumerate(exp_max_parts):
        e.astype(np.float32).tofile(os.path.join(path, f'exp_max_part{idx}.bin'))

    o_running = np.zeros((s0, head_size), dtype=np.float32)
    for ti, part in enumerate(pv_tile_fifo_parts):
        if ti == 0:
            o_running = part.copy()
        else:
            exp_max_tile = exp_max_parts[ti].reshape((s0, 1)).astype(np.float32)
            o_running = exp_max_tile * o_running + part
            if ti == num_tiles - 1:
                new_global_sum_tile = global_sums[ti].reshape((s0, 1)).astype(np.float32)
                o_running = o_running / new_global_sum_tile
        o_running.astype(np.float32).tofile(os.path.join(path, f'o_part{ti}.bin'))
    o_running.astype(np.float32).tofile(os.path.join(path, 'o.bin'))


def gen_gqa_case(path, num_q_heads, num_kv_heads, s0, s1, head_size=HEAD_SIZE,
                 cube_s1=128, tile_s1=TILE_S1_DEFAULT, is_causal=False):
    num_queries_per_kv = num_q_heads // num_kv_heads

    q_all = (np.random.randn(num_q_heads, s0, head_size).astype(np.float16) * 1.5).astype(np.float32)
    k_all = (np.random.randn(num_kv_heads, head_size, s1).astype(np.float16) * 1.5).astype(np.float32)
    v_all = (np.random.randn(num_kv_heads, s1, head_size).astype(np.float16) * 1.2).astype(np.float32)

    q_all_fp16 = q_all.astype(np.float16)
    k_all_fp16 = k_all.astype(np.float16)
    v_all_fp16 = v_all.astype(np.float16)

    kt_all = np.transpose(k_all, (0, 2, 1)).astype(np.float16)
    kt_all.tofile(os.path.join(path, 'kt_all.bin'))

    q_all_fp16.tofile(os.path.join(path, 'q_all.bin'))
    k_all_fp16.tofile(os.path.join(path, 'k_all.bin'))
    v_all_fp16.tofile(os.path.join(path, 'v_all.bin'))

    o_all = np.zeros((num_q_heads, s0, head_size), dtype=np.float32)

    for kv_group in range(num_kv_heads):
        k_g = k_all[kv_group]
        v_g = v_all[kv_group]

        for q_idx in range(num_queries_per_kv):
            q_head = kv_group * num_queries_per_kv + q_idx
            q_h = q_all[q_head]

            head_dir = os.path.join(path, f"h_q{q_head}_g_kv{kv_group}")
            os.makedirs(head_dir, exist_ok=True)

            gen_per_head_data(head_dir, q_h, k_g, v_g, s0, s1, head_size, cube_s1, tile_s1, is_causal)

            golden_o = np.fromfile(os.path.join(head_dir, 'o.bin'), dtype=np.float32)
            golden_o = golden_o.reshape(s0, head_size)
            o_all[q_head] = golden_o

    o_all.astype(np.float32).tofile(os.path.join(path, 'o_all.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate GQA golden data")
    parser.add_argument("--case", dest="case_name", help="Generate only the specified case name")
    parser.add_argument(
        "--cases",
        action="append",
        help="Case entry NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1] (repeatable)",
    )
    parser.add_argument("--cases-json", dest="cases_json", default=None)
    parser.add_argument("--num-q-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-size", type=int)
    parser.add_argument("--s0", type=int)
    parser.add_argument("--s1", type=int)
    parser.add_argument("--causal-mask", type=int)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    default_json = script_root / "generated_cases.json"

    def parse_case_entry(entry: str):
        parts = [p.strip() for p in entry.split(',') if p.strip()]
        if len(parts) < 6:
            raise ValueError(
                "Case entry must be NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1[,CUBE_S0[,TILE_S1]]"
            )
        num_q_heads, num_kv_heads, head_size, s0, s1 = map(int, parts[:5])
        cube_s0 = int(parts[5]) if len(parts) >= 6 else s0
        if s0 % cube_s0 != 0:
            raise ValueError("S0 must be divisible by CUBE_S0")
        tile_s1 = int(parts[6]) if len(parts) >= 7 else TILE_S1_DEFAULT
        cube_s1 = 128
        return num_q_heads, num_kv_heads, head_size, s0, s1, cube_s1, tile_s1

    cases = []
    if args.cases:
        for entry in args.cases:
            num_q, num_kv, head, s0, s1, cube_s1, tile_s1 = parse_case_entry(entry)
            name = f"case_float_GQA_Q{num_q}_K{num_kv}_H{head}_S0_{s0}_S1_{s1}"
            cases.append((name, (num_q, num_kv, s0, head, s1, cube_s1, tile_s1)))
    elif args.num_q_heads and args.num_kv_heads and args.head_size and args.s0 and args.s1:
        cases.append((
            f"case_float_GQA_Q{args.num_q_heads}_K{args.num_kv_heads}_H{args.head_size}_S0_{args.s0}_S1_{args.s1}",
            (args.num_q_heads, args.num_kv_heads, args.s0, args.head_size, args.s1, 128, TILE_S1_DEFAULT),
        ))
    elif args.cases_json or default_json.exists():
        json_path = Path(args.cases_json) if args.cases_json else default_json
        payload = json.loads(json_path.read_text())
        for entry in payload:
            cases.append((
                entry["name"],
                (
                    entry["num_q_heads"],
                    entry["num_kv_heads"],
                    entry["s0"],
                    entry["head_size"],
                    entry["s1"],
                    entry.get("cube_s1", 128),
                    entry.get("tile_s1", TILE_S1_DEFAULT),
                ),
            ))
    else:
        cases = [
            ('case_float_GQA_Q8_K2_H128_S0_128_S1_512', (8, 2, 128, 128, 512, 128, TILE_S1_DEFAULT)),
            ('case_float_GQA_Q8_K2_H128_S0_128_S1_1024', (8, 2, 128, 128, 1024, 128, TILE_S1_DEFAULT)),
        ]

    if args.case_name:
        target = args.case_name
        if target.endswith('_precision_debug'):
            target = target[:-len('_precision_debug')]
        filtered = [entry for entry in cases if entry[0] == target]
        if filtered:
            cases = filtered
        else:
            try:
                num_q, num_kv, head, s0, s1, cube_s1, tile_s1 = parse_case_entry(target)
                synthetic_name = f"case_float_GQA_Q{num_q}_K{num_kv}_H{head}_S0_{s0}_S1_{s1}"
                cases = [(synthetic_name, (num_q, num_kv, s0, head, s1, cube_s1, tile_s1))]
            except Exception as exc:
                raise ValueError(f"Requested case '{args.case_name}' not found: {exc}") from exc

    build_dir = script_root / "build"
    for name, (num_q_heads, num_kv_heads, s0, head_size, s1, cube_s1, tile_s1) in cases:
        case_dir = build_dir / name
        os.makedirs(case_dir, exist_ok=True)
        gen_gqa_case(
            str(case_dir), num_q_heads, num_kv_heads, s0, s1, head_size, cube_s1, tile_s1,
            bool(args.causal_mask),
        )
