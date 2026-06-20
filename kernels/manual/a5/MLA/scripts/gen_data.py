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
Generate MLA (weight-absorbed inference) inputs and golden outputs for DN flash attention.

Inference layout (single query head):
  - q_absorbed [S0, kv_latent_dim] = q_raw @ W_uk  (W_uk absorbed into query offline)
  - c_kv       [S1, kv_latent_dim] compressed KV cache
  - W_uv       [kv_latent_dim, head_size] V up-projection (used on-NPU to reconstruct V)
  - scores     q_absorbed @ c_kv^T
  - V          c_kv @ W_uv
  - O          softmax(scores / sqrt(head_size)) @ V
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

np.random.seed(7)

HEAD_SIZE = 128
KV_LATENT_DIM = 256
TILE_S1_DEFAULT = 128


def gen_case(path, s0, s1, head_size=HEAD_SIZE, kv_latent_dim=KV_LATENT_DIM,
             cube_s1=128, tile_s1=TILE_S1_DEFAULT, is_causal=False):
    assert s1 % tile_s1 == 0, "S1 must be divisible by TILE_S1"
    assert tile_s1 % cube_s1 == 0, "TILE_S1 must be divisible by CUBE_S1"

    # Random per-head query, compressed KV cache, and up-projection weights.
    q_raw_fp32 = (np.random.randn(s0, head_size).astype(np.float16) * 1.5).astype(np.float32)
    c_kv_fp32 = (np.random.randn(s1, kv_latent_dim).astype(np.float16) * 1.2).astype(np.float32)
    w_uk_fp32 = (np.random.randn(head_size, kv_latent_dim).astype(np.float16) * 0.05).astype(np.float32)
    w_uv_fp32 = (np.random.randn(kv_latent_dim, head_size).astype(np.float16) * 0.05).astype(np.float32)

    # Weight absorption: merge W_uk into query so K reconstruction is skipped at inference.
    q_absorbed_fp32 = q_raw_fp32.astype(np.float32).dot(w_uk_fp32.astype(np.float32))
    q_absorbed = q_absorbed_fp32.astype(np.float16)
    c_kv = c_kv_fp32.astype(np.float16)
    w_uv = w_uv_fp32.astype(np.float16)

    # QK scores use absorbed query against compressed KV (latent inner dim).
    golden = (q_absorbed_fp32.dot(c_kv_fp32.T)).astype(np.float32)

    q_absorbed.tofile(os.path.join(path, 'q.bin'))
    c_kv.tofile(os.path.join(path, 'c_kv.bin'))
    w_uv.tofile(os.path.join(path, 'w_uv.bin'))
    c_kv_t = c_kv.T.astype(np.float16)
    c_kv_t.tofile(os.path.join(path, 'c_kv_t.bin'))
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

    # Reconstruct V from compressed KV: V[s] = c_kv[s] @ W_uv.
    v_fp32 = c_kv_fp32.dot(w_uv_fp32.astype(np.float32))
    v = v_fp32.astype(np.float16)

    soft_f32 = soft.astype(np.float32)
    pv = np.zeros((s0, head_size), dtype=np.float32)
    pv_tile_fifo_parts = []
    for ti in range(num_tiles):
        c0 = ti * tile_s1
        soft_tile = soft_f32[:, c0:c0 + tile_s1]
        v_tile = v[c0:c0 + tile_s1, :].astype(np.float32)
        pv_tile_fifo = soft_tile.dot(v_tile).astype(np.float32)
        pv_tile_fifo_parts.append(pv_tile_fifo)
        pv += pv_tile_fifo

    v.tofile(os.path.join(path, 'v.bin'))
    v.T.astype(np.float16).tofile(os.path.join(path, 'vt.bin'))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MLA golden data")
    parser.add_argument("--case", dest="case_name", help="Generate only the specified case name")
    parser.add_argument(
        "--cases",
        action="append",
        help="Case entry HEAD_SIZE,KV_LATENT_DIM,S0,S1,CUBE_S0[,TILE_S1] (repeatable)",
    )
    parser.add_argument("--cases-json", dest="cases_json", default=None)
    parser.add_argument("--head-size", type=int)
    parser.add_argument("--kv-latent-dim", type=int)
    parser.add_argument("--s0", type=int)
    parser.add_argument("--s1", type=int)
    parser.add_argument("--causal-mask", type=int)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    default_json = script_root / "generated_cases.json"

    def parse_case_entry(entry: str):
        parts = [p.strip() for p in entry.split(',') if p.strip()]
        if len(parts) < 5:
            raise ValueError(
                "Case entry must be HEAD_SIZE,KV_LATENT_DIM,S0,S1[,CUBE_S0[,TILE_S1]]"
            )
        head, kv_latent, s0, s1 = map(int, parts[:4])
        cube_s0 = int(parts[4]) if len(parts) >= 5 else s0
        if s0 % cube_s0 != 0:
            raise ValueError("S0 must be divisible by CUBE_S0")
        tile_s1 = int(parts[5]) if len(parts) >= 6 else TILE_S1_DEFAULT
        cube_s1 = 128
        return head, kv_latent, s0, s1, cube_s1, tile_s1

    cases = []
    if args.cases:
        for entry in args.cases:
            head, kv_latent, s0, s1, cube_s1, tile_s1 = parse_case_entry(entry)
            name = f"case_float_H_{head}_L_{kv_latent}_S0_{s0}_S1_{s1}"
            cases.append((name, (s0, head, kv_latent, s1, cube_s1, tile_s1)))
    elif args.head_size and args.kv_latent_dim and args.s0 and args.s1:
        cases.append((
            f"case_float_H_{args.head_size}_L_{args.kv_latent_dim}_S0_{args.s0}_S1_{args.s1}",
            (args.s0, args.head_size, args.kv_latent_dim, args.s1, 128, TILE_S1_DEFAULT),
        ))
    elif args.cases_json or default_json.exists():
        json_path = Path(args.cases_json) if args.cases_json else default_json
        payload = json.loads(json_path.read_text())
        for entry in payload:
            cases.append((
                entry["name"],
                (
                    entry["s0"],
                    entry["head_size"],
                    entry["kv_latent_dim"],
                    entry["s1"],
                    entry.get("cube_s1", 128),
                    entry.get("tile_s1", TILE_S1_DEFAULT),
                ),
            ))
    else:
        cases = [
            ('case_float_H_128_L_256_S0_128_S1_512', (128, 128, 256, 512, 128, TILE_S1_DEFAULT)),
            ('case_float_H_128_L_256_S0_128_S1_1024', (128, 128, 256, 1024, 128, TILE_S1_DEFAULT)),
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
                head, kv_latent, s0, s1, cube_s1, tile_s1 = parse_case_entry(target)
                synthetic_name = f"case_float_H_{head}_L_{kv_latent}_S0_{s0}_S1_{s1}"
                cases = [(synthetic_name, (s0, head, kv_latent, s1, cube_s1, tile_s1))]
            except Exception as exc:
                raise ValueError(f"Requested case '{args.case_name}' not found: {exc}") from exc

    build_dir = script_root / "build"
    for name, (s0, head_size, kv_latent_dim, s1, cube_s1, tile_s1) in cases:
        case_dir = build_dir / name
        os.makedirs(case_dir, exist_ok=True)
        gen_case(
            str(case_dir), s0, s1, head_size, kv_latent_dim, cube_s1, tile_s1,
            bool(args.causal_mask),
        )
