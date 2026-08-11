#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# MMAD_MX experiment golden generator.
# Self-contained (no sibling imports) to avoid run_st.py build/ path issues.
#
# 4 cases, all 128x128x128 ND, e8m0 per-32-K scales (MX_A_ZZ / MX_B_NN):
#   E1: fp8_e4m3 x fp4_e2m1, varied scales   (failing baseline — reproduces NaN)
#   E2: fp8_e4m3 x fp4_e2m1, neutral A-scale (isolates A-scale values)
#   E3: fp8_e4m3 x fp4_e2m1, neutral B-scale (isolates B-scale values)
#   E4: fp4_e1m2x2 x fp4_e1m2x2, varied     (passing baseline — harness sanity)
#
# Note: fp8xfp8 and fp4xfp8 MX combos were found to be UNSUPPORTED by the
# mad_mx intrinsic (compile error: "3rd parameter maybe need a type
# __fp4e1m2x2"), so this experiment uses only known-compiling combos.
# The fp8-A NaN is isolated by varying scale VALUES on the compiling fp8xfp4
# combo, and contrasted against the passing fp4e1m2xfp4e1m2 baseline.

import math
import os

import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn

np.random.seed(19)

MX_SCALE_GROUP = 32
E8M0_BIAS = 127
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


# ============================================================
#  e2m1 (fp4) quantize + decode primitives
# ============================================================

def _scale_exp_for_group(chunk_32):
    max_abs = float(np.abs(chunk_32).max())
    if max_abs == 0.0:
        return 0
    return int(math.ceil(math.log2(max_abs / 6.0)))


def bf16_to_e2m1(scaled_flat):
    vals = np.asarray(scaled_flat, dtype=np.float32)
    sign = (vals < 0).astype(np.uint8)
    mag = np.abs(vals)
    codes = np.zeros(len(vals), dtype=np.uint8)
    for i, m in enumerate(mag):
        diffs = np.abs(E2M1_VALUES - m)
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        if len(candidates) == 1:
            best = int(candidates[0])
        else:
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


def e2m1_mx_quantize(bf16_mat, group_axis="row"):
    rows, cols = bf16_mat.shape
    work = bf16_mat.T.copy() if group_axis == "col" else bf16_mat.copy()
    nr, nc = work.shape
    flat = work.astype(np.float32).ravel()
    n_groups = len(flat) // MX_SCALE_GROUP
    code_flat = np.empty(len(flat), dtype=np.uint8)
    e8m0_flat = np.empty(n_groups, dtype=np.uint8)
    for g in range(n_groups):
        s, e = g * MX_SCALE_GROUP, (g + 1) * MX_SCALE_GROUP
        chunk = flat[s:e]
        exp = _scale_exp_for_group(chunk)
        scale = np.float32(2.0) ** exp
        scaled = (chunk / scale).astype(bfloat16).astype(np.float32)
        code_flat[s:e] = bf16_to_e2m1(scaled)
        e8m0_flat[g] = exp + E8M0_BIAS
    codes_2d = code_flat.reshape(nr, nc)
    e8m0_2d = e8m0_flat.reshape(nr, nc // MX_SCALE_GROUP)
    if group_axis == "col":
        codes_2d = codes_2d.T.copy()
        e8m0_2d = e8m0_2d.T.copy()
    return codes_2d, e8m0_2d


def decode_e2m1(codes_2d):
    sign = (codes_2d >> 3) & 1
    mag = E2M1_VALUES[codes_2d & 0x07]
    return np.where(sign == 1, -mag, mag).astype(np.float32)


# ============================================================
#  e1m2 (fp4) quantize + decode primitives (for E4 baseline)
# ============================================================

E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)


def _scale_exp_for_group_e1m2(chunk_32):
    max_abs = float(np.abs(chunk_32).max())
    if max_abs == 0.0:
        return 0
    return int(math.ceil(math.log2(max_abs / 1.75)))


def bf16_to_e1m2(scaled_flat):
    vals = np.asarray(scaled_flat, dtype=np.float32)
    sign = (vals < 0).astype(np.uint8)
    mag = np.abs(vals)
    codes = np.zeros(len(vals), dtype=np.uint8)
    for i, m in enumerate(mag):
        diffs = np.abs(E1M2_VALUES - m)
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        if len(candidates) == 1:
            best = int(candidates[0])
        else:
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


def e1m2_mx_quantize(bf16_mat, group_axis="row"):
    rows, cols = bf16_mat.shape
    work = bf16_mat.T.copy() if group_axis == "col" else bf16_mat.copy()
    nr, nc = work.shape
    flat = work.astype(np.float32).ravel()
    n_groups = len(flat) // MX_SCALE_GROUP
    code_flat = np.empty(len(flat), dtype=np.uint8)
    e8m0_flat = np.empty(n_groups, dtype=np.uint8)
    for g in range(n_groups):
        s, e = g * MX_SCALE_GROUP, (g + 1) * MX_SCALE_GROUP
        chunk = flat[s:e]
        exp = _scale_exp_for_group_e1m2(chunk)
        scale = np.float32(2.0) ** exp
        scaled = (chunk / scale).astype(bfloat16).astype(np.float32)
        code_flat[s:e] = bf16_to_e1m2(scaled)
        e8m0_flat[g] = exp + E8M0_BIAS
    codes_2d = code_flat.reshape(nr, nc)
    e8m0_2d = e8m0_flat.reshape(nr, nc // MX_SCALE_GROUP)
    if group_axis == "col":
        codes_2d = codes_2d.T.copy()
        e8m0_2d = e8m0_2d.T.copy()
    return codes_2d, e8m0_2d


def decode_e1m2(codes_2d):
    sign = (codes_2d >> 3) & 1
    mag = E1M2_VALUES[codes_2d & 0x07]
    return np.where(sign == 1, -mag, mag).astype(np.float32)


def pack_fp4_nd(fp4_codes_1d):
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    out = bytearray()
    for i in range(0, len(flat), 2):
        out.append(int(flat[i]) | (int(flat[i + 1]) << 4))
    return bytes(out)


# ============================================================
#  e8m0 scale fractal layouts (MX_A_ZZ / MX_B_NN) — from A5/e1m2
# ============================================================

def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    padded = np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0) if (pad_m or pad_k) else x1_mx_gm
    mp, kp = padded.shape
    x = padded.reshape((mp // block_size, block_size, kp // c0_size_mx, c0_size_mx)).transpose(0, 2, 1, 3)
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    padded = np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0) if (pad_n or pad_k) else x2_mx_gm
    kp, np_ = padded.shape
    x = padded.reshape((kp // c0_size_mx, c0_size_mx, np_ // 16, 16)).transpose(2, 0, 3, 1)
    return x.reshape(x.shape[1] * x.shape[3], x.shape[0] * x.shape[2])


# ============================================================
#  Per-case generation
# ============================================================

def _gen_fp8_input(m, k, lo, hi):
    src = np.random.uniform(lo, hi, (m, k)).astype(np.float32)
    return src.astype(float8_e4m3fn), src


def _gen_e2m1_input(m, k, group_axis="row"):
    """Generate bf16 input with per-64-group structure (non-trivial scales), quantize to e2m1."""
    total = m * k
    rng = np.random.default_rng(42 + m * 1000 + k)
    base = rng.uniform(-1.0, 1.0, total).astype(np.float32)
    scales = np.ones(total, dtype=np.float32)
    for g in range((total + 63) // 64):
        s, e = g * 64, min(g * 64 + 64, total)
        scales[s:e] = rng.uniform(0.5, 10.0)
    if group_axis == "col":
        vals = (base * scales).reshape(k, m).T.astype(bfloat16)
    else:
        vals = (base * scales).reshape(m, k).astype(bfloat16)
    codes, e8m0 = e2m1_mx_quantize(vals, group_axis=group_axis)
    return codes, e8m0, vals.astype(np.float32)


def _gen_e1m2_input(m, k, group_axis="row"):
    """Generate bf16 input with per-64-group structure, quantize to e1m2 (for E4 baseline)."""
    total = m * k
    rng = np.random.default_rng(99 + m * 1000 + k)
    base = rng.uniform(-1.0, 1.0, total).astype(np.float32)
    scales = np.ones(total, dtype=np.float32)
    for g in range((total + 63) // 64):
        s, e = g * 64, min(g * 64 + 64, total)
        scales[s:e] = rng.uniform(0.5, 4.0)
    if group_axis == "col":
        vals = (base * scales).reshape(k, m).T.astype(bfloat16)
    else:
        vals = (base * scales).reshape(m, k).astype(bfloat16)
    codes, e8m0 = e1m2_mx_quantize(vals, group_axis=group_axis)
    return codes, e8m0, vals.astype(np.float32)


def gen_case(case_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    M, K, N = 128, 128, 128

    if case_id == 1:  # fp8 x fp4_e2m1, varied scales (failing baseline)
        a_fp8, a_f32 = _gen_fp8_input(M, K, -8, 8)
        a_scale = np.random.randint(126, 130, (M, K // 32), dtype=np.uint8)
        b_codes, b_scale, _ = _gen_e2m1_input(K, N, "col")
        a_deq = a_f32 * np.repeat(np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=1)
        b_deq = decode_e2m1(b_codes) * np.repeat(np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=0)
        a_data = a_fp8.view(np.uint8).tobytes()
        b_data = pack_fp4_nd(b_codes.ravel())
    elif case_id == 2:  # fp8 x fp4_e2m1, neutral A-scale
        a_fp8, a_f32 = _gen_fp8_input(M, K, -8, 8)
        a_scale = np.full((M, K // 32), E8M0_BIAS, dtype=np.uint8)  # neutral
        b_codes, b_scale, _ = _gen_e2m1_input(K, N, "col")
        a_deq = a_f32  # scale = 1.0
        b_deq = decode_e2m1(b_codes) * np.repeat(np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=0)
        a_data = a_fp8.view(np.uint8).tobytes()
        b_data = pack_fp4_nd(b_codes.ravel())
    elif case_id == 3:  # fp8 x fp4_e2m1, neutral B-scale
        a_fp8, a_f32 = _gen_fp8_input(M, K, -8, 8)
        a_scale = np.random.randint(126, 130, (M, K // 32), dtype=np.uint8)
        b_codes, _, _ = _gen_e2m1_input(K, N, "col")
        b_scale = np.full((K // 32, N), E8M0_BIAS, dtype=np.uint8)  # neutral
        a_deq = a_f32 * np.repeat(np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=1)
        b_deq = decode_e2m1(b_codes)  # scale = 1.0
        a_data = a_fp8.view(np.uint8).tobytes()
        b_data = pack_fp4_nd(b_codes.ravel())
    else:  # case_id == 4: fp4_e1m2 x fp4_e1m2, varied scales (passing baseline)
        a_codes, a_scale, _ = _gen_e1m2_input(M, K, "row")
        b_codes, b_scale, _ = _gen_e1m2_input(K, N, "col")
        a_deq = decode_e1m2(a_codes) * np.repeat(np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=1)
        b_deq = decode_e1m2(b_codes) * np.repeat(np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), 32, axis=0)
        a_data = pack_fp4_nd(a_codes.ravel())
        b_data = pack_fp4_nd(b_codes.ravel())

    golden = (a_deq.astype(np.float32) @ b_deq.astype(np.float32)).astype(np.float32)
    a_scale_zz = convert_x1_scale_format(a_scale, 16, 2)
    b_scale_nn = convert_x2_scale_format(b_scale, 16, 2)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_data)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale_zz.tobytes())
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale_nn.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden.tobytes())

    tag = {1: "fp8xfp4_varied", 2: "fp8xfp4_neutralA", 3: "fp8xfp4_neutralB", 4: "fp4e1m2xfp4e1m2"}[case_id]
    print(f"[E{case_id} {tag}] a_data={len(a_data)}B a_scale={a_scale_zz.nbytes}B "
          f"b_data={len(b_data)}B b_scale={b_scale_nn.nbytes}B golden={golden.nbytes}B")


CASES = [
    ("TMATMUL_MX_EXP_TEST.case_e1_fp8xfp4_varied", 1),
    ("TMATMUL_MX_EXP_TEST.case_e2_fp8xfp4_neutral_a", 2),
    ("TMATMUL_MX_EXP_TEST.case_e3_fp8xfp4_neutral_b", 3),
    ("TMATMUL_MX_EXP_TEST.case_e4_fp4e1m2xfp4e1m2_baseline", 4),
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for name, cid in CASES:
            gen_case(cid, os.path.join(script_dir, name))
        # Also emit E1 at script dir (run_st.py fallback)
        gen_case(CASES[0][1], script_dir)
    else:
        name, cid = CASES[args.case]
        gen_case(cid, os.path.join(script_dir, name))


if __name__ == "__main__":
    main()
