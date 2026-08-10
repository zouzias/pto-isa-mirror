#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import math
import os

import en_dtypes
import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16


fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp4_e2m1x2 = en_dtypes.float4_e2m1
MX_SCALE_GROUP = 32
E8M0_BIAS = 127
E2M1_MAX_MAG = 6.0
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


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


def pack_fp4_nd(fp4_codes_1d):
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    assert len(flat) % 2 == 0
    fp4_bytes = bytearray()
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0x0F
        hi = int(flat[i + 1]) & 0x0F
        fp4_bytes.append(lo | (hi << 4))
    return bytes(fp4_bytes)


def _scale_exp_for_group(chunk_32):
    max_abs = float(np.abs(chunk_32).max())
    if max_abs == 0.0:
        return 0
    return int(math.ceil(math.log2(max_abs / E2M1_MAX_MAG)))


def e2m1_mx_quantize_group(chunk_32):
    scale_exp = _scale_exp_for_group(chunk_32)
    e8m0_byte = np.uint8(scale_exp + E8M0_BIAS)
    scale = np.float32(2.0) ** scale_exp
    scaled = (chunk_32.astype(np.float32) / scale).astype(bfloat16).astype(np.float32)
    nibbles = bf16_to_e2m1(scaled)
    return nibbles, e8m0_byte


def e2m1_mx_quantize(bf16_mat, group_axis="row"):
    """Quantize BF16 matrix to e2m1 nibbles + per-32 e8m0 scale bytes.

    group_axis="row": 32-element groups walk cols (the row-major ravel).
        Use for A [M, K] — groups walk K. Returns:
          codes_2d shape [M, K]   (matches A's GM layout, pack_fp4_nd directly)
          e8m0_2d shape [M, K/32] (feeds convert_x1_scale_format)
    group_axis="col": 32-element groups walk along the CONTRACTION axis (cols)
        per row of the matrix as stored, i.e. for B [K, N] we want groups of 32
        consecutive K-elements for each n-column. Transpose to [N, K] so the
        row-major ravel walks K for each n, quantize there, then transpose back.
        Returns:
          codes_2d shape [K, N]   (matches B's GM layout)
          e8m0_2d shape [K/32, N] (feeds convert_x2_scale_format)
    """
    rows, cols = bf16_mat.shape
    if group_axis == "col":
        work = bf16_mat.T.copy()  # [cols, rows]
    else:
        work = bf16_mat.copy()
    nr, nc = work.shape
    assert nc % MX_SCALE_GROUP == 0, f"cols={nc} not multiple of {MX_SCALE_GROUP}"

    flat = work.astype(np.float32).ravel()
    n_groups = len(flat) // MX_SCALE_GROUP
    code_flat = np.empty(len(flat), dtype=np.uint8)
    e8m0_flat = np.empty(n_groups, dtype=np.uint8)

    for g in range(n_groups):
        start = g * MX_SCALE_GROUP
        stop = start + MX_SCALE_GROUP
        codes, e = e2m1_mx_quantize_group(flat[start:stop])
        code_flat[start:stop] = codes
        e8m0_flat[g] = e

    codes_2d = code_flat.reshape(nr, nc)
    e8m0_2d = e8m0_flat.reshape(nr, nc // MX_SCALE_GROUP)
    if group_axis == "col":
        # codes were computed on the [cols, rows] transpose — restore [rows, cols].
        codes_2d = codes_2d.T.copy()
        # e8m0 was [cols, rows/32] in work orientation; transpose to [rows/32, cols]
        # so it is ready for convert_x2_scale_format (which expects [K_groups, N]).
        e8m0_2d = e8m0_2d.T.copy()
    return codes_2d, e8m0_2d


def decode_e2m1(codes_2d):
    sign = (codes_2d >> 3) & 1
    mag = E2M1_VALUES[codes_2d & 0x07]
    return np.where(sign == 1, -mag, mag).astype(np.float32)


def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0)
    else:
        padded = x1_mx_gm

    m_padded = m + pad_m
    k_padded = k + pad_k
    x1_scale_gm = padded.reshape((m_padded // block_size, block_size, k_padded // c0_size_mx, c0_size_mx))
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(x1_scale_gm.shape[0] * x1_scale_gm.shape[1], x1_scale_gm.shape[2] * x1_scale_gm.shape[3])
    return x1_scale_gm


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = x2_mx_gm

    k_padded, n_padded = padded.shape
    x2_scale_gm = padded.reshape((k_padded // c0_size_mx, c0_size_mx, n_padded // 16, 16)).transpose(2, 0, 3, 1)
    x2_scale_gm = x2_scale_gm.reshape(x2_scale_gm.shape[1] * x2_scale_gm.shape[3], x2_scale_gm.shape[0] * x2_scale_gm.shape[2])
    return x2_scale_gm


# Module-level RNG so A and B get different values. Seeded for reproducibility:
# the first make_bf16_matrix() call consumes the first chunk, the next call
# consumes the next chunk — never the same values. Byte-identical to the
# tmatmul_mx_hif4 / tmatmul_mx_e1m2 canonical version so the B input
# distribution matches across the MX testcases.
_BF16_RNG = np.random.default_rng(19)


def make_bf16_matrix(valid_m, valid_n, group_axis="row"):
    """Generate BF16 input with values in [-10, 10] and per-64-group magnitude
    variation along the requested axis.

    group_axis="col": per-64 groups run along the contraction axis (cols), so for
    B[K, N] each N-column's K-axis is split into 64-element blocks with distinct
    scale factors. This makes the e2m1 e8m0 scale bytes non-trivial (not all 127),
    which exercises the K-axis grouping in e2m1_mx_quantize.
    """
    total = valid_m * valid_n
    base = _BF16_RNG.uniform(-1.0, 1.0, size=total).astype(np.float32)

    if group_axis == "row":
        scales = np.ones(total, dtype=np.float32)
        gp64 = 64
        num_groups = (total + gp64 - 1) // gp64
        for g in range(num_groups):
            begin = g * gp64
            end = min(begin + gp64, total)
            scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
        values = base * scales
        return values.reshape(valid_m, valid_n).astype(bfloat16)

    # group_axis == "col": generate in transposed orientation, apply scales,
    # then transpose back to [valid_m, valid_n].
    values_t = base.reshape(valid_n, valid_m)
    scales = np.ones(total, dtype=np.float32)
    gp64 = 64
    num_groups = (total + gp64 - 1) // gp64
    for g in range(num_groups):
        begin = g * gp64
        end = min(begin + gp64, total)
        scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
    values_t = values_t * scales.reshape(valid_n, valid_m)
    return values_t.T.astype(bfloat16)


def gen_case(valid_m, valid_k, valid_n, out_dir, neutral_a_scale=False):
    assert valid_k % 64 == 0
    assert valid_k % MX_SCALE_GROUP == 0

    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(19)

    a_fp32 = np.random.uniform(-8.0, 8.0, (valid_m, valid_k)).astype(np.float32)
    a_fp8 = a_fp32.astype(fp8_e4m3fn)
    if neutral_a_scale:
        # Experiment A: every A-scale byte = 127 (e8m0 neutral, scale = 2^0 = 1.0).
        # Isolates whether the all-NaN MMAD_MX.E4M3E2M1 failure is caused by
        # specific A-scale VALUES being misread, or by the MX_A_ZZ A-scale
        # tile PRESENCE/ENCODING being incompatible with a non-fp4 A dtype.
        # With scale=1.0 the golden reduces to fp8(A) @ dequant_e2m1(B); any
        # NaN/Inf at MMAD_MX output then cannot come from scale values.
        a_scale = np.full((valid_m, valid_k // MX_SCALE_GROUP), E8M0_BIAS, dtype=np.uint8)
    else:
        a_scale = np.random.randint(126, 130, (valid_m, valid_k // MX_SCALE_GROUP), dtype=np.uint8)

    b_src = make_bf16_matrix(valid_k, valid_n, group_axis="col").astype(np.float32)
    b_codes, b_scale = e2m1_mx_quantize(b_src, group_axis="col")

    a_deq_scale = np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32)
    a_deq = a_fp8.astype(np.float32) * np.repeat(a_deq_scale, MX_SCALE_GROUP, axis=1)

    b_deq_scale = np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32)
    b_deq = decode_e2m1(b_codes) * np.repeat(b_deq_scale, MX_SCALE_GROUP, axis=0)

    golden_fp32 = (a_deq.astype(np.float32) @ b_deq.astype(np.float32)).astype(np.float32)
    golden_bf16 = golden_fp32.astype(bfloat16)

    b_data = pack_fp4_nd(b_codes.ravel())
    a_scale_zz = convert_x1_scale_format(a_scale, 16, 2)
    b_scale_nn = convert_x2_scale_format(b_scale, 16, 2)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_fp8.astype(np.uint8).tobytes())
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale_zz.tobytes())
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale_nn.tobytes())
    with open(os.path.join(out_dir, "golden_out.bin"), "wb") as f:
        f.write(golden_bf16.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden_bf16.tobytes())

    print(
        f"[{os.path.basename(out_dir)}] M={valid_m} K={valid_k} N={valid_n}: "
        f"a_data={valid_m * valid_k}B a_scale={a_scale_zz.nbytes}B "
        f"b_data={len(b_data)}B b_scale={b_scale_nn.nbytes}B golden={golden_bf16.nbytes}B"
    )


DEFAULT_CASES = [
    ("TMATMUL_MX_E4M3E2M1_TEST.case_e4m3e2m1_128x128x128_nd", 128, 128, 128, False),
    # Experiment A: neutral A-scale (all 127). If this case still produces
    # NaN at MMAD_MX output, the failure is the MX_A_ZZ A-scale tile being
    # incompatible with a non-fp4 A dtype (presence/encoding), NOT scale values.
    ("TMATMUL_MX_E4M3E2M1_TEST.case_e4m3e2m1_128x128x128_neutral_a_scale", 128, 128, 128, True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for name, m, k, n, neutral in DEFAULT_CASES:
            gen_case(m, k, n, os.path.join(script_dir, name), neutral_a_scale=neutral)
        # Backward-compat: also emit the baseline (non-neutral) golden at the
        # script dir, as run_st.py's in-tree direct execution expects.
        name0, m0, k0, n0, _ = DEFAULT_CASES[0]
        gen_case(m0, k0, n0, script_dir)
    else:
        name, m, k, n, neutral = DEFAULT_CASES[args.case]
        gen_case(m, k, n, os.path.join(script_dir, name), neutral_a_scale=neutral)


if __name__ == "__main__":
    main()
