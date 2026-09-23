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

import math
import os

import numpy as np
from ml_dtypes import bfloat16

np.random.seed(19)

HIF4_SCALE_GROUP = 64
GP4_SIZE = 4
GP8_SIZE = 8
GP64_SIZE = 64

E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)


def _bf16(x):
    return x.astype(np.float32).astype(bfloat16).astype(np.float32)


def bf16_to_e6m2(ma_flat):
    """Emulates the CCE intrinsic vcvt_bf162e6m2(..., ROUND_R, PART_EVEN)."""
    ma = np.abs(ma_flat).astype(np.float32).astype(bfloat16).astype(np.float32)
    recp_7 = np.float32(1.0 / 7.0).astype(bfloat16).astype(np.float32)
    e6m2_codes = np.zeros(len(ma), dtype=np.uint8)
    for i, v in enumerate(ma):
        if v == 0.0:
            continue
        sf = (v * recp_7).astype(bfloat16).astype(np.float32)
        if sf == 0.0:
            continue
        exp_raw = int(math.floor(math.log2(sf)))
        exp_raw = max(-48, min(exp_raw, 15))
        biased_exp = exp_raw + 48
        mantissa_frac = sf / (2.0**exp_raw)
        mant_real = (mantissa_frac - 1.0) * 4
        mant_bits = int(round(mant_real))
        if mant_bits >= 4:
            mant_bits = 0
            biased_exp += 1
        if biased_exp > 63 or (biased_exp == 63 and mant_bits == 3):
            biased_exp = 63
            mant_bits = 2
        e6m2_codes[i] = (biased_exp << 2) | mant_bits
    return e6m2_codes


def e6m2_code_to_value(codes):
    vals = np.zeros(len(codes), dtype=np.float64)
    for i, c in enumerate(codes):
        exp = int((c >> 2) & 0x3F)
        mant = int(c & 0x03)
        if exp == 0 and mant == 0:
            continue
        vals[i] = (1.0 + mant * 0.25) * (2.0 ** (exp - 48))
    return vals


def e6m2_code_to_reciprocal_bf16(codes):
    vals = e6m2_code_to_value(codes)
    with np.errstate(divide="ignore"):
        recips = np.where(vals > 0, 1.0 / vals, 0.0)
    return recips.astype(np.float32).astype(bfloat16)


def bf16_to_e1m2_hif4(scaled_flat):
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


def hif4_quantize(src_bf16):
    """Three-level Ea/Eb/Ec HiF4 quantization of a 2D bf16 matrix (grouped on axis 1)."""
    src = src_bf16.astype(np.float32).ravel()
    abs_src = _bf16(np.abs(src))
    mc = _bf16(abs_src.reshape(-1, GP4_SIZE).max(axis=1))
    mb = _bf16(abs_src.reshape(-1, GP8_SIZE).max(axis=1))
    ma = _bf16(abs_src.reshape(-1, GP64_SIZE).max(axis=1))

    ea_codes = bf16_to_e6m2(ma)
    ea_rec = e6m2_code_to_reciprocal_bf16(ea_codes)
    ea_rec_f32 = ea_rec.astype(np.float32)

    ea_rec_per8 = np.repeat(ea_rec_f32, GP64_SIZE // GP8_SIZE)
    eb_tmp = _bf16(mb * ea_rec_per8)
    eb_bits = (eb_tmp >= 4.0).astype(np.uint8)
    eb_rec = np.where(eb_bits == 1, 0.5, 1.0)

    ea_rec_per4 = np.repeat(ea_rec_f32, GP64_SIZE // GP4_SIZE)
    eb_rec_per4 = np.repeat(eb_rec, GP8_SIZE // GP4_SIZE)
    ec_tmp_0 = _bf16(mc * ea_rec_per4)
    ec_tmp_1 = _bf16(ec_tmp_0 * eb_rec_per4)
    ec_bits = (ec_tmp_1 >= 2.0).astype(np.uint8)
    ec_rec = np.where(ec_bits == 1, 0.5, 1.0)

    ebc_rec = _bf16(eb_rec_per4 * ec_rec)
    scale = _bf16(ea_rec_per4 * ebc_rec)

    scale_per_elem = np.repeat(scale, GP4_SIZE)
    scaled_src = _bf16(src * scale_per_elem)
    fp4_codes = bf16_to_e1m2_hif4(scaled_src)
    return {"ea": ea_codes, "eb": eb_bits, "ec": ec_bits, "fp4_codes": fp4_codes}


def dequantize_for_matmul(bf16_mat):
    """Reconstruct fp32 values from HiF4 codes + Ea/Eb/Ec, for the golden matmul."""
    res = hif4_quantize(bf16_mat)
    fp4_codes = res["fp4_codes"]
    n_elem = bf16_mat.size
    ea_vals = e6m2_code_to_value(res["ea"]).astype(np.float32)
    ea_per_elem = np.repeat(ea_vals, GP64_SIZE)[:n_elem]
    eb_per_elem = np.repeat(res["eb"], GP8_SIZE)[:n_elem]
    eb_factor = np.where(eb_per_elem == 1, 2.0, 1.0).astype(np.float32)
    ec_per_elem = np.repeat(res["ec"], GP4_SIZE)[:n_elem]
    ec_factor = np.where(ec_per_elem == 1, 2.0, 1.0).astype(np.float32)
    sign = (fp4_codes >> 3) & 1
    mag_code = fp4_codes & 0x07
    mag = E1M2_VALUES[mag_code].astype(np.float32)
    fp4_vals = np.where(sign == 1, -mag, mag)
    dequant = (fp4_vals * ea_per_elem).astype(bfloat16).astype(np.float32)
    dequant = (dequant * eb_factor).astype(bfloat16).astype(np.float32)
    dequant = (dequant * ec_factor).astype(bfloat16).astype(np.float32)
    return dequant.reshape(bf16_mat.shape).astype(bfloat16)


def pack_fp4_nd(fp4_codes_1d):
    """Pack two 4-bit codes per byte, low nibble first (ND order)."""
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    out = bytearray()
    for i in range(0, len(flat), 2):
        out.append(int(flat[i]) | (int(flat[i + 1]) << 4))
    return bytes(out)


def pack_bits_lsb(bits):
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


def _build_hif4_scale_patch_layout(ea, eb, ec, rows, cols):
    """Fractalize flat Ea/Eb/Ec into the on-GM [rows/16][cols/64][16,4]=64B cell layout."""
    row_fractals = (rows + 15) // 16
    k_groups = cols // 64
    num_groups = len(ea)
    out = np.zeros(row_fractals * k_groups * 64, dtype=np.uint8)
    view = out.reshape(row_fractals, k_groups, 2, 16, 2)
    for rf in range(row_fractals):
        for kg in range(k_groups):
            for r in range(16):
                g_lin = (rf * 16 + r) * k_groups + kg
                if g_lin >= num_groups:
                    continue
                view[rf, kg, 0, r, 0] = ea[g_lin]
                view[rf, kg, 0, r, 1] = eb[g_lin]
                view[rf, kg, 1, r, 0] = ec[g_lin * 2]
                view[rf, kg, 1, r, 1] = ec[g_lin * 2 + 1]
    return out.tobytes()


def build_hif4_scale_a_zz(a_bf16):
    valid_m, valid_k = a_bf16.shape
    res = hif4_quantize(a_bf16)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_m, valid_k)


def build_hif4_scale_b_nn(b_bf16):
    valid_k, valid_n = b_bf16.shape
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_n, valid_k)


def quantize_to_hif4_a(a_bf16):
    res = hif4_quantize(a_bf16)
    return pack_fp4_nd(res["fp4_codes"]), build_hif4_scale_a_zz(a_bf16)


def quantize_to_hif4_b(b_bf16):
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    fp4_codes = res["fp4_codes"].reshape(b_t.shape).T.copy()
    return pack_fp4_nd(fp4_codes.ravel()), build_hif4_scale_b_nn(b_bf16)


def make_bf16_matrix(rows, cols):
    rng = np.random.default_rng(19)
    total = rows * cols
    base = rng.uniform(-1.0, 1.0, size=total).astype(np.float32)
    scales = np.ones(total, dtype=np.float32)
    for g in range((total + 63) // 64):
        s, e = g * 64, min(g * 64 + 64, total)
        scales[s:e] = rng.uniform(0.5, 10.0)
    return (base * scales).reshape(rows, cols).astype(bfloat16)


def main():
    m, k, n = 2048, 2048, 2048

    # A: row-major (m x k), groups along k.
    a_bf16 = make_bf16_matrix(m, k)
    # B: row-major (k x n), groups along k (via transpose in the quantizer).
    b_bf16 = make_bf16_matrix(k, n)

    a_data, a_scale = quantize_to_hif4_a(a_bf16)
    b_data, b_scale = quantize_to_hif4_b(b_bf16)

    a_deq = dequantize_for_matmul(a_bf16).astype(np.float32)
    b_deq = dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)
    golden = (a_deq @ b_deq).astype(np.float32).astype(bfloat16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    with open("./input/x1_gm.bin", "wb") as f:
        f.write(a_data)
    with open("./input/x2_gm.bin", "wb") as f:
        f.write(b_data)
    with open("./input/x1_scale_gm.bin", "wb") as f:
        f.write(a_scale)
    with open("./input/x2_scale_gm.bin", "wb") as f:
        f.write(b_scale)
    with open("./output/golden.bin", "wb") as f:
        f.write(golden.tobytes())

    print(
        f"m={m} k={k} n={n}: "
        f"a_data={len(a_data)}B a_scale={len(a_scale)}B b_data={len(b_data)}B b_scale={len(b_scale)}B "
        f"golden={golden.nbytes}B"
    )


if __name__ == "__main__":
    main()
