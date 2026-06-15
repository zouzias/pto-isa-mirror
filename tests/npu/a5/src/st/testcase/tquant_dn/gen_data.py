"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import numpy as np
import os
import math


def fp32_to_bf16_bits(x):
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16


def bf16_bits_to_fp32(bf16_bits):
    return np.frombuffer(bf16_bits.astype(np.uint16).tobytes(), dtype=np.float16).astype(np.float32)


def get_group_max_dn(src, group_size=32):
    M, N = src.shape
    hatM = M // group_size
    max_vals = np.zeros((hatM, N), dtype=np.float32)
    for rb in range(hatM):
        for c in range(N):
            max_vals[rb, c] = np.max(np.abs(src[rb * group_size : (rb + 1) * group_size, c]))
    return max_vals


def fp32_maxes_to_fp8(group_max, emax=8):
    max_bits = np.asarray(group_max, dtype=np.float32).view(np.uint32)
    exponent_b32 = (max_bits & 0x7F800000) >> 23
    e8m0 = exponent_b32.astype(np.int32) - emax
    e8m0 = np.clip(e8m0, 0, 254).astype(np.uint8)
    scale_exp = 254 - e8m0.astype(np.int32)
    scale_exp = np.clip(scale_exp, 0, 255).astype(np.uint32)
    scaling_bits = (scale_exp << 23).view(np.float32)
    nan_mask = exponent_b32 == 255
    e8m0[nan_mask] = 0xFF
    scaling_bits[nan_mask] = np.float32(np.nan)
    return e8m0, scaling_bits


def scale_data_dn(src, scaling, group_size=32):
    M, N = src.shape
    hatM = M // group_size
    result = np.zeros_like(src)
    for rb in range(hatM):
        for r in range(rb * group_size, (rb + 1) * group_size):
            result[r, :] = src[r, :] * scaling[rb, :]
    return result


def fp32_to_e4m3(x):
    x = np.asarray(x, dtype=np.float32)
    clipped = np.clip(x, -448.0, 448.0)
    u32 = clipped.view(np.uint32)
    sign = (u32 >> 31) & 1
    exp_u32 = (u32 >> 23) & 0xFF
    mant_u32 = u32 & 0x7FFFFF
    e4m3_exp = exp_u32.astype(np.int32) - 127 + 7
    e4m3_mant = mant_u32 >> 20
    e4m3_exp = np.clip(e4m3_exp, 0, 15)
    e4m3_val = (sign << 7) | (e4m3_exp << 3) | e4m3_mant
    zero_mask = (clipped == 0.0)
    e4m3_val[zero_mask] = 0
    return e4m3_val.astype(np.uint8)


def nd2nz_mxfp8(data_fp8, M, N):
    paddedRows16 = ((M + 15) // 16) * 16
    virtualRow = paddedRows16 + 1
    paddedCols = ((N + 31) // 32) * 32
    nColGroups = paddedCols // 32
    nz = np.zeros(virtualRow * paddedCols, dtype=np.int8)
    data_flat = data_fp8.reshape(-1) if data_fp8.ndim > 1 else data_fp8
    for cg in range(nColGroups):
        for r in range(paddedRows16):
            src_idx = r * paddedCols + cg * 32
            dst_idx = cg * virtualRow * 32 + r * 32
            if r < M:
                nz[dst_idx : dst_idx + 32] = data_flat[src_idx : src_idx + 32]
            else:
                nz[dst_idx : dst_idx + 32] = 0
    return nz


def dn2zz_e8m0(e8m0_dn, hat_m, N):
    P = hat_m // 2
    col_block_count = N // 16
    result = np.zeros(hat_m * N, dtype=np.uint8)
    out_idx = 0
    for cb in range(col_block_count):
        for p in range(P):
            for q in range(16):
                c = cb * 16 + q
                result[out_idx] = e8m0_dn[c * hat_m + 2 * p]
                out_idx += 1
                result[out_idx] = e8m0_dn[c * hat_m + 2 * p + 1]
                out_idx += 1
    return result


def quant_bf16_to_mxfp8_dn(src_bf16_fp32, M, N_pad):
    src_fp32 = src_bf16_fp32
    group_max = get_group_max_dn(src_fp32, group_size=32)
    e8m0, scaling = fp32_maxes_to_fp8(group_max)
    scaled = scale_data_dn(src_fp32, scaling, group_size=32)
    fp8 = fp32_to_e4m3(scaled).reshape(M, N_pad)

    paddedCols = ((N_pad + 31) // 32) * 32
    fp8_padded = np.zeros((M, paddedCols), dtype=np.int8)
    fp8_padded[:, :N_pad] = fp8
    fp8_nz = nd2nz_mxfp8(fp8_padded, M, N_pad)

    hatM = M // 32
    e8m0_dn = np.zeros(hatM * N_pad, dtype=np.uint8)
    for c in range(N_pad):
        for rb in range(hatM):
            e8m0_dn[c * hatM + rb] = e8m0[rb, c]

    e8_zz = dn2zz_e8m0(e8m0_dn, hatM, N_pad)
    return fp8_nz, e8_zz


CASE_PARAMS = [
    ("TQUANTDNTest.case_bf16_64x64", 64, 64),
    ("TQUANTDNTest.case_bf16_128x64", 128, 64),
    ("TQUANTDNTest.case_bf16_64x128", 64, 128),
    ("TQUANTDNTest.case_bf16_128x128", 128, 128),
    ("TQUANTDNTest.case_bf16_64x256", 64, 256),
    ("TQUANTDNTest.case_bf16_128x256", 128, 256),
]

GOLDEN_DIR = os.environ.get("PTO_GOLDEN_DIR", ".")


def gen_golden_data(case_name, M, N):
    N_pad = N
    src = np.random.randn(M, N_pad).astype(np.float32)
    bf16_bits = fp32_to_bf16_bits(src).reshape(M, N_pad)
    src_bf16_fp32 = bf16_bits_to_fp32(bf16_bits.flatten()).reshape(M, N_pad)

    fp8_nz, e8_zz = quant_bf16_to_mxfp8_dn(src_bf16_fp32, M, N_pad)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "input.bin"), "wb") as f:
        f.write(bf16_bits.reshape(-1).tobytes())
    with open(os.path.join(out_dir, "golden_fp8_nz.bin"), "wb") as f:
        f.write(fp8_nz.tobytes())
    with open(os.path.join(out_dir, "golden_e8_zz.bin"), "wb") as f:
        f.write(e8_zz.tobytes())


if __name__ == "__main__":
    np.random.seed(42)
    for case_name, M, N in CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data(case_name, M, N)
    print("Done.")