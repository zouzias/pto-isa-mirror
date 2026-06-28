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

"""
Validate UB and L1 buffer usage for MLA DN cases.
"""
import argparse
import json
import sys
from pathlib import Path

MAX_UB_BYTES = 256 * 1024
MAX_L1_BYTES = 512 * 1024


def compute_l1_usage_dn(cube_s0: int, cube_s1: int, head_size: int, kv_latent_dim: int, use_ub_to_l1: bool = False, qk_preload: int = 2) -> dict:
    q_mat_tn_buffers = 1
    c_kv_mat_tn_buffers = qk_preload + 1
    w_uv_mat_tn_buffers = 1
    p_mat_tn_buffers = 3 if use_ub_to_l1 else 2
    v_mat_tn_buffers = 2

    tile_mat_q_bytes = kv_latent_dim * cube_s0 * 2
    tile_mat_c_kv_bytes = cube_s1 * kv_latent_dim * 2
    tile_mat_w_uv_bytes = kv_latent_dim * head_size * 2
    tile_mat_p_bytes = cube_s0 * cube_s1 * 2
    tile_mat_v_bytes = cube_s1 * head_size * 2

    total_bytes = (
        tile_mat_q_bytes * q_mat_tn_buffers
        + tile_mat_c_kv_bytes * c_kv_mat_tn_buffers
        + tile_mat_w_uv_bytes * w_uv_mat_tn_buffers
        + tile_mat_p_bytes * p_mat_tn_buffers
        + tile_mat_v_bytes * v_mat_tn_buffers
    )

    return {
        "total_bytes": total_bytes,
        "fits_in_l1": total_bytes <= MAX_L1_BYTES,
    }


def compute_ub_usage_dn(cube_s0: int, cube_s1: int, head_size: int, tile_s1: int, cv_fifo_size: int) -> dict:
    VEC_CORES = 2
    src_vec_tn_buffers = 2
    xexp_vec_tn_buffers = 2
    out_o_tile_n_buffers = 2

    k_tile_factor = tile_s1 // cube_s1
    vec_s0 = cube_s0 // VEC_CORES // k_tile_factor
    vec_gu_rows = cube_s0 // VEC_CORES
    subblock_rows = cube_s0 // VEC_CORES

    tile_data_f_bytes = tile_s1 * vec_s0 * 4
    reduce_tile_f_bytes = subblock_rows * 4
    tile_data_h_bytes = tile_s1 * vec_s0 * 2
    tile_out_gu_bytes = vec_gu_rows * head_size * 4

    nz_buf_rows = cube_s1 + 1
    tile_data_h_nz_bytes = nz_buf_rows * vec_s0 * 2

    src_bytes = tile_data_f_bytes * src_vec_tn_buffers
    pv_bytes = tile_out_gu_bytes * out_o_tile_n_buffers
    xexp_bytes = tile_data_h_bytes * xexp_vec_tn_buffers
    exp_max_buffers = cv_fifo_size

    total_bytes = src_bytes + pv_bytes + xexp_bytes + \
                  (reduce_tile_f_bytes * (3 + exp_max_buffers)) + tile_out_gu_bytes + tile_data_h_nz_bytes

    return {
        "total_bytes": total_bytes,
        "fits_in_ub": total_bytes <= MAX_UB_BYTES,
    }


def format_size(bytes_val: int) -> str:
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB ({bytes_val} bytes)"
    return f"{bytes_val} bytes"


def main():
    parser = argparse.ArgumentParser(description="Validate buffer usage for MLA cases")
    parser.add_argument("--mode", choices=["dn"], default="dn")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cv_fifo_size", type=int, default=8)
    parser.add_argument("--fifo_mode", type=int, default=1, help="FIFO_MODE: 0=ALL_GM, 1=ALL_UB, 2=QK_PV_UB_ONLY")
    parser.add_argument("--qk-preload", type=int, default=2, help="QK preload depth (default: 2)")
    args = parser.parse_args()
    use_ub_to_l1 = (args.fifo_mode == 1)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[ERROR] Cases file not found: {cases_path}")
        sys.exit(1)

    with open(cases_path, "r") as f:
        cases = json.load(f)

    all_pass = True
    failed_cases = []

    print("=" * 100)
    print("Buffer Usage Validation (Mode: MLA DN)")
    print("=" * 100)
    print(f"Max UB: {format_size(MAX_UB_BYTES)}")
    print(f"Max L1: {format_size(MAX_L1_BYTES)}")
    print("-" * 100)

    for case in cases:
        name = case["name"]
        cube_s0 = case["cube_s0"]
        cube_s1 = case["cube_s1"]
        head_size = case["head_size"]
        kv_latent_dim = case["kv_latent_dim"]
        tile_s1 = case["tile_s1"]

        l1_result = compute_l1_usage_dn(cube_s0, cube_s1, head_size, kv_latent_dim, use_ub_to_l1, args.qk_preload)
        ub_result = compute_ub_usage_dn(cube_s0, cube_s1, head_size, tile_s1, args.cv_fifo_size)

        ub_ok = ub_result["fits_in_ub"]
        l1_ok = l1_result["fits_in_l1"]
        status = "PASS" if (ub_ok and l1_ok) else "FAIL"
        if not (ub_ok and l1_ok):
            all_pass = False
            failed_cases.append(name)

        print(f"[{status}] {name}")
        print(f"       UB: {format_size(ub_result['total_bytes']):<20} {'OK' if ub_ok else 'OVERFLOW'}")
        print(f"       L1: {format_size(l1_result['total_bytes']):<20} {'OK' if l1_ok else 'OVERFLOW'}")

    print("=" * 100)
    if all_pass:
        print("[INFO] All cases passed buffer usage validation")
        sys.exit(0)

    print(f"[ERROR] {len(failed_cases)} case(s) failed buffer usage validation:")
    for name in failed_cases:
        print(f"  - {name}")
    sys.exit(1)


if __name__ == "__main__":
    main()
