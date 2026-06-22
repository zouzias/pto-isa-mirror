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

"""Validate UB/L1 usage for each NSA branch (worst case across cmp/slc/win)."""
import argparse
import json
import sys
from pathlib import Path

MAX_UB_BYTES = 256 * 1024
MAX_L1_BYTES = 512 * 1024


def compute_l1_usage_dn(cube_s0, cube_s1, head_size):
    q_buffers, k_buffers, p_buffers, v_buffers = 1, 2, 2, 2
    q_bytes = head_size * cube_s0 * 2 * q_buffers
    k_bytes = cube_s1 * head_size * 2 * k_buffers
    p_bytes = cube_s0 * cube_s1 * 2 * p_buffers
    v_bytes = cube_s1 * head_size * 2 * v_buffers
    return q_bytes + k_bytes + p_bytes + v_bytes


def compute_ub_usage_dn(cube_s0, cube_s1, head_size, tile_s1, cv_fifo_size=8):
    vec_cores = 2
    k_tile_factor = tile_s1 // cube_s1
    vec_s0 = cube_s0 // vec_cores // k_tile_factor
    vec_gu_rows = cube_s0 // vec_cores
    subblock_rows = cube_s0 // vec_cores
    tile_data_f_bytes = tile_s1 * vec_s0 * 4
    reduce_tile_f_bytes = subblock_rows * 4
    tile_data_h_bytes = tile_s1 * vec_s0 * 2
    tile_out_gu_bytes = vec_gu_rows * head_size * 4
    nz_buf_rows = cube_s1 + 1
    tile_data_h_nz_bytes = nz_buf_rows * vec_s0 * 2
    src_bytes = tile_data_f_bytes * 2
    pv_bytes = tile_out_gu_bytes * 2
    xexp_bytes = tile_data_h_bytes * 2
    return (
        src_bytes + pv_bytes + xexp_bytes + (reduce_tile_f_bytes * (3 + cv_fifo_size))
        + tile_out_gu_bytes + tile_data_h_nz_bytes
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dn"], default="dn")
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    ok_all = True
    print("=" * 100)
    print("Buffer Usage Validation (Mode: NSA DN)")
    print("=" * 100)
    for case in cases:
        cube_s0 = case["cube_s0"]
        cube_s1 = case["cube_s1"]
        head = case["head_size"]
        tile_s1 = case["tile_s1"]
        max_s1 = max(case["s1_cmp"], case["s1_slc"], case["s1_win"])
        l1 = compute_l1_usage_dn(cube_s0, cube_s1, head)
        ub = compute_ub_usage_dn(cube_s0, cube_s1, head, min(tile_s1, max_s1))
        l1_ok = l1 <= MAX_L1_BYTES
        ub_ok = ub <= MAX_UB_BYTES
        ok = l1_ok and ub_ok
        ok_all = ok_all and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"       UB: {ub/1024:.1f} KB {'OK' if ub_ok else 'OVERFLOW'}")
        print(f"       L1: {l1/1024:.1f} KB {'OK' if l1_ok else 'OVERFLOW'}")
    print("=" * 100)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
