#!/usr/bin/env python3
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

"""Reference radix stages for draft.cpp: MSB winner, remainK, LSB winner (TopK largest uint16 keys)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

N_DEFAULT = 8192
TOPK_DEFAULT = 512


def cumulative_asc(byte_u8: np.ndarray) -> np.ndarray:
    c = np.bincount(byte_u8.astype(np.int64), minlength=256).astype(np.uint64)
    return np.cumsum(c).astype(np.uint64)


def find_winner_min_bin_ge(cum_asc: np.ndarray, thr: int) -> int:
    """Smallest bin b with C[b] >= thr (FindWinnerBucketDescending: TSELS lane = b when C[b] >= thr)."""
    for b in range(256):
        if cum_asc[b] >= thr:
            return b
    return 0


def find_winner_min_bin_ge_u32(cum_asc: np.ndarray, thr_u32: int) -> int:
    """Smallest bin b with C[b] >= thr (unsigned); matches LsbHistGeRemainKToLanes CmpMode::GE."""
    t = int(np.uint32(thr_u32))
    for b in range(256):
        if int(cum_asc[b]) >= t:
            return b
    return 255


def msb_broadcast_u8_from_find_bin(b_find: int) -> int:
    """Match WinnerBinU8FromSelsMin: TROWMIN then TADDS(-1); if result > 256 (wrap from b==0) TSEL -> 0."""
    # b_find is u32 row min; TADDS(-1): b_find==0 -> 0xffffffff -> TCMPS GT 256 -> mask true -> selOut 0.
    if b_find == 0:
        return 0
    return int(b_find - 1)


def radix_stages(keys: np.ndarray, topk: int) -> dict:
    n = int(keys.shape[0])
    msb = ((keys.astype(np.uint32) >> 8) & 0xFF).astype(np.uint8)
    lsb = (keys.astype(np.uint32) & 0xFF).astype(np.uint8)

    c_msb = cumulative_asc(msb)
    thr_msb = n - topk
    msb_find_b = find_winner_min_bin_ge(c_msb, thr_msb)
    # draft.cpp: idxFilter / TGATHER C[w] use WinnerBinU8FromSelsMin output = (find_bin - 1), not find_bin.
    msb_idx_u8 = msb_broadcast_u8_from_find_bin(msb_find_b)
    cw = int(c_msb[msb_idx_u8])
    sum_above = n - cw
    # draft.cpp RemainKMsbFromTiles: TSUB(remainK, thr_msb, C[w]) => (N-TopK) - C[w].
    remain_k = int(thr_msb) - cw

    mask_row = msb == msb_idx_u8
    lsb_sub = lsb[mask_row]
    c_lsb = cumulative_asc(lsb_sub)
    total_lsb = int(c_lsb[255])
    rk_u32 = int(np.uint32(remain_k))
    # draft.cpp LsbWinnerBinFromChistTiles: TCMPS GE(chistLSB, remainKTile) + TCI + TSELS + ROWMIN.
    lsb_winner = find_winner_min_bin_ge_u32(c_lsb, rk_u32)

    packed = int(((msb_idx_u8 & 0xFF) << 8) | (lsb_winner & 0xFF)) & 0xFFFF
    return {
        "n": n,
        "topk": topk,
        "thr_msb": thr_msb,
        "msb_find_min_ge": msb_find_b,
        "msb_idx_filter_u8": msb_idx_u8,
        "C_msb_at_tgather_index": cw,
        "sum_keys_msb_gt_tgather_index": sum_above,
        "remain_k": remain_k,
        "total_lsb_hist255": total_lsb,
        "remain_k_u32": rk_u32,
        "lsb_winner_min_bin_ge": lsb_winner,
        "packed_threshold_u16": packed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Print MSB/remainK/LSB radix stages (matches draft.cpp logic).")
    p.add_argument(
        "keys_bin",
        nargs="?",
        default=None,
        help="Path to uint16 keys.bin (default: ../input/keys.bin from script dir)",
    )
    p.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    p.add_argument("--n", type=int, default=None, help="Expected key count (verify only)")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    topk_dir = os.path.dirname(script_dir)
    path = args.keys_bin or os.path.join(topk_dir, "input", "keys.bin")
    if not os.path.isfile(path):
        print(f"Missing keys file: {path}", file=sys.stderr)
        sys.exit(1)

    keys = np.fromfile(path, dtype=np.uint16)
    if args.n is not None and keys.size != args.n:
        print(f"Expected N={args.n}, got {keys.size}", file=sys.stderr)
        sys.exit(1)

    out = radix_stages(keys, args.topk)
    print(f"keys: {path}  N={out['n']}  TopK={out['topk']}")
    print(f"  thr_msb (N-TopK)     = {out['thr_msb']}")
    print(
        f"  MSB find (min b, C[b]>=thr) = {out['msb_find_min_ge']}  (0x{out['msb_find_min_ge']:02x})  "
        "[FindWinnerBucketDescending]"
    )
    print(
        f"  MSB idx / LSB filter (THISTOGRAM<false>) = {out['msb_idx_filter_u8']}  "
        f"(0x{out['msb_idx_filter_u8']:02x})  [WinnerBinU8FromSelsMin: TADDS(-1) on find bin]"
    )
    print(f"  C_msb[TGATHER idx]   = {out['C_msb_at_tgather_index']}  (RemainKMsbFromTiles reads this bin)")
    print(f"  count MSB > TGATHER idx = {out['sum_keys_msb_gt_tgather_index']}")
    print(
        f"  remainK              = {out['remain_k']}  (= thr_msb - C[TGATHER_idx] = {out['thr_msb']} - "
        f"{out['C_msb_at_tgather_index']})"
    )
    print(f"  LSB hist total C[255]= {out['total_lsb_hist255']}  (keys with MSB==idx_filter)")
    print(
        f"  LSB winner           = {out['lsb_winner_min_bin_ge']}  (0x{out['lsb_winner_min_bin_ge']:02x})  "
        f"[min b : C_lsb[b] >= remainK u32 {out['remain_k_u32']}]"
    )
    print(f"  packed threshold u16 = 0x{out['packed_threshold_u16']:04x} ({out['packed_threshold_u16']})")


if __name__ == "__main__":
    main()
