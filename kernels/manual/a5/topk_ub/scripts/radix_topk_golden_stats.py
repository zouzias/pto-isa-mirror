#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reference stats for MSB/LSB radix TopK (Python uses MSB winner bin directly for LSB histogram; no find-1).
#
# MSB: min { b : C[b] >= thr_msb } (draft FindWinnerBucketDescending: CmpMode::GE). thr_msb = N - TopK.
# remainK = thr_msb - C[winner-1]: keys with MSB < winner fully counted; refine inside MSB==winner via LSB hist.
#
# LSB: min { b : C_lsb[b] > remain_k } (draft LsbHistGeRemainKToLanes: CmpMode::GT on cumulative vs remain_k).
# K-th largest (this script): (msb_winner << 8) | lsb_winner == packed_threshold.

import os
import sys

import numpy as np

# keys length N is always taken from input/keys.bin (any size, e.g. 64K).
TOPK = 512
# When printing index lists, show at most this many (full lists are written to output/*.txt).
MAX_PRINT_INDICES = 64


def winner_bin_min_ge(C: np.ndarray, thr: int) -> int:
    """Smallest b with C[b] >= thr (C ascending cumulative, length 256). MSB winner; matches CmpMode::GE."""
    for b in range(256):
        if int(C[b]) >= thr:
            return b
    return 255


def winner_bin_min_gt(C: np.ndarray, thr: int) -> int:
    """Smallest b with C[b] > thr. LSB winner; matches CmpMode::GT on cumulative vs threshold."""
    for b in range(256):
        if int(C[b]) > thr:
            return b
    return 255


def winner_bin_max_lt(C: np.ndarray, thr: int) -> int:
    """Largest b with C[b] < thr; returns -1 if C[0] >= thr."""
    last = -1
    for b in range(256):
        if int(C[b]) < thr:
            last = b
    return last


def get_k_index_thistogram(cumulative_hist_asc: np.ndarray, k: int) -> int:
    """Same as tests/npu/a5/.../thistogram/gen_data.py get_k_index (LSB golden MSB pick)."""
    total = int(cumulative_hist_asc[-1])
    cumulative_hist_desc = total - np.concatenate(([0], cumulative_hist_asc[:-1])).astype(np.uint64)
    valid_bins = np.flatnonzero(cumulative_hist_desc >= k)
    if valid_bins.size == 0:
        return 0
    return int(valid_bins[-1])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys_path = os.path.join(root, "input", "keys.bin")
    if not os.path.isfile(keys_path):
        print(f"Missing {keys_path}; run: python3 scripts/gen_data.py", file=sys.stderr)
        sys.exit(1)

    keys = np.fromfile(keys_path, dtype=np.dtype("<u2"))
    n = int(keys.size)
    topk = TOPK

    msb = (keys.astype(np.uint32) >> 8) & 0xFF
    lsb = keys.astype(np.uint32) & 0xFF

    counts_msb = np.bincount(msb, minlength=256).astype(np.uint64)
    C_msb = np.cumsum(counts_msb).astype(np.uint64)
    assert int(C_msb[255]) == n

    thr_msb = n - topk
    msb_winner_find = winner_bin_min_ge(C_msb, thr_msb)
    msb_last_below_thr = winner_bin_max_lt(C_msb, thr_msb)
    # LSB pass: MSB == winner bin (not winner-1). remain_k uses cumulative strictly below winner.
    msb_idx_lsb_filter = int(msb_winner_find)
    cw_below_winner = int(C_msb[msb_winner_find - 1]) if msb_winner_find > 0 else 0
    cw_at_winner = int(C_msb[msb_idx_lsb_filter])
    remain_k = thr_msb - cw_below_winner
    msb_k_index_hist = get_k_index_thistogram(C_msb, topk)

    print("=== Golden / CPU reference (LSB hist MSB == winner bin; gate C_lsb[b] > remain_k, draft CmpMode::GT) ===")
    print(f"N = {n}, TopK = {topk}")
    print()
    print("--- MSB (THISTOGRAM<true>, ascending C[b]) ---")
    print(f"  thr_msb = N - TopK = {thr_msb}")
    print(
        f"  msb_last_below_thr = max{{b : C[b] < thr_msb}} = {msb_last_below_thr} "
        f"(C[{msb_last_below_thr}] = {int(C_msb[msb_last_below_thr])} < {thr_msb})"
    )
    print(f"  msb_winner_find = min{{b : C[b] >= thr_msb}} = {msb_winner_find}")
    print(
        f"  msb_idx_lsb_filter = winner bin (same as msb_winner_find) = {msb_idx_lsb_filter} "
        f"(0x{msb_idx_lsb_filter:02x}) → LSB histogram uses MSB == this byte"
    )
    print(
        f"  get_k_index(C_msb, TopK) [thistogram/gen_data.py] = {msb_k_index_hist} "
        "(ST golden; may differ from winner bin)"
    )
    print(
        f"  C_msb[winner-1] = {cw_below_winner}  (keys with MSB < winner; used for remain_k)"
    )
    print(f"  C_msb[winner] = {cw_at_winner}  (includes winner MSB bucket)")
    print(f"  remain_k = thr_msb - C[winner-1] = {thr_msb} - {cw_below_winner} = {remain_k}")

    mask_msb_winner = msb == msb_idx_lsb_filter
    lsb_keys = lsb[mask_msb_winner]
    counts_lsb = np.bincount(lsb_keys.astype(np.int64), minlength=256).astype(np.uint64)
    C_lsb = np.cumsum(counts_lsb).astype(np.uint64)
    total_lsb = int(C_lsb[255])

    rk_u32 = int(np.uint32(remain_k))
    lsb_winner = winner_bin_min_gt(C_lsb, rk_u32)
    max_c = int(C_lsb[255])
    if max_c <= rk_u32:
        lsb_note = f" (no bin has C_lsb[b] > {rk_u32}; max C_lsb[255] = {max_c})"
    else:
        lsb_note = ""

    print()
    print(f"--- LSB (keys with MSB == {msb_idx_lsb_filter} == 0x{msb_idx_lsb_filter:02x}, cumulative C_lsb) ---")
    print(f"  total_lsb = key count in bucket = {total_lsb}")
    print(f"  gate: remain_k (u32) = {rk_u32}  →  lsb_winner = min{{b : C_lsb[b] > remain_k}} (draft CmpMode::GT)")
    print(f"  lsb_winner = {lsb_winner} (0x{lsb_winner:02x}){lsb_note}")

    sub = keys[mask_msb_winner]
    sub_desc = np.sort(sub)[::-1]
    if rk_u32 > 0 and len(sub_desc) >= rk_u32:
        lsb_of_rk_th = int(sub_desc[rk_u32 - 1] & 0xFF)
        print(f"  sanity: LSB of {rk_u32}-th largest in this bucket = {lsb_of_rk_th}")
    else:
        print(f"  sanity: (skip — need remain_k >= 1 and total_lsb >= remain_k for tail-in-bucket check)")

    gather_cmp_append_ub = int(msb_idx_lsb_filter)
    kth_key = int((gather_cmp_append_ub << 8) | int(lsb_winner))
    kth_lsb = int(lsb_winner)

    keys_sorted_desc = np.sort(keys)[::-1]
    kth_sorted = int(keys_sorted_desc[topk - 1])

    print()
    print("--- K-th largest (gather_cmp_append_ub << 8 | lsb_winner, same as packed_threshold) ---")
    print(
        f"  K-th largest key = (0x{gather_cmp_append_ub:02x} << 8) | 0x{kth_lsb:02x} = 0x{kth_key:04x} ({kth_key})"
    )
    sort_note = "(match)" if kth_sorted == kth_key else "(diff — check radix vs golden)"
    print(f"  sanity: K-th largest from full sort = 0x{kth_sorted:04x}  {sort_note}")
    print()
    print("--- Packed threshold (uint16; MSB byte = winner bin) ---")
    packed = kth_key
    msb_packed = gather_cmp_append_ub
    print(f"  packed_threshold = (0x{msb_packed:02x} << 8) | 0x{lsb_winner:02x} = 0x{packed:04x} ({packed})")

    # uint16 order matches golden multiset / main.cpp; int16 reinterpret would change counts for high keys.
    thr_u16 = np.uint16(kth_key & 0xFFFF)
    idx_gt_thr = np.flatnonzero(keys > thr_u16).astype(np.int64)
    idx_eq_thr = np.flatnonzero(keys == thr_u16).astype(np.int64)
    n_gt = int(idx_gt_thr.size)
    n_eq = int(idx_eq_thr.size)
    print()
    print("--- Input vs packed_threshold (uint16 key > / ==; multiset sanity) ---")
    print(f"  thr_u16 = 0x{int(thr_u16):04x}")
    print(f"  |{{i : keys[i] > thr_u16}}| = {n_gt}")
    print(f"  |{{i : keys[i] == thr_u16}}| = {n_eq}")
    print(f"  n_gt + n_eq = {n_gt + n_eq}  (TopK = {topk})")
    print()
    gt_show = idx_gt_thr[:MAX_PRINT_INDICES].tolist()
    eq_show = idx_eq_thr[:MAX_PRINT_INDICES].tolist()
    gt_suffix = f" ... ({n_gt} total)" if n_gt > MAX_PRINT_INDICES else ""
    eq_suffix = f" ... ({n_eq} total)" if n_eq > MAX_PRINT_INDICES else ""
    print(f"  indices i with keys[i] > thr_u16 (n_gt={n_gt}):")
    print(f"  {gt_show}{gt_suffix}")
    print()
    print(f"  indices i with keys[i] == thr_u16 (n_eq={n_eq}):")
    print(f"  {eq_show}{eq_suffix}")

    os.makedirs(os.path.join(root, "output"), exist_ok=True)
    out_gt = os.path.join(root, "output", "indices_gt_thr_u16.txt")
    out_eq = os.path.join(root, "output", "indices_eq_thr_u16.txt")
    np.savetxt(out_gt, idx_gt_thr, fmt="%d")
    np.savetxt(out_eq, idx_eq_thr, fmt="%d")
    print()
    print(f"  Wrote: {out_gt}  (n_gt lines)")
    print(f"  Wrote: {out_eq}  (n_eq lines)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
