#!/usr/bin/env python3
# coding=utf-8
"""Reference stats for BF16 radix TopK (MSB/LSB, TopK pipeline).

This adapts topk_ub/scripts/radix_topk_golden_stats.py for BF16 score inputs.
Key point: BF16 bits must be converted to order-preserving unsigned keys first.
"""

import os
import sys

import numpy as np

TOPK = int(os.getenv("INDEXER_TOPK", "512"))
BS_HINT = int(os.getenv("INDEXER_TEST_BS", "2"))
MAX_PRINT_INDICES = 64


def winner_bin_min_ge(cum_hist: np.ndarray, thr: int) -> int:
    for b in range(256):
        if int(cum_hist[b]) >= thr:
            return b
    return 255


def winner_bin_min_gt(cum_hist: np.ndarray, thr: int) -> int:
    for b in range(256):
        if int(cum_hist[b]) > thr:
            return b
    return 255


def ordered_bf16_key(bits_u16: np.ndarray) -> np.ndarray:
    """Map BF16 bit pattern to uint16 ordered key (ascending by numeric value)."""
    bits = bits_u16.astype(np.uint16)
    sign = (bits & np.uint16(0x8000)) != 0
    out = np.empty_like(bits, dtype=np.uint16)
    out[sign] = np.uint16((~bits[sign]) & np.uint16(0xFFFF))
    out[~sign] = np.uint16(bits[~sign] ^ np.uint16(0x8000))
    return out


def load_bf16_bits(root: str) -> np.ndarray:
    """Load BF16 bits from score_bf16.bin, else derive from golden_score.bin(float32)."""
    score_bf16_path = os.path.join(root, "output", "score_bf16.bin")
    golden_score_path = os.path.join(root, "output", "golden_score.bin")
    if os.path.isfile(score_bf16_path):
        return np.fromfile(score_bf16_path, dtype=np.dtype("<u2"))
    if os.path.isfile(golden_score_path):
        f32 = np.fromfile(golden_score_path, dtype=np.dtype("<f4"))
        bits32 = f32.view(np.uint32)
        return (bits32 >> 16).astype(np.uint16)
    print(
        f"Missing both {score_bf16_path} and {golden_score_path}",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bf16_bits = load_bf16_bits(root)
    total_len = int(bf16_bits.size)
    if total_len == 0:
        print("No keys loaded.", file=sys.stderr)
        return 1
    n_hint = int(os.getenv("INDEXER_TEST_N", "0"))
    if n_hint > 0 and total_len % n_hint == 0:
        bs = total_len // n_hint
        n = n_hint
    elif BS_HINT > 0 and total_len % BS_HINT == 0:
        bs = BS_HINT
        n = total_len // BS_HINT
    else:
        bs = 1
        n = total_len
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    print("=== BF16 Radix TopK Golden Stats ===")
    print(f"Total keys = {total_len}, inferred BS = {bs}, N = {n}")
    if n_hint == 0:
        print("Note: INDEXER_TEST_N is not set; using inferred BS/N.")
    print(f"TopK = min(INDEXER_TOPK, N) = {min(TOPK, n)}")

    for b in range(bs):
        bits_b = bf16_bits[b * n : (b + 1) * n]
        keys = ordered_bf16_key(bits_b)
        topk = min(TOPK, n)
        thr_msb = n - topk

        msb = (keys.astype(np.uint32) >> 8) & 0xFF
        lsb = keys.astype(np.uint32) & 0xFF

        counts_msb = np.bincount(msb, minlength=256).astype(np.uint64)
        c_msb = np.cumsum(counts_msb).astype(np.uint64)
        msb_winner = winner_bin_min_ge(c_msb, thr_msb)
        c_below_winner = int(c_msb[msb_winner - 1]) if msb_winner > 0 else 0
        remain_k = int(np.uint32(thr_msb - c_below_winner))
        msb_winner_bin_count = int(counts_msb[msb_winner])

        mask = msb == msb_winner
        lsb_keys = lsb[mask]
        counts_lsb = np.bincount(lsb_keys.astype(np.int64), minlength=256).astype(np.uint64)
        c_lsb = np.cumsum(counts_lsb).astype(np.uint64)
        lsb_winner = winner_bin_min_gt(c_lsb, remain_k)
        lsb_hist_total = int(c_lsb[-1])

        packed_threshold = int((msb_winner << 8) | lsb_winner)
        thr_u16 = np.uint16(packed_threshold)
        idx_gt_thr = np.flatnonzero(keys > thr_u16).astype(np.int64)
        idx_eq_thr = np.flatnonzero(keys == thr_u16).astype(np.int64)

        print(f"\n[b={b}] N = {n}, TopK = {topk}")
        print(f"[b={b}] thr_msb = N - TopK = {thr_msb}")
        print(f"[b={b}] msb_winner(min C>=thr) = {msb_winner} (0x{msb_winner:02x})")
        print(f"[b={b}] msb_winner_bin_count = {msb_winner_bin_count}")
        print(f"[b={b}] remain_k = {remain_k}")
        print(f"[b={b}] lsb_winner(min C>remain_k) = {lsb_winner} (0x{lsb_winner:02x})")
        print(f"[b={b}] lsb_hist_total = {lsb_hist_total}")
        print(f"[b={b}] packed_threshold = 0x{packed_threshold:04x} ({packed_threshold})")
        print(
            f"[b={b}] |keys > thr| = {idx_gt_thr.size}, |keys == thr| = {idx_eq_thr.size}, sum = {idx_gt_thr.size + idx_eq_thr.size}"
        )

        gt_show = idx_gt_thr[:MAX_PRINT_INDICES].tolist()
        eq_show = idx_eq_thr[:MAX_PRINT_INDICES].tolist()
        gt_suffix = f" ... ({idx_gt_thr.size} total)" if idx_gt_thr.size > MAX_PRINT_INDICES else ""
        eq_suffix = f" ... ({idx_eq_thr.size} total)" if idx_eq_thr.size > MAX_PRINT_INDICES else ""
        print(f"[b={b}] indices > thr: {gt_show}{gt_suffix}")
        print(f"[b={b}] indices == thr: {eq_show}{eq_suffix}")

        out_gt = os.path.join(out_dir, f"indices_gt_thr_bf16_b{b}.txt")
        out_eq = os.path.join(out_dir, f"indices_eq_thr_bf16_b{b}.txt")
        np.savetxt(out_gt, idx_gt_thr, fmt="%d")
        np.savetxt(out_eq, idx_eq_thr, fmt="%d")
        print(f"[b={b}] Wrote: {out_gt}")
        print(f"[b={b}] Wrote: {out_eq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
